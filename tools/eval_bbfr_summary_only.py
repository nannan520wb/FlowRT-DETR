import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import YAMLConfig


def import_bbfr_metrics():
    candidates = [
        'tools.bbfr.bbfr_metric',
        'tools.bbfr.bbfr.bbfr_metric',
        'tools.bbfr_metric',
    ]
    last_error = None
    for name in candidates:
        try:
            module = __import__(name, fromlist=['compute_bbfr_det'])
            return module.compute_bbfr_det, module.compute_bbfr_gt, module.aggregate_bbfr
        except Exception as exc:
            last_error = exc
    raise ImportError(f'Cannot import BBFR metric functions. Last error: {last_error}')


compute_bbfr_det, compute_bbfr_gt, aggregate_bbfr = import_bbfr_metrics()


def parse_ua_detrac_filename(file_name):
    parts = file_name.replace('\\', '/').split('/')
    if len(parts) >= 2:
        video_id = parts[0]
        leaf = parts[-1]
    else:
        video_id = 'default'
        leaf = parts[0]
    match = re.search(r'(\d+)', leaf)
    frame_idx = int(match.group(1)) if match else 0
    return video_id, frame_idx


def load_checkpoint_state(path):
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'ema' in checkpoint and checkpoint['ema'] is not None:
            ema = checkpoint['ema']
            if isinstance(ema, dict):
                return ema.get('module', ema)
        if 'model' in checkpoint:
            return checkpoint['model']
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
    return checkpoint


def strip_module_prefix(state):
    return {k.replace('module.', ''): v for k, v in state.items()}


def rebuild_loader(loader, batch_size=-1, num_workers=-1):
    if batch_size < 1 and num_workers < 0:
        return loader
    from torch.utils.data import DataLoader
    return DataLoader(
        dataset=loader.dataset,
        batch_size=batch_size if batch_size >= 1 else loader.batch_size,
        shuffle=False,
        num_workers=num_workers if num_workers >= 0 else loader.num_workers,
        collate_fn=loader.collate_fn,
        drop_last=False,
    )


def unpack_batch(batch):
    data, targets = batch
    if isinstance(data, (tuple, list)) and len(data) == 3:
        samples, prev_samples, flow_img = data
        return samples, prev_samples, flow_img, targets
    return data, None, None, targets


def postprocess_to_list(results):
    if isinstance(results, tuple):
        labels_b, boxes_b, scores_b = results
        return [
            {'labels': labels_b[i], 'boxes': boxes_b[i], 'scores': scores_b[i]}
            for i in range(labels_b.shape[0])
        ]
    return results


def get_coco_from_loader(loader):
    dataset = loader.dataset
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    if hasattr(dataset, 'coco'):
        return dataset.coco
    raise AttributeError('Cannot find dataset.coco from val_dataloader.')


@torch.no_grad()
def run_inference_and_collect(cfg, device, score_thresh_keep=0.05, num_workers=-1, batch_size=-1):
    model = cfg.model
    postproc = cfg.postprocessor

    state = strip_module_prefix(load_checkpoint_state(cfg.resume))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'[BBFR] Loaded checkpoint: {cfg.resume}')
    print(f'[BBFR] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}')

    if hasattr(model, 'deploy'):
        model = model.deploy()
    if hasattr(postproc, 'deploy'):
        postproc = postproc.deploy()

    model = model.to(device).eval()
    postproc = postproc.to(device).eval() if isinstance(postproc, torch.nn.Module) else postproc

    val_loader = rebuild_loader(cfg.val_dataloader, batch_size=batch_size, num_workers=num_workers)
    coco_api = get_coco_from_loader(val_loader)

    video_dets = defaultdict(dict)
    video_gts = defaultdict(dict)

    n_imgs = 0
    for batch in val_loader:
        samples, prev_samples, flow_img, targets = unpack_batch(batch)
        samples = samples.to(device)
        prev_samples = prev_samples.to(device) if prev_samples is not None else None
        flow_img = flow_img.to(device) if flow_img is not None else None

        if prev_samples is not None and flow_img is not None:
            outputs = model(samples, x_prev=prev_samples, flow=flow_img)
        else:
            outputs = model(samples)

        orig_target_sizes = torch.stack([t['orig_size'].to(device) for t in targets], dim=0)
        results = postprocess_to_list(postproc(outputs, orig_target_sizes))

        for tgt, res in zip(targets, results):
            img_id = int(tgt['image_id'].item())
            file_name = coco_api.loadImgs(img_id)[0]['file_name']
            video_id, frame_idx = parse_ua_detrac_filename(file_name)

            scores = res['scores'].detach().cpu().numpy()
            keep = scores >= score_thresh_keep
            video_dets[video_id][frame_idx] = {
                'boxes': res['boxes'].detach().cpu().numpy()[keep],
                'labels': res['labels'].detach().cpu().numpy()[keep],
                'scores': scores[keep],
            }

            ann_ids = coco_api.getAnnIds(imgIds=img_id)
            anns = coco_api.loadAnns(ann_ids)
            gt_boxes, gt_labels, gt_tids = [], [], []
            for ann in anns:
                if ann.get('iscrowd', 0):
                    continue
                x, y, w, h = ann['bbox']
                gt_boxes.append([x, y, x + w, y + h])
                gt_labels.append(int(ann['category_id']))
                tid = ann.get('track_id', ann.get('instance_id', ann.get('object_id', -1)))
                gt_tids.append(int(tid) if tid is not None else -1)

            if gt_boxes:
                video_gts[video_id][frame_idx] = {
                    'boxes': np.asarray(gt_boxes, dtype=np.float32),
                    'labels': np.asarray(gt_labels, dtype=np.int64),
                    'track_ids': np.asarray(gt_tids, dtype=np.int64),
                }

            n_imgs += 1
            if n_imgs % 500 == 0:
                print(f'[BBFR] processed {n_imgs} frames...')

    print(f'[BBFR] Inference done. frames={n_imgs}, videos={len(video_dets)}')
    return dict(video_dets), dict(video_gts)


def keep_summary_keys(d, wanted_keys):
    if d is None:
        return None
    return {k: d[k] for k in wanted_keys if k in d}


def compute_summary(video_dets, video_gts, args):
    per_video_det = {}
    for vid, dets in video_dets.items():
        per_video_det[vid] = compute_bbfr_det(
            dets,
            score_thresh=args.score_thresh,
            iou_thresh=args.iou_thresh,
            max_lost=args.max_lost,
            min_track_len=args.min_track_len,
        )
    agg_det_raw = aggregate_bbfr(per_video_det, key='BBFR')
    bbfr_det = keep_summary_keys(agg_det_raw, [
        'BBFR_micro',
        'BBFR_macro',
        'total_flicker_events',
        'total_track_frames',
        'num_videos',
    ])

    bbfr_gt = None
    if not args.no_gt and video_gts:
        sample_video = next(iter(video_gts.values()))
        sample_frame = next(iter(sample_video.values())) if sample_video else None
        has_tid = sample_frame is not None and (sample_frame['track_ids'] >= 0).any()
        if has_tid:
            per_video_gt = {}
            for vid, dets in video_dets.items():
                if vid not in video_gts:
                    continue
                per_video_gt[vid] = compute_bbfr_gt(
                    dets,
                    video_gts[vid],
                    score_thresh=args.score_thresh,
                    iou_match=args.iou_thresh,
                )
            agg_gt_raw = aggregate_bbfr(per_video_gt, key='BBFR_GT')
            bbfr_gt = keep_summary_keys(agg_gt_raw, [
                'BBFR_GT_micro',
                'BBFR_GT_macro',
                'total_gt_flicker_events',
                'total_gt_frames',
                'num_videos',
            ])

    return bbfr_det, bbfr_gt


def main():
    parser = argparse.ArgumentParser(description='BBFR evaluation with summary-only JSON output.')
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-r', '--resume', required=True)
    parser.add_argument('--score-thresh', type=float, default=0.3)
    parser.add_argument('--iou-thresh', type=float, default=0.5)
    parser.add_argument('--max-lost', type=int, default=5)
    parser.add_argument('--min-track-len', type=int, default=3)
    parser.add_argument('--score-thresh-keep', type=float, default=0.05)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-workers', type=int, default=-1)
    parser.add_argument('--batch-size', type=int, default=-1)
    parser.add_argument('--no-gt', action='store_true')
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device.index if device.index is not None else 0)

    cfg = YAMLConfig(args.config, resume=args.resume)
    cfg.resume = args.resume
    video_dets, video_gts = run_inference_and_collect(
        cfg,
        device,
        score_thresh_keep=args.score_thresh_keep,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
    bbfr_det, bbfr_gt = compute_summary(video_dets, video_gts, args)

    out = {
        'config': args.config,
        'checkpoint': args.resume,
        'hparams': {
            'score_thresh': args.score_thresh,
            'iou_thresh': args.iou_thresh,
            'max_lost': args.max_lost,
            'min_track_len': args.min_track_len,
        },
        'BBFR_Det': bbfr_det,
        'BBFR_GT': bbfr_gt,
    }

    # Alias for papers that call the GT-trajectory metric BBFR-T.
    if bbfr_gt is not None:
        out['BBFR_T'] = {
            'BBFR_T_micro': bbfr_gt.get('BBFR_GT_micro'),
            'BBFR_T_macro': bbfr_gt.get('BBFR_GT_macro'),
            'total_gt_flicker_events': bbfr_gt.get('total_gt_flicker_events'),
            'total_gt_frames': bbfr_gt.get('total_gt_frames'),
            'num_videos': bbfr_gt.get('num_videos'),
        }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=lambda x: float(x) if hasattr(x, 'item') else str(x))

    print('\n[BBFR Summary]')
    print(f"BBFR-D micro: {bbfr_det.get('BBFR_micro'):.4f}")
    print(f"BBFR-D macro: {bbfr_det.get('BBFR_macro'):.4f}")
    if bbfr_gt is not None:
        print(f"BBFR-T/GT micro: {bbfr_gt.get('BBFR_GT_micro'):.4f}")
        print(f"BBFR-T/GT macro: {bbfr_gt.get('BBFR_GT_macro'):.4f}")
    print(f'Saved summary-only JSON to: {args.out_json}')


if __name__ == '__main__':
    main()