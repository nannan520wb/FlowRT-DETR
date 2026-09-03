import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_checkpoint_state(path):
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'ema' in checkpoint and isinstance(checkpoint['ema'], dict) and checkpoint['ema'] is not None:
            return checkpoint['ema'].get('module', checkpoint['ema'])
        if 'model' in checkpoint:
            return checkpoint['model']
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
    return checkpoint


def strip_module_prefix(state):
    return {k.replace('module.', ''): v for k, v in state.items()}


def xywh_to_xyxy(box):
    x, y, w, h = [float(v) for v in box[:4]]
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return [x1, y1, x2 - x1, y2 - y1]


def iou_xyxy(a, b):
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def category_names_from_coco(coco):
    cats = coco.get('categories', [])
    return {int(c['id']): c.get('name', str(c['id'])) for c in cats}


def load_gt_from_ann(ann_file):
    coco = load_json(ann_file)
    gt_by_img = defaultdict(list)
    for ann in coco.get('annotations', []):
        if ann.get('iscrowd', 0):
            continue
        if 'bbox' not in ann:
            continue
        gt_by_img[int(ann['image_id'])].append({
            'image_id': int(ann['image_id']),
            'category_id': int(ann['category_id']),
            'bbox_xyxy': xywh_to_xyxy(ann['bbox']),
            'matched': False,
        })
    img_ids = [int(img['id']) for img in coco.get('images', [])]
    return coco, gt_by_img, img_ids


def normalize_predictions(preds, score_thr, category_offset=0):
    if isinstance(preds, dict) and 'annotations' in preds:
        preds = preds['annotations']
    pred_by_img = defaultdict(list)
    for pred in preds:
        score = float(pred.get('score', 1.0))
        if score < score_thr:
            continue
        if 'bbox' not in pred:
            continue
        pred_by_img[int(pred['image_id'])].append({
            'image_id': int(pred['image_id']),
            'category_id': int(pred['category_id']) + category_offset,
            'bbox_xyxy': xywh_to_xyxy(pred['bbox']),
            'score': score,
        })
    for img_id in pred_by_img:
        pred_by_img[img_id].sort(key=lambda x: x['score'], reverse=True)
    return pred_by_img


def rebuild_loader(loader, batch_size=None, num_workers=None):
    if batch_size is None and num_workers is None:
        return loader
    return DataLoader(
        dataset=loader.dataset,
        batch_size=batch_size if batch_size is not None else loader.batch_size,
        shuffle=False,
        num_workers=num_workers if num_workers is not None else loader.num_workers,
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


@torch.no_grad()
def infer_predictions(args):
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    cfg = YAMLConfig(args.config, resume=args.resume)
    model = cfg.model
    postprocessor = cfg.postprocessor

    state = strip_module_prefix(load_checkpoint_state(args.resume))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'[PR] Loaded checkpoint: {args.resume}')
    print(f'[PR] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}')

    if not args.no_deploy and hasattr(model, 'deploy'):
        model = model.deploy()
    if not args.no_deploy and hasattr(postprocessor, 'deploy'):
        postprocessor = postprocessor.deploy()

    model = model.to(device).eval()
    postprocessor = postprocessor.to(device).eval() if isinstance(postprocessor, torch.nn.Module) else postprocessor
    loader = rebuild_loader(
        cfg.val_dataloader,
        batch_size=args.batch_size if args.batch_size > 0 else None,
        num_workers=args.num_workers if args.num_workers >= 0 else None,
    )
    print(f'[PR] val_loader: batch_size={loader.batch_size}, num_workers={loader.num_workers}')

    predictions = []
    n_images = 0
    for batch in loader:
        samples, prev_samples, flow_img, targets = unpack_batch(batch)
        samples = samples.to(device)
        prev_samples = prev_samples.to(device) if prev_samples is not None else None
        flow_img = flow_img.to(device) if flow_img is not None else None

        if prev_samples is not None and flow_img is not None:
            outputs = model(samples, x_prev=prev_samples, flow=flow_img)
        else:
            outputs = model(samples)

        orig_target_sizes = torch.stack([t['orig_size'].to(device) for t in targets], dim=0)
        results = postprocess_to_list(postprocessor(outputs, orig_target_sizes))

        for target, result in zip(targets, results):
            image_id = int(target['image_id'].item())
            scores = result['scores'].detach().cpu()
            labels = result['labels'].detach().cpu().to(torch.int64)
            boxes = result['boxes'].detach().cpu()
            keep = scores >= args.score_thr
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]
            if args.max_dets > 0 and scores.numel() > args.max_dets:
                top_scores, idx = torch.topk(scores, args.max_dets)
                scores = top_scores
                labels = labels[idx]
                boxes = boxes[idx]
            for label, box, score in zip(labels.tolist(), boxes.tolist(), scores.tolist()):
                predictions.append({
                    'image_id': image_id,
                    'category_id': int(label),
                    'bbox': xyxy_to_xywh(box),
                    'score': float(score),
                })
            n_images += 1
            if args.max_images > 0 and n_images >= args.max_images:
                break
        if n_images % args.print_freq == 0:
            print(f'[PR] processed {n_images} images, predictions={len(predictions)}')
        if args.max_images > 0 and n_images >= args.max_images:
            break
    return predictions


def evaluate_pr(gt_by_img, img_ids, pred_by_img, iou_thr):
    counts = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0, 'pred': 0})
    total = {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0, 'pred': 0}

    for img_id in img_ids:
        gts = [dict(g, matched=False) for g in gt_by_img.get(img_id, [])]
        preds = pred_by_img.get(img_id, [])

        for gt in gts:
            counts[gt['category_id']]['gt'] += 1
            total['gt'] += 1
        for pred in preds:
            counts[pred['category_id']]['pred'] += 1
            total['pred'] += 1

        for pred in preds:
            best_idx = -1
            best_iou = 0.0
            for idx, gt in enumerate(gts):
                if gt['matched']:
                    continue
                if gt['category_id'] != pred['category_id']:
                    continue
                iou = iou_xyxy(pred['bbox_xyxy'], gt['bbox_xyxy'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= iou_thr:
                gts[best_idx]['matched'] = True
                counts[pred['category_id']]['tp'] += 1
                total['tp'] += 1
            else:
                counts[pred['category_id']]['fp'] += 1
                total['fp'] += 1

        for gt in gts:
            if not gt['matched']:
                counts[gt['category_id']]['fn'] += 1
                total['fn'] += 1

    def finalize(d):
        tp, fp, fn = d['tp'], d['fp'], d['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            **d,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    per_class = {cat: finalize(dict(v)) for cat, v in sorted(counts.items())}
    micro = finalize(total)
    valid_classes = [v for v in per_class.values() if v['gt'] > 0]
    macro = {
        'precision': sum(v['precision'] for v in valid_classes) / len(valid_classes) if valid_classes else 0.0,
        'recall': sum(v['recall'] for v in valid_classes) / len(valid_classes) if valid_classes else 0.0,
        'f1': sum(v['f1'] for v in valid_classes) / len(valid_classes) if valid_classes else 0.0,
        'num_classes': len(valid_classes),
    }
    return micro, macro, per_class


def print_table(micro, macro, per_class, cat_names):
    print('\n[Overall Micro]')
    print(f"TP={micro['tp']} FP={micro['fp']} FN={micro['fn']} GT={micro['gt']} Pred={micro['pred']}")
    print(f"Precision={micro['precision']:.4f} Recall={micro['recall']:.4f} F1={micro['f1']:.4f}")
    print('\n[Macro over classes with GT]')
    print(f"Precision={macro['precision']:.4f} Recall={macro['recall']:.4f} F1={macro['f1']:.4f} Classes={macro['num_classes']}")
    print('\n[Per Class]')
    print('category_id\tname\tTP\tFP\tFN\tGT\tPred\tP\tR\tF1')
    for cat, row in per_class.items():
        print(
            f"{cat}\t{cat_names.get(cat, str(cat))}\t{row['tp']}\t{row['fp']}\t{row['fn']}\t"
            f"{row['gt']}\t{row['pred']}\t{row['precision']:.4f}\t{row['recall']:.4f}\t{row['f1']:.4f}"
        )


def save_json(path, args, micro, macro, per_class, cat_names):
    if not path:
        return
    out = {
        'score_thr': args.score_thr,
        'iou_thr': args.iou_thr,
        'micro': micro,
        'macro': macro,
        'per_class': {
            str(cat): {'name': cat_names.get(cat, str(cat)), **row}
            for cat, row in per_class.items()
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\n[PR] Saved json: {path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate fixed-threshold Precision/Recall for RT-DETR or FlowRT-DETR.')
    parser.add_argument('--ann-file', required=True, help='COCO-style GT annotation json.')
    parser.add_argument('--pred-file', default=None, help='COCO detection json. If omitted, run inference from --config/--resume.')
    parser.add_argument('-c', '--config', default=None)
    parser.add_argument('-r', '--resume', default=None)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument('--max-dets', type=int, default=300)
    parser.add_argument('--category-offset', type=int, default=0,
                        help='Use 1 if model labels/pred labels are 0-based but GT category ids are 1-based.')
    parser.add_argument('--batch-size', type=int, default=-1)
    parser.add_argument('--num-workers', type=int, default=-1)
    parser.add_argument('--max-images', type=int, default=-1)
    parser.add_argument('--print-freq', type=int, default=200)
    parser.add_argument('--no-deploy', action='store_true')
    parser.add_argument('-o', '--output', default=None, help='Optional json output.')
    return parser.parse_args()


def main(args):
    coco, gt_by_img, img_ids = load_gt_from_ann(args.ann_file)
    cat_names = category_names_from_coco(coco)

    if args.pred_file:
        preds = load_json(args.pred_file)
    else:
        if not args.config or not args.resume:
            raise ValueError('Either --pred-file or both --config/--resume are required.')
        preds = infer_predictions(args)

    pred_by_img = normalize_predictions(preds, args.score_thr, category_offset=args.category_offset)
    micro, macro, per_class = evaluate_pr(gt_by_img, img_ids, pred_by_img, args.iou_thr)
    print_table(micro, macro, per_class, cat_names)
    save_json(args.output, args, micro, macro, per_class, cat_names)


if __name__ == '__main__':
    main(parse_args())