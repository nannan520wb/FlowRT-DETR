"""
排队估计端到端评估脚本
======================
针对 FlowRT-DETR 论文的排队估计实验, 跑三种 setting 做对比:

    Setting A: Detection-only (无 tracker, 直接用阈值判断速度需要相邻帧 box 匹配,
                                 退化为"低速度+低 IoU 帧间匹配"——近似 BBFR-tracker 的输出)
    Setting B: Detection + SORT (RT-DETR + SORT baseline)
    Setting C: Detection + SORT + Kalman 插值 (强行消除 flicker, 但可能引入假轨迹)
    Setting D: GT (上限参考)

一次运行可以跑 A/B/C/D 中任意子集. 通过命令行 --settings 控制.

输出:
    JSON, 每个视频每个 setting 的:
        - 逐帧排队长度序列 (qlen_seq)
        - 稳定性指标 (std, jitter, peak_jitter, smoothness)
        - 精度指标 (mae, rmse, mape, vs GT)

用法:
    python tools/queue_estimation/eval_queue.py \
        -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
        -r output/ua_4.28/checkpoint.pth \
        --settings B C D \
        --out-json output/queue_flowrt.json \
        --num-workers 0
"""
import os
import sys
import json
import argparse
from collections import defaultdict
from pathlib import Path

from typing import List, Dict, Tuple, Optional
import torch
import numpy as np

_PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ))

from src.core import YAMLConfig
from tools.queue_estimation.queue_metrics import (
    SORT, KalmanBoxTracker,
    compute_queue_length,
    compute_queue_stability_metrics,
    compute_queue_accuracy,
)


# =====================================================================
# UA-DETRAC 视频名 / 帧序号解析
# =====================================================================
def parse_filename(file_name: str):
    import re
    parts = file_name.replace('\\', '/').split('/')
    video_id = parts[0] if len(parts) >= 2 else 'default'
    leaf = parts[-1]
    m = re.search(r'(\d+)', leaf)
    frame_idx = int(m.group(1)) if m else 0
    return video_id, frame_idx


# =====================================================================
# ROI 自动估计: 用前 N 帧 GT 的 box 底边中点的凸包作为 ROI
# =====================================================================
def estimate_roi_from_gt(video_gts: dict, n_warmup: int = 100) -> np.ndarray:
    """
    返回 [K, 2] 的多边形 (凸包). 若数据不足返回 None.
    """
    pts = []
    sorted_frames = sorted(video_gts.keys())[:n_warmup]
    for f in sorted_frames:
        gt = video_gts[f]
        for box in gt['boxes']:
            cx = (box[0] + box[2]) / 2
            cy = box[3]
            pts.append([cx, cy])
    if len(pts) < 10:
        return None
    pts = np.array(pts, dtype=np.float32)
    # 简易凸包 (用 scipy 如果有, 否则用矩形包围)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
        return pts[hull.vertices]
    except ImportError:
        # 退化: 用最小外接矩形
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


# =====================================================================
# 核心: 跑推理 + 收集每帧检测结果 (复用 BBFR 的逻辑)
# =====================================================================
@torch.no_grad()
def run_inference(cfg, device: str, num_workers: int = 0,
                  score_thresh_keep: float = 0.05):
    """跑模型, 返回 video_dets[video_id][frame_idx] = {'boxes','labels','scores'}"""
    # 加载模型 (注意顺序: 先 load weight, 再 deploy 融合 RepVgg)
    model = cfg.model
    postproc = cfg.postprocessor

    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location='cpu')
        if 'ema' in ckpt and ckpt['ema'] is not None:
            state = ckpt['ema']['module']
            print("[Queue] Loading EMA weights")
        elif 'model' in ckpt:
            state = ckpt['model']
            print("[Queue] Loading model weights")
        else:
            state = ckpt
        new_state = {k.replace('module.', ''): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        print(f"[Queue] missing keys: {len(missing)}, unexpected: {len(unexpected)}")

    if hasattr(model, 'deploy'):
        model = model.deploy()
    if hasattr(postproc, 'deploy'):
        postproc = postproc.deploy()

    model = model.to(device).eval()
    postproc = postproc.to(device) if isinstance(postproc, torch.nn.Module) else postproc

    val_loader = cfg.val_dataloader
    if num_workers >= 0:
        from torch.utils.data import DataLoader
        val_loader = DataLoader(
            dataset    =val_loader.dataset,
            batch_size =val_loader.batch_size,
            shuffle    =False,
            num_workers=num_workers,
            collate_fn =val_loader.collate_fn,
            drop_last  =False,
        )
    coco_api = val_loader.dataset.coco

    video_dets = defaultdict(dict)
    video_gts  = defaultdict(dict)
    n_imgs = 0
    for batch in val_loader:
        (samples, prev_samples, flow_img), targets = batch
        samples = samples.to(device)
        prev_samples = prev_samples.to(device)
        flow_img = flow_img.to(device)

        outputs = model(samples, x_prev=prev_samples, flow=flow_img)
        orig_target_sizes = torch.stack(
            [t['orig_size'].to(device) for t in targets], dim=0
        )
        results = postproc(outputs, orig_target_sizes)
        if isinstance(results, tuple):
            labels_b, boxes_b, scores_b = results
            results = [
                {'labels': labels_b[i], 'boxes': boxes_b[i], 'scores': scores_b[i]}
                for i in range(labels_b.shape[0])
            ]

        for tgt, res in zip(targets, results):
            img_id = int(tgt['image_id'].item())
            file_name = coco_api.loadImgs(img_id)[0]['file_name']
            vid, fidx = parse_filename(file_name)

            scores = res['scores'].detach().cpu().numpy()
            keep = scores >= score_thresh_keep
            video_dets[vid][fidx] = {
                'boxes' : res['boxes' ].detach().cpu().numpy()[keep],
                'labels': res['labels'].detach().cpu().numpy()[keep],
                'scores': scores[keep],
            }

            ann_ids = coco_api.getAnnIds(imgIds=img_id)
            anns = coco_api.loadAnns(ann_ids)
            gt_boxes, gt_labels, gt_tids = [], [], []
            for ann in anns:
                x, y, w, h = ann['bbox']
                gt_boxes.append([x, y, x + w, y + h])
                gt_labels.append(int(ann['category_id']))
                tid = ann.get('track_id', ann.get('instance_id', ann.get('object_id', -1)))
                gt_tids.append(int(tid))
            if len(gt_boxes) > 0:
                video_gts[vid][fidx] = {
                    'boxes'    : np.asarray(gt_boxes,  dtype=np.float32),
                    'labels'   : np.asarray(gt_labels, dtype=np.int64),
                    'track_ids': np.asarray(gt_tids,   dtype=np.int64),
                }
            n_imgs += 1
        if n_imgs % 500 == 0:
            print(f"[Queue] processed {n_imgs} frames...")

    print(f"[Queue] Inference done. Total frames: {n_imgs}, videos: {len(video_dets)}")
    return dict(video_dets), dict(video_gts)


# =====================================================================
# 跑一种 setting 在一个视频上的排队序列
# =====================================================================
def run_setting_on_video(setting: str,
                         video_dets: dict,
                         video_gts: dict,
                         video_id: str,
                         roi: np.ndarray,
                         score_thresh: float,
                         speed_thresh: float,
                         iou_thresh: float = 0.3,
                         max_age: int = 5,
                         min_hits: int = 3) -> List[int]:
    """
    返回逐帧排队长度序列 (按帧序号升序).

    setting:
        'B' - Det + SORT (无插值)
        'C' - Det + SORT + Kalman 插值
        'D' - GT (用 GT 框 + GT track_id 算速度)
    """
    sorted_frames = sorted(video_dets.keys())
    queue_seq = []

    if setting in ('B', 'C'):
        KalmanBoxTracker.count = 0  # 重置 ID, 避免跨视频干扰
        sort = SORT(iou_thresh=iou_thresh, max_age=max_age, min_hits=min_hits)
        for fidx in sorted_frames:
            d = video_dets[fidx]
            keep = d['scores'] >= score_thresh
            boxes = d['boxes'][keep]
            labels = d['labels'][keep]
            scores = d['scores'][keep]
            if setting == 'B':
                tracks = sort.update(fidx, boxes, labels, scores)
            else:  # C
                tracks = sort.update_with_interp(fidx, boxes, labels, scores)
            qlen, _ = compute_queue_length(tracks, roi, speed_thresh)
            queue_seq.append(qlen)

    elif setting == 'D':
        # 用 GT 算: 直接根据 track_id 串轨迹, 用相邻帧 box 中心位移算速度
        gt_track_history = defaultdict(list)  # tid -> [(fidx, cx, cy)]
        for fidx in sorted_frames:
            gt = video_gts.get(fidx, None)
            tracks_now = []
            if gt is None:
                queue_seq.append(0)
                continue
            for k in range(len(gt['boxes'])):
                tid = int(gt['track_ids'][k])
                if tid < 0:
                    continue
                box = gt['boxes'][k]
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                gt_track_history[tid].append((fidx, cx, cy))
                # 估速度 (用最近 5 个观测)
                hist = gt_track_history[tid][-5:]
                if len(hist) >= 2:
                    f0, x0, y0 = hist[0]
                    f1, x1, y1 = hist[-1]
                    dt = max(f1 - f0, 1)
                    v = float(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / dt)
                else:
                    v = 0.0
                tracks_now.append((tid, box, int(gt['labels'][k]), v))
            qlen, _ = compute_queue_length(tracks_now, roi, speed_thresh)
            queue_seq.append(qlen)

    return queue_seq


# =====================================================================
# Main
# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', required=True)
    p.add_argument('-r', '--resume', required=True)
    p.add_argument('--settings', nargs='+', default=['B', 'C', 'D'],
                   choices=['B', 'C', 'D'],
                   help="哪些 setting 要跑. B=Det+SORT, C=Det+SORT+插值, D=GT")
    p.add_argument('--score-thresh', type=float, default=0.3)
    p.add_argument('--speed-thresh', type=float, default=2.0,
                   help="排队速度阈值 (pixel/frame). UA-DETRAC 25fps, 2 px/frame ≈ 50 px/s")
    p.add_argument('--iou-thresh',   type=float, default=0.3, help="SORT IoU 阈值")
    p.add_argument('--max-age',      type=int,   default=5,   help="SORT max_age")
    p.add_argument('--min-hits',     type=int,   default=3,   help="SORT min_hits")
    p.add_argument('--roi-warmup',   type=int,   default=100, help="用前 N 帧 GT 估计 ROI")
    p.add_argument('--out-json',     type=str,   default='queue_result.json')
    p.add_argument('--device',       type=str,
                   default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--num-workers',  type=int,   default=0)
    p.add_argument('--save-detections', type=str, default='',
                   help="保存推理结果到 npz, 后续可重复用")
    p.add_argument('--load-detections', type=str, default='',
                   help="从 npz 加载推理结果, 跳过模型推理")
    args = p.parse_args()

    # 加载配置 + 推理
    cfg = YAMLConfig(args.config, resume=args.resume)
    cfg.resume = args.resume

    if args.load_detections and Path(args.load_detections).exists():
        print(f"[Queue] Loading cached detections from {args.load_detections}")
        data = np.load(args.load_detections, allow_pickle=True)
        video_dets = data['video_dets'].item()
        video_gts  = data['video_gts'].item()
    else:
        video_dets, video_gts = run_inference(
            cfg, args.device, num_workers=args.num_workers
        )
        if args.save_detections:
            Path(args.save_detections).parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.save_detections,
                     video_dets=video_dets, video_gts=video_gts)
            print(f"[Queue] Cached to {args.save_detections}")

    # 逐视频跑各 setting
    print(f"\n[Queue] Running settings: {args.settings}")
    print(f"[Queue] score_thresh={args.score_thresh}, speed_thresh={args.speed_thresh}, "
          f"iou_thresh={args.iou_thresh}, max_age={args.max_age}, min_hits={args.min_hits}")

    per_video = {}
    for vid in sorted(video_dets.keys()):
        if vid not in video_gts:
            print(f"[Queue] [{vid}] no GT, skip")
            continue
        roi = estimate_roi_from_gt(video_gts[vid], args.roi_warmup)
        if roi is None:
            print(f"[Queue] [{vid}] ROI estimation failed, skip")
            continue

        # 跑 GT setting (作为 ground-truth queue length 序列)
        gt_seq = run_setting_on_video(
            'D', video_dets[vid], video_gts[vid], vid, roi,
            args.score_thresh, args.speed_thresh
        ) if 'D' in args.settings or len(args.settings) > 0 else []

        result = {'roi_polygon': roi.tolist(),
                  'n_frames'   : len(sorted(video_dets[vid].keys()))}

        for setting in args.settings:
            if setting == 'D':
                seq = gt_seq
            else:
                seq = run_setting_on_video(
                    setting, video_dets[vid], video_gts[vid], vid, roi,
                    args.score_thresh, args.speed_thresh,
                    iou_thresh=args.iou_thresh, max_age=args.max_age, min_hits=args.min_hits
                )

            stab = compute_queue_stability_metrics(seq)
            acc = compute_queue_accuracy(seq, gt_seq) if (setting != 'D' and len(gt_seq) > 0) else \
                  {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0, 'n_frames': len(seq)}
            result[f'setting_{setting}'] = {
                'qlen_seq'   : seq,
                'stability'  : stab,
                'accuracy_vs_gt': acc,
            }

        per_video[vid] = result
        # 简要打印
        line = f"[{vid}] "
        for s in args.settings:
            r = result[f'setting_{s}']
            line += f"{s}: mean={r['stability']['mean']:.1f} std={r['stability']['std']:.2f} " \
                    f"jitter={r['stability']['jitter']:.2f} mae={r['accuracy_vs_gt']['mae']:.2f}  "
        print(line)

    # 全局聚合 (跨视频)
    print("\n[Queue] ===== Aggregated =====")
    aggregated = {}
    for setting in args.settings:
        all_std    = [r[f'setting_{setting}']['stability']['std']         for r in per_video.values()]
        all_jitter = [r[f'setting_{setting}']['stability']['jitter']      for r in per_video.values()]
        all_smooth = [r[f'setting_{setting}']['stability']['smoothness']  for r in per_video.values()]
        all_mae    = [r[f'setting_{setting}']['accuracy_vs_gt']['mae']    for r in per_video.values()]
        all_rmse   = [r[f'setting_{setting}']['accuracy_vs_gt']['rmse']   for r in per_video.values()]
        agg = {
            'mean_std'       : float(np.mean(all_std)),
            'mean_jitter'    : float(np.mean(all_jitter)),
            'mean_smoothness': float(np.mean(all_smooth)),
            'mean_mae'       : float(np.mean(all_mae)),
            'mean_rmse'      : float(np.mean(all_rmse)),
            'n_videos'       : len(all_std),
        }
        aggregated[f'setting_{setting}'] = agg
        print(f"  Setting {setting}: std={agg['mean_std']:.3f}  "
              f"jitter={agg['mean_jitter']:.3f}  "
              f"smoothness={agg['mean_smoothness']:.3f}  "
              f"mae={agg['mean_mae']:.3f}  rmse={agg['mean_rmse']:.3f}")

    # 输出 JSON
    out = {
        'config'     : args.config,
        'checkpoint' : args.resume,
        'hparams'    : vars(args),
        'aggregated' : aggregated,
        'per_video'  : per_video,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, 'item') else
                          (x.tolist() if hasattr(x, 'tolist') else str(x)))
    print(f"\n[Queue] Saved to {args.out_json}")


if __name__ == '__main__':
    main()