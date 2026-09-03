"""
BBFR Evaluation Script for FlowRT-DETR
=======================================
跑完整个 val 集, 按视频分组计算 BBFR-Det (主) 和 BBFR-GT (可选).
结果输出 JSON, 可直接贴论文表格.

用法 (放在项目根目录下运行):
    python tools/bbfr/eval_bbfr.py \
        -c configs/rtdetr/<your_cfg>.yml \
        -r output/<your_run>/best.pth \
        --score-thresh 0.3 \
        --iou-thresh   0.5 \
        --max-lost     5 \
        --out-json     output/bbfr_result.json

设计要点:
    - 直接复用 src/data 的 dataloader, 保证与训练验证完全一致的数据预处理 / 光流计算.
    - 视频分组依据 file_name 的前缀 (UA-DETRAC 格式: MVI_XXXX/img00001.jpg).
    - 可选 --gt-track-key 指定 ann json 里的 track id 字段 (如 'track_id'),
      没有这个字段就只算 BBFR-Det.
"""

import os
import sys
import json
import argparse
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np

# 把项目根目录加进 sys.path
_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

from src.core import YAMLConfig
from tools.bbfr.bbfr_metric import (
    compute_bbfr_det,
    compute_bbfr_gt,
    aggregate_bbfr,
)


# =====================================================================
# UA-DETRAC: 从 file_name 解析 (video_id, frame_idx)
# =====================================================================
def parse_ua_detrac_filename(file_name: str):
    """
    UA-DETRAC 的 COCO file_name 形如 'MVI_20011/img00001.jpg'.
    返回 (video_id_str, frame_idx_int).
    若解析失败, 返回 (file_name 的 dirname, idx_in_video).
    """
    # video_id = 第一级目录名
    parts = file_name.replace('\\', '/').split('/')
    if len(parts) >= 2:
        video_id = parts[0]
        leaf = parts[-1]
    else:
        video_id = 'default'
        leaf = parts[0]

    # 从文件名里抽数字 (匹配 img00001.jpg / 00001.jpg / frame_5.png 等)
    import re
    m = re.search(r'(\d+)', leaf)
    frame_idx = int(m.group(1)) if m else 0
    return video_id, frame_idx


# =====================================================================
# 主流程
# =====================================================================
@torch.no_grad()
def run_inference_and_collect(cfg, device: str, score_thresh_keep: float = 0.05,
                              num_workers: int = -1, batch_size: int = -1):
    """
    跑一遍 val_dataloader, 收集每张图的检测结果 + GT, 按视频分组.
    score_thresh_keep: 推理时保留多少分数以上的预测 (设很低, 后续 BBFR 再卡阈值).
    num_workers: 若 >= 0, 重建 dataloader 并覆盖配置中的 num_workers.
    batch_size : 若 >= 1, 重建 dataloader 并覆盖配置中的 batch_size.
    """
    # 1. 模型 (注意: 先不要 deploy, 否则 RepVggBlock 会丢掉 conv1/conv2)
    model = cfg.model
    postproc = cfg.postprocessor

    # 加载权重 (与 det_solver 一致的逻辑)
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location='cpu')
        if 'ema' in ckpt and ckpt['ema'] is not None:
            state = ckpt['ema']['module']
            print("[BBFR] Loading EMA weights")
        elif 'model' in ckpt:
            state = ckpt['model']
            print("[BBFR] Loading model weights")
        else:
            state = ckpt
        # 处理 wrapper 前缀
        new_state = {}
        for k, v in state.items():
            new_state[k.replace('module.', '')] = v
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        print(f"[BBFR] Loaded weights from: {cfg.resume}")
        print(f"[BBFR] missing keys: {len(missing)}, unexpected: {len(unexpected)}")
        if len(missing) > 0:
            # 容忍数量极少的 missing (例如 num_batches_tracked), 否则报错让用户知道
            critical = [k for k in missing
                        if 'num_batches_tracked' not in k and 'running_' not in k]
            if len(critical) > 0:
                print(f"[BBFR][WARN] {len(critical)} CRITICAL missing keys, "
                      f"e.g.: {critical[:5]}")
                print("[BBFR][WARN] 模型可能未正确加载, 检测结果可能全部无效!")

    # 加载完权重后, 再切到 deploy 模式 (融合 RepVggBlock 的 conv1+conv2 -> conv)
    if hasattr(model, 'deploy'):
        model = model.deploy()
    if hasattr(postproc, 'deploy'):
        postproc = postproc.deploy()

    model = model.to(device).eval()
    postproc = postproc.to(device) if isinstance(postproc, torch.nn.Module) else postproc

    # 2. dataloader + COCO API 拿 file_name
    val_loader = cfg.val_dataloader

    # 如果用户传了 num_workers / batch_size, 就用 torch 原生 DataLoader 重建一个
    # (沿用原 dataset 和 collate_fn, 只覆盖这两个字段)
    if num_workers >= 0 or batch_size >= 1:
        from torch.utils.data import DataLoader
        new_nw = num_workers if num_workers >= 0 else val_loader.num_workers
        new_bs = batch_size  if batch_size  >= 1 else val_loader.batch_size
        val_loader = DataLoader(
            dataset    =val_loader.dataset,
            batch_size =new_bs,
            shuffle    =False,
            num_workers=new_nw,
            collate_fn =val_loader.collate_fn,
            drop_last  =False,
        )
        print(f"[BBFR] Rebuilt val_loader: batch_size={new_bs}, num_workers={new_nw}")

    coco_api = val_loader.dataset.coco

    # 3. 收集容器
    # video_dets[video_id][frame_idx] = {'boxes':..., 'labels':..., 'scores':...}
    video_dets: dict = defaultdict(dict)
    # video_gts[video_id][frame_idx]  = {'boxes':..., 'labels':..., 'track_ids':...}
    video_gts: dict = defaultdict(dict)

    # 4. 推理循环 (复用 evaluate 的解包逻辑)
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

        # ---- 兼容 postprocessor 的两种返回格式 ----
        # 格式 1 (deploy_mode=True): tuple (labels[B,N], boxes[B,N,4], scores[B,N])
        # 格式 2 (deploy_mode=False): list[dict{'labels','boxes','scores'}]
        if isinstance(results, tuple):
            labels_b, boxes_b, scores_b = results
            results = [
                {'labels': labels_b[i], 'boxes': boxes_b[i], 'scores': scores_b[i]}
                for i in range(labels_b.shape[0])
            ]

        for tgt, res in zip(targets, results):
            img_id = int(tgt['image_id'].item())
            file_name = coco_api.loadImgs(img_id)[0]['file_name']
            video_id, frame_idx = parse_ua_detrac_filename(file_name)

            # 过滤掉极低分以省内存 (后面 BBFR 会再卡阈值)
            scores = res['scores'].detach().cpu().numpy()
            keep = scores >= score_thresh_keep
            video_dets[video_id][frame_idx] = {
                'boxes' : res['boxes' ].detach().cpu().numpy()[keep],
                'labels': res['labels'].detach().cpu().numpy()[keep],
                'scores': scores[keep],
            }

            # 顺便收集 GT (boxes 已经被 dataset transforms 处理过, 走原 ann 更稳)
            ann_ids = coco_api.getAnnIds(imgIds=img_id)
            anns = coco_api.loadAnns(ann_ids)
            gt_boxes, gt_labels, gt_tids = [], [], []
            for ann in anns:
                # COCO 的 bbox 是 [x, y, w, h], 转成 xyxy
                x, y, w, h = ann['bbox']
                gt_boxes.append([x, y, x + w, y + h])
                gt_labels.append(int(ann['category_id']))
                # UA-DETRAC 的 track id 字段名可能不同, 兼容多种
                tid = ann.get('track_id',
                       ann.get('instance_id',
                       ann.get('object_id', -1)))
                gt_tids.append(int(tid))
            if len(gt_boxes) > 0:
                video_gts[video_id][frame_idx] = {
                    'boxes'    : np.asarray(gt_boxes,  dtype=np.float32),
                    'labels'   : np.asarray(gt_labels, dtype=np.int64),
                    'track_ids': np.asarray(gt_tids,   dtype=np.int64),
                }
            n_imgs += 1

        if n_imgs % 200 == 0:
            print(f"[BBFR] processed {n_imgs} frames, "
                  f"{len(video_dets)} videos so far...")

    print(f"[BBFR] Inference done. Total frames: {n_imgs}, videos: {len(video_dets)}")
    return dict(video_dets), dict(video_gts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', required=True)
    p.add_argument('-r', '--resume', required=True)
    p.add_argument('--score-thresh', type=float, default=0.3,
                   help="BBFR 计算时的检测置信度阈值 (默认 0.3)")
    p.add_argument('--iou-thresh',   type=float, default=0.5,
                   help="tracker 帧间匹配 IoU 阈值")
    p.add_argument('--max-lost',     type=int,   default=5,
                   help="track 允许的最大丢失帧数 (即 flicker 最大跨度)")
    p.add_argument('--min-track-len', type=int,  default=3,
                   help="过滤短轨迹: 长度 < 此值的 track 不计入统计")
    p.add_argument('--out-json',     type=str,   default='bbfr_result.json')
    p.add_argument('--device',       type=str,
                   default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--no-gt',        action='store_true',
                   help="跳过 BBFR-GT (若数据集没有 track_id 时必须设此项)")
    p.add_argument('--score-thresh-keep', type=float, default=0.05,
                   help="推理时保留的最低分数 (省内存, 不影响 BBFR)")
    p.add_argument('--num-workers', type=int, default=-1,
                   help="dataloader 的 num_workers, -1 表示沿用配置文件里的设置；"
                        "单进程光流推理建议设为 0。")
    p.add_argument('--batch-size', type=int, default=-1,
                   help="dataloader 的 batch_size, -1 表示沿用配置文件里的设置.")
    args = p.parse_args()

    # 加载配置
    cfg = YAMLConfig(args.config, resume=args.resume)
    cfg.resume = args.resume

    # 跑推理 + 收集
    video_dets, video_gts = run_inference_and_collect(
        cfg, args.device,
        score_thresh_keep=args.score_thresh_keep,
        num_workers      =args.num_workers,
        batch_size       =args.batch_size,
    )

    # ============== BBFR-Det (主指标) ==============
    print("\n[BBFR] ===== Computing BBFR-Det =====")
    per_video_det = {}
    for vid, dets in video_dets.items():
        r = compute_bbfr_det(
            dets,
            score_thresh=args.score_thresh,
            iou_thresh  =args.iou_thresh,
            max_lost    =args.max_lost,
            min_track_len=args.min_track_len,
        )
        per_video_det[vid] = r
        print(f"  [{vid}] BBFR={r['BBFR']:.3f}  tracks={r['num_tracks']}  "
              f"flickers={r['flicker_events']}  track_frames={r['total_track_frames']}")
    agg_det = aggregate_bbfr(per_video_det, key='BBFR')
    print(f"\n[BBFR-Det] micro = {agg_det['BBFR_micro']:.4f}   "
          f"macro = {agg_det['BBFR_macro']:.4f}   "
          f"(per 1000 track-frames)")

    # ============== BBFR-GT (辅助验证) ==============
    agg_gt = None
    if not args.no_gt and len(video_gts) > 0:
        # 检查是否有 track_id (UA-DETRAC 标准 ann 通常有)
        sample_v = next(iter(video_gts.values()))
        sample_f = next(iter(sample_v.values()))
        has_tid = (sample_f['track_ids'] >= 0).any()
        if has_tid:
            print("\n[BBFR] ===== Computing BBFR-GT =====")
            per_video_gt = {}
            for vid, dets in video_dets.items():
                if vid not in video_gts:
                    continue
                r = compute_bbfr_gt(
                    dets, video_gts[vid],
                    score_thresh=args.score_thresh,
                    iou_match   =args.iou_thresh,
                )
                per_video_gt[vid] = r
                print(f"  [{vid}] BBFR_GT={r['BBFR_GT']:.3f}  "
                      f"gt_tracks={r['num_gt_tracks']}  "
                      f"flickers={r['gt_flicker_events']}")
            agg_gt = aggregate_bbfr(per_video_gt, key='BBFR_GT')
            print(f"\n[BBFR-GT] micro = {agg_gt['BBFR_GT_micro']:.4f}   "
                  f"macro = {agg_gt['BBFR_GT_macro']:.4f}   "
                  f"(per 1000 GT-frames)")
        else:
            print("\n[BBFR] No track_id found in annotations, skipping BBFR-GT.")

    # ============== 保存 ==============
    out = {
        'config'              : args.config,
        'checkpoint'          : args.resume,
        'hparams'             : {
            'score_thresh': args.score_thresh,
            'iou_thresh'  : args.iou_thresh,
            'max_lost'    : args.max_lost,
            'min_track_len': args.min_track_len,
        },
        'BBFR_Det'            : agg_det,
        'BBFR_GT'             : agg_gt,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print(f"\n[BBFR] Saved to {args.out_json}")


if __name__ == '__main__':
    main()
