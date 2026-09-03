"""
BBFR 诊断脚本: 跑 5 个 batch, 把每张图的检测情况打印出来,
定位为什么 BBFR 全是 0.

用法:
    python tools/bbfr/diag_bbfr.py \
        -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
        -r output/ua_4.28/best.pth
"""
import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np

_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

from src.core import YAMLConfig


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', required=True)
    p.add_argument('-r', '--resume', required=True)
    p.add_argument('--n-batches', type=int, default=3)
    p.add_argument('--device', type=str,
                   default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    cfg = YAMLConfig(args.config, resume=args.resume)
    cfg.resume = args.resume

    # 1. 加载模型 (先不 deploy, 让 RepVggBlock 保持 conv1+conv2 结构)
    model = cfg.model
    postproc = cfg.postprocessor

    ckpt = torch.load(cfg.resume, map_location='cpu')
    if 'ema' in ckpt and ckpt['ema'] is not None:
        state = ckpt['ema']['module']
        print("[DIAG] Loading EMA weights")
    elif 'model' in ckpt:
        state = ckpt['model']
        print("[DIAG] Loading model weights")
    else:
        state = ckpt
    new_state = {k.replace('module.', ''): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"[DIAG] missing keys : {len(missing)}")
    print(f"[DIAG] unexpected   : {len(unexpected)}")
    if missing:
        print(f"[DIAG] sample missing: {missing[:3]}")
    if unexpected:
        print(f"[DIAG] sample unexpect: {unexpected[:3]}")

    # 加载完权重再切 deploy (融合 RepVgg)
    if hasattr(model, 'deploy'):
        model = model.deploy()
    if hasattr(postproc, 'deploy'):
        postproc = postproc.deploy()

    model = model.to(args.device).eval()

    # 2. dataloader
    val_loader = cfg.val_dataloader
    from torch.utils.data import DataLoader
    val_loader = DataLoader(
        dataset    =val_loader.dataset,
        batch_size =val_loader.batch_size,
        shuffle    =False,
        num_workers=0,
        collate_fn =val_loader.collate_fn,
        drop_last  =False,
    )
    coco_api = val_loader.dataset.coco
    print(f"[DIAG] val dataset size: {len(val_loader.dataset)}")
    print(f"[DIAG] num classes: {len(coco_api.getCatIds())}")
    print(f"[DIAG] cats: {coco_api.loadCats(coco_api.getCatIds())}")

    # 3. 跑前几个 batch
    for bi, batch in enumerate(val_loader):
        if bi >= args.n_batches:
            break
        (samples, prev_samples, flow_img), targets = batch
        samples = samples.to(args.device)
        prev_samples = prev_samples.to(args.device)
        flow_img = flow_img.to(args.device)

        outputs = model(samples, x_prev=prev_samples, flow=flow_img)
        orig_target_sizes = torch.stack(
            [t['orig_size'].to(args.device) for t in targets], dim=0
        )

        # ---- 先看原始 outputs 的样子 (在 postprocessor 之前) ----
        print(f"\n========== Batch {bi} ==========")
        print(f"orig_target_sizes: {orig_target_sizes.tolist()}")
        if isinstance(outputs, dict):
            print(f"outputs keys: {list(outputs.keys())}")
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    print(f"  outputs[{k}]: shape={tuple(v.shape)}, "
                          f"min={v.min().item():.4f}, max={v.max().item():.4f}, "
                          f"mean={v.mean().item():.4f}")

        # ---- 走 postprocessor ----
        results = postproc(outputs, orig_target_sizes)
        if isinstance(results, tuple):
            labels_b, boxes_b, scores_b = results
            results = [
                {'labels': labels_b[i], 'boxes': boxes_b[i], 'scores': scores_b[i]}
                for i in range(labels_b.shape[0])
            ]

        for i, (tgt, res) in enumerate(zip(targets, results)):
            img_id = int(tgt['image_id'].item())
            file_name = coco_api.loadImgs(img_id)[0]['file_name']
            scores = res['scores'].detach().cpu().numpy()
            boxes = res['boxes'].detach().cpu().numpy()
            labels = res['labels'].detach().cpu().numpy()

            # 关键诊断
            print(f"\n  [img {i}] file: {file_name}")
            print(f"    orig_size:  {tgt['orig_size'].tolist()}")
            print(f"    scores: shape={scores.shape}, "
                  f"min={scores.min():.4f}, max={scores.max():.4f}, "
                  f"top5={sorted(scores.tolist(), reverse=True)[:5]}")
            print(f"    >0.3 count: {(scores >= 0.3).sum()}, "
                  f">0.5: {(scores >= 0.5).sum()}, "
                  f">0.1: {(scores >= 0.1).sum()}, "
                  f">0.05: {(scores >= 0.05).sum()}")
            print(f"    labels unique: {np.unique(labels[scores>=0.1])}")
            top_idx = np.argsort(-scores)[:3]
            print(f"    top3 boxes (score>=0.05): ")
            for ti in top_idx:
                if scores[ti] >= 0.05:
                    print(f"      score={scores[ti]:.3f} label={labels[ti]} "
                          f"box={boxes[ti].tolist()}")

            # GT 对比
            ann_ids = coco_api.getAnnIds(imgIds=img_id)
            anns = coco_api.loadAnns(ann_ids)
            print(f"    GT count: {len(anns)}")
            if len(anns) > 0:
                print(f"    GT[0]: bbox={anns[0]['bbox']} cat_id={anns[0]['category_id']}")

    print("\n[DIAG] Done.")


if __name__ == '__main__':
    main()