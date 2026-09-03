"""
BBFR-Queue 相关性分析脚本 (Section V.D 用)
============================================

核心问题: 在每个视频上, BBFR (检测稳定性) 和 queue stability (排队
稳定性) 是不是真的相关? 如果是, 这就是 BBFR 应用价值的定量证据.

输出:
  1. 文字报告 (Pearson r, Spearman r, p-value)
  2. 论文级散点图 (.pdf 和 .png)
  3. CSV (40 个视频的所有指标, 方便你自己进一步分析)

用法:
  python bbfr_queue_correlation.py \
      --bbfr-json    output/bbfr_flowrt.json \
      --queue-json   output/queue_flowrt.json \
      --bbfr-baseline-json    output/bbfr_baseline.json \
      --queue-baseline-json   output/queue_baseline.json \
      --setting B \
      --bbfr-key BBFR \
      --queue-key jitter \
      --out-dir output/correlation_analysis
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


# =====================================================================
# 1. 读取 BBFR JSON 的 per_video 数据
# =====================================================================
def load_bbfr_per_video(json_path, bbfr_type='det'):
    """
    返回 {video_id: bbfr_value} dict.
    bbfr_type: 'det' (BBFR-D, 主指标) 或 'gt' (BBFR-T)
    """
    with open(json_path) as f:
        d = json.load(f)

    if bbfr_type == 'det':
        if 'BBFR_Det' not in d or 'per_video' not in d['BBFR_Det']:
            raise ValueError(f"{json_path} 里没有 BBFR_Det.per_video 字段, "
                             f"请用最新版 eval_bbfr.py 重跑")
        per_video = d['BBFR_Det']['per_video']
        result = {vid: pv['BBFR'] for vid, pv in per_video.items()}
    elif bbfr_type == 'gt':
        if 'BBFR_GT' not in d or d['BBFR_GT'] is None or 'per_video' not in d['BBFR_GT']:
            raise ValueError(f"{json_path} 里没有 BBFR_GT.per_video 字段, "
                             f"或 BBFR_GT 为 null")
        per_video = d['BBFR_GT']['per_video']
        result = {vid: pv['BBFR_GT'] for vid, pv in per_video.items()}
    else:
        raise ValueError(f"未知 bbfr_type: {bbfr_type}")

    print(f"  [BBFR-{bbfr_type.upper()}] 加载 {len(result)} 个视频")
    return result


# =====================================================================
# 2. 读取 Queue JSON 的 per_video 数据
# =====================================================================
def load_queue_per_video(json_path, setting='B', metric='jitter'):
    """
    返回 {video_id: queue_metric_value} dict.
    setting: 'B' (Det+SORT 无插值) 或 'C' (插值) 或 'D' (GT)
    metric: 'std' / 'jitter' / 'smoothness' / 'mae' / 'rmse' / 'mean'
    """
    with open(json_path) as f:
        d = json.load(f)

    if 'per_video' not in d:
        raise ValueError(f"{json_path} 里没有 per_video 字段")

    setting_key = f'setting_{setting}'
    result = {}
    for vid, pv in d['per_video'].items():
        if setting_key not in pv:
            print(f"  [WARN] {vid} 缺少 {setting_key}, 跳过")
            continue
        # stability 指标 (mean/std/jitter/peak_jitter/smoothness)
        if metric in pv[setting_key]['stability']:
            result[vid] = pv[setting_key]['stability'][metric]
        # accuracy 指标 (mae/rmse/mape)
        elif metric in pv[setting_key]['accuracy_vs_gt']:
            result[vid] = pv[setting_key]['accuracy_vs_gt'][metric]
        else:
            raise ValueError(f"未知 metric: {metric}")

    print(f"  [Queue-{setting}-{metric}] 加载 {len(result)} 个视频")
    return result


# =====================================================================
# 3. 主分析: 计算相关系数 + 画图
# =====================================================================
def analyze_correlation(bbfr_dict, queue_dict, label='', out_dir='.', tag=''):
    """
    在每个视频上配对 (BBFR, queue_metric), 计算相关系数并画散点图.
    """
    from scipy.stats import pearsonr, spearmanr, kendalltau
    import matplotlib.pyplot as plt

    # 1. 配对 (取交集视频)
    common_videos = sorted(set(bbfr_dict.keys()) & set(queue_dict.keys()))
    if len(common_videos) < 5:
        raise ValueError(f"配对视频数 = {len(common_videos)}, 太少, 检查 vid 命名是否一致")

    bbfr_vals = np.array([bbfr_dict[v] for v in common_videos])
    queue_vals = np.array([queue_dict[v] for v in common_videos])

    # 2. 计算 3 种相关系数
    r_pearson, p_pearson = pearsonr(bbfr_vals, queue_vals)
    r_spearman, p_spearman = spearmanr(bbfr_vals, queue_vals)
    r_kendall, p_kendall = kendalltau(bbfr_vals, queue_vals)

    # 3. 文字报告
    print(f"\n{'='*70}")
    print(f"  Correlation Analysis: {label}")
    print(f"{'='*70}")
    print(f"  Configured videos:     {len(common_videos)}")
    print(f"  BBFR  range: [{bbfr_vals.min():.3f}, {bbfr_vals.max():.3f}], mean={bbfr_vals.mean():.3f}")
    print(f"  Queue range: [{queue_vals.min():.3f}, {queue_vals.max():.3f}], mean={queue_vals.mean():.3f}")
    print(f"")
    print(f"  Pearson  r = {r_pearson:+.4f}   (p = {p_pearson:.4g})")
    print(f"  Spearman r = {r_spearman:+.4f}   (p = {p_spearman:.4g})")
    print(f"  Kendall  τ = {r_kendall:+.4f}    (p = {p_kendall:.4g})")
    print(f"")
    # 解读
    if r_pearson > 0.5 and p_pearson < 0.01:
        print(f"  ✅ 强正相关, 显著 (论文核心证据成立)")
    elif r_pearson > 0.3 and p_pearson < 0.05:
        print(f"  ⚠️  中等正相关, 显著 (论文证据较弱, 但可写)")
    elif p_pearson > 0.05:
        print(f"  ❌ 不显著, 没有统计学证据 (论文不能写'BBFR 与 queue 相关')")
    else:
        print(f"  ⚠️  弱相关或反向, 需要重新审视")

    # 4. 画论文级散点图
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(bbfr_vals, queue_vals, alpha=0.65, s=55, c='#2E86AB',
               edgecolors='black', linewidths=0.5, zorder=3)

    # 拟合线
    z = np.polyfit(bbfr_vals, queue_vals, 1)
    p_fit = np.poly1d(z)
    x_line = np.linspace(bbfr_vals.min(), bbfr_vals.max(), 100)
    ax.plot(x_line, p_fit(x_line), '--', color='#A23B72', linewidth=1.5,
            zorder=2, label=f'Linear fit (slope={z[0]:.3f})')

    # 标注几个 outlier 视频名 (可选)
    # 找残差最大的 3 个点标记出来
    residuals = queue_vals - p_fit(bbfr_vals)
    outlier_idx = np.argsort(np.abs(residuals))[-3:]
    for i in outlier_idx:
        ax.annotate(common_videos[i].replace('MVI_', ''),
                    (bbfr_vals[i], queue_vals[i]),
                    fontsize=7, color='#666', alpha=0.7,
                    xytext=(3, 3), textcoords='offset points')

    # 标题里写相关系数
    title = (f'{label}\n'
             f'Pearson r = {r_pearson:.3f} (p = {p_pearson:.3g}),  '
             f'n = {len(common_videos)} videos')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('BBFR (per 1000 detection-frames)', fontsize=11)
    ax.set_ylabel('Queue Length Jitter (vehicles/frame)', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=':', zorder=1)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)

    plt.tight_layout()

    # 保存
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pdf_path = Path(out_dir) / f'correlation_{tag}.pdf'
    png_path = Path(out_dir) / f'correlation_{tag}.png'
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")

    return {
        'label': label,
        'n_videos': len(common_videos),
        'pearson_r': float(r_pearson),
        'pearson_p': float(p_pearson),
        'spearman_r': float(r_spearman),
        'spearman_p': float(p_spearman),
        'kendall_tau': float(r_kendall),
        'kendall_p': float(p_kendall),
        'bbfr_mean': float(bbfr_vals.mean()),
        'queue_mean': float(queue_vals.mean()),
        'common_videos': common_videos,
        'bbfr_values': bbfr_vals.tolist(),
        'queue_values': queue_vals.tolist(),
    }


# =====================================================================
# 4. 多方法多指标对比图 (推荐, 直接出论文级 figure)
# =====================================================================
def make_multimethod_figure(results_list, out_path):
    """
    在一张图上画多个相关性子图 (例如 baseline + flowrt 各一个子图).
    适合论文 Figure X 直接用.
    """
    import matplotlib.pyplot as plt

    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(5.5*n, 4.5), squeeze=False)

    for ax, r in zip(axes[0], results_list):
        bbfr = np.array(r['bbfr_values'])
        queue = np.array(r['queue_values'])

        ax.scatter(bbfr, queue, alpha=0.65, s=55, c='#2E86AB',
                   edgecolors='black', linewidths=0.5, zorder=3)
        z = np.polyfit(bbfr, queue, 1)
        p_fit = np.poly1d(z)
        x_line = np.linspace(bbfr.min(), bbfr.max(), 100)
        ax.plot(x_line, p_fit(x_line), '--', color='#A23B72', linewidth=1.5,
                zorder=2)

        ax.set_title(f"{r['label']}\nPearson r = {r['pearson_r']:.3f} "
                     f"(p = {r['pearson_p']:.3g})", fontsize=10)
        ax.set_xlabel('BBFR (per 1000 frames)', fontsize=10)
        ax.set_ylabel('Queue Jitter (veh./frame)', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(str(out_path).replace('.pdf', '.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\n[Multi-method figure] Saved: {out_path}")


# =====================================================================
# 5. CSV 导出 (per-video 全部数据)
# =====================================================================
def save_per_video_csv(bbfr_dicts, queue_dicts, out_path):
    """
    bbfr_dicts: {method_name: {vid: bbfr}}
    queue_dicts: {method_name: {(vid, setting, metric): value}}
    输出 CSV 方便你后续做多变量回归等分析.
    """
    import csv
    all_videos = set()
    for d in bbfr_dicts.values():
        all_videos |= set(d.keys())
    for d in queue_dicts.values():
        all_videos |= set(d.keys())
    all_videos = sorted(all_videos)

    headers = ['video_id']
    for m in bbfr_dicts:
        headers.append(f'BBFR_{m}')
    for m in queue_dicts:
        headers.append(f'Queue_{m}')

    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for v in all_videos:
            row = [v]
            for m, d in bbfr_dicts.items():
                row.append(d.get(v, ''))
            for m, d in queue_dicts.items():
                row.append(d.get(v, ''))
            w.writerow(row)
    print(f"[CSV] Saved: {out_path}")


# =====================================================================
# Main
# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bbfr-json', required=True, help='FlowRT-DETR BBFR JSON')
    p.add_argument('--queue-json', required=True, help='FlowRT-DETR Queue JSON')
    p.add_argument('--bbfr-baseline-json', default='', help='Baseline BBFR JSON (可选)')
    p.add_argument('--queue-baseline-json', default='', help='Baseline Queue JSON (可选)')
    p.add_argument('--setting', default='B', choices=['B', 'C', 'D'],
                   help='Queue setting: B (无插值) / C (插值) / D (GT)')
    p.add_argument('--bbfr-type', default='det', choices=['det', 'gt'],
                   help='BBFR-D 还是 BBFR-T')
    p.add_argument('--queue-metric', default='jitter',
                   choices=['mean', 'std', 'jitter', 'peak_jitter', 'smoothness',
                            'mae', 'rmse', 'mape'],
                   help='Queue 维度的哪个指标')
    p.add_argument('--out-dir', default='./correlation_output')
    args = p.parse_args()

    print(f"=== BBFR-Queue Correlation Analysis ===")
    print(f"BBFR type:    {args.bbfr_type.upper()}")
    print(f"Queue setting: {args.setting}")
    print(f"Queue metric:  {args.queue_metric}")
    print()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # 1. 加载 FlowRT-DETR 数据
    print("[1/3] Loading FlowRT-DETR data...")
    bbfr_flowrt = load_bbfr_per_video(args.bbfr_json, args.bbfr_type)
    queue_flowrt = load_queue_per_video(args.queue_json, args.setting, args.queue_metric)

    # 2. 分析 FlowRT-DETR 的相关性
    print("\n[2/3] Computing correlation for FlowRT-DETR...")
    label_flowrt = (f'FlowRT-DETR: BBFR-{args.bbfr_type.upper()} '
                    f'vs Queue {args.queue_metric} (Setting {args.setting})')
    r_flowrt = analyze_correlation(
        bbfr_flowrt, queue_flowrt,
        label=label_flowrt,
        out_dir=args.out_dir,
        tag=f'flowrt_{args.bbfr_type}_{args.setting}_{args.queue_metric}'
    )

    all_results = [r_flowrt]
    bbfr_dicts = {'FlowRT': bbfr_flowrt}
    queue_dicts = {f'FlowRT_{args.queue_metric}': queue_flowrt}

    # 3. 如果有 baseline, 也分析
    if args.bbfr_baseline_json and args.queue_baseline_json:
        print("\n[3/3] Loading & analyzing Baseline data...")
        bbfr_bl = load_bbfr_per_video(args.bbfr_baseline_json, args.bbfr_type)
        queue_bl = load_queue_per_video(args.queue_baseline_json, args.setting, args.queue_metric)

        label_bl = (f'Baseline RT-DETR: BBFR-{args.bbfr_type.upper()} '
                    f'vs Queue {args.queue_metric} (Setting {args.setting})')
        r_bl = analyze_correlation(
            bbfr_bl, queue_bl,
            label=label_bl,
            out_dir=args.out_dir,
            tag=f'baseline_{args.bbfr_type}_{args.setting}_{args.queue_metric}'
        )
        all_results.insert(0, r_bl)  # baseline 在左
        bbfr_dicts['Baseline'] = bbfr_bl
        queue_dicts[f'Baseline_{args.queue_metric}'] = queue_bl

        # 多方法对比图 (论文用)
        make_multimethod_figure(
            all_results,
            Path(args.out_dir) / f'correlation_compare_{args.bbfr_type}_{args.setting}_{args.queue_metric}.pdf'
        )

    # 4. 导出 CSV (论文 supplementary)
    save_per_video_csv(
        bbfr_dicts, queue_dicts,
        Path(args.out_dir) / f'per_video_data_{args.bbfr_type}_{args.setting}_{args.queue_metric}.csv'
    )

    # 5. 保存所有结果到 JSON
    out_json = Path(args.out_dir) / f'analysis_summary_{args.bbfr_type}_{args.setting}_{args.queue_metric}.json'
    with open(out_json, 'w') as f:
        json.dump({'results': all_results, 'config': vars(args)}, f, indent=2)
    print(f"[Summary JSON] Saved: {out_json}")

    print(f"\n=== Done ===")


if __name__ == '__main__':
    main()