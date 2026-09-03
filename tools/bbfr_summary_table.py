import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_GRID = [
    (0.25, 0.50, 5),
    (0.30, 0.50, 5),
    (0.40, 0.50, 5),
    (0.50, 0.50, 5),
    (0.30, 0.40, 5),
    (0.30, 0.60, 5),
    (0.30, 0.50, 3),
    (0.30, 0.50, 10),
]


def tag(score, iou, max_lost):
    return f's{score:.2f}_iou{iou:.2f}_l{max_lost}'.replace('.', 'p')


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_valid_result(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        data = load_json(path)
        return isinstance(data, dict) and data.get('BBFR_Det') is not None
    except Exception:
        return False


def run_one(eval_script, config, ckpt, out_json, score, iou, max_lost, args, name):
    if is_valid_result(out_json) and not args.overwrite:
        print(f'\n[{name}] skip existing result: {out_json}')
        return

    cmd = [
        sys.executable,
        str(eval_script),
        '-c', str(config),
        '-r', str(ckpt),
        '--score-thresh', str(score),
        '--iou-thresh', str(iou),
        '--max-lost', str(max_lost),
        '--out-json', str(out_json),
        '--device', args.device,
        '--num-workers', str(args.num_workers),
        '--score-thresh-keep', str(args.score_thresh_keep),
    ]
    if args.batch_size > 0:
        cmd += ['--batch-size', str(args.batch_size)]
    if args.no_gt:
        cmd += ['--no-gt']

    out_json.parent.mkdir(parents=True, exist_ok=True)
    print(f'\n[{name}] ' + ' '.join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


def get_bbfr(data, metric_name):
    if metric_name == 'BBFR-D':
        return data['BBFR_Det']['BBFR_micro']
    if metric_name == 'BBFR-D macro':
        return data['BBFR_Det']['BBFR_macro']
    if metric_name == 'BBFR-T':
        if data.get('BBFR_T') is not None:
            return data['BBFR_T']['BBFR_T_micro']
        return data['BBFR_GT']['BBFR_GT_micro']
    if metric_name == 'BBFR-T macro':
        if data.get('BBFR_T') is not None:
            return data['BBFR_T']['BBFR_T_macro']
        return data['BBFR_GT']['BBFR_GT_macro']
    raise ValueError(metric_name)


def make_rows(args):
    rows = []
    for score, iou, max_lost in DEFAULT_GRID:
        t = tag(score, iou, max_lost)
        base_json = Path(args.output_dir) / 'baseline' / f'baseline_{t}.json'
        flow_json = Path(args.output_dir) / 'flowrtdetr' / f'flowrtdetr_{t}.json'

        if args.run_baseline:
            run_one(
                args.baseline_eval_script,
                args.baseline_config,
                args.baseline_ckpt,
                base_json,
                score,
                iou,
                max_lost,
                args,
                'baseline',
            )
        if args.run_flowrtdetr:
            run_one(
                args.flowrtdetr_eval_script,
                args.flowrtdetr_config,
                args.flowrtdetr_ckpt,
                flow_json,
                score,
                iou,
                max_lost,
                args,
                'flowrtdetr',
            )

        if args.dry_run:
            continue

        if not base_json.exists() or not flow_json.exists():
            print(f'[WARN] Missing result for {t}: baseline={base_json.exists()}, flow={flow_json.exists()}')
            continue

        base = load_json(base_json)
        flow = load_json(flow_json)
        base_val = float(get_bbfr(base, args.metric))
        flow_val = float(get_bbfr(flow, args.metric))
        change = (flow_val - base_val) / base_val * 100.0 if abs(base_val) > 1e-12 else 0.0
        rows.append({
            'sigma': score,
            'theta_iou': iou,
            'Lmax': max_lost,
            'RT-DETR BBFR-D': base_val,
            'FlowRT-DETR BBFR-D': flow_val,
            'Change': change,
            'baseline_json': str(base_json),
            'flowrtdetr_json': str(flow_json),
        })
    return rows


def save_table(rows, args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / args.csv_name
    md_path = out_dir / args.md_name

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'sigma', 'theta_iou', 'Lmax', 'RT-DETR BBFR-D',
            'FlowRT-DETR BBFR-D', 'Change', 'baseline_json', 'flowrtdetr_json'
        ])
        writer.writeheader()
        writer.writerows(rows)

    lines = []
    lines.append(f'| sigma | thetaIoU | Lmax | RT-DETR {args.metric}↓ | FlowRT-DETR {args.metric}↓ | Change |')
    lines.append('|---:|---:|---:|---:|---:|---:|')
    for r in rows:
        lines.append(
            f"| {r['sigma']:.2f} | {r['theta_iou']:.2f} | {r['Lmax']} | "
            f"{r['RT-DETR BBFR-D']:.2f} | {r['FlowRT-DETR BBFR-D']:.2f} | {r['Change']:+.2f}% |"
        )

    md = '\n'.join(lines)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md + '\n')

    print('\n' + md)
    print(f'\nSaved CSV: {csv_path}')
    print(f'Saved Markdown: {md_path}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run BBFR summary-only evaluation on the paper sensitivity grid and generate a table.'
    )
    parser.add_argument('--baseline-eval-script', required=True,
                        help='Path to eval_bbfr_summary_only.py inside the baseline repo.')
    parser.add_argument('--flowrtdetr-eval-script', required=True,
                        help='Path to eval_bbfr_summary_only.py inside the FlowRT-DETR repo.')
    parser.add_argument('--baseline-config', required=True)
    parser.add_argument('--baseline-ckpt', required=True)
    parser.add_argument('--flowrtdetr-config', required=True)
    parser.add_argument('--flowrtdetr-ckpt', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--metric', default='BBFR-D',
                        choices=['BBFR-D', 'BBFR-D macro', 'BBFR-T', 'BBFR-T macro'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=-1)
    parser.add_argument('--score-thresh-keep', type=float, default=0.05)
    parser.add_argument('--run-baseline', action='store_true',
                        help='Run baseline jobs. Omit this if baseline JSON already exists.')
    parser.add_argument('--run-flowrtdetr', action='store_true',
                        help='Run FlowRT-DETR jobs. Omit this if FlowRT-DETR JSON already exists.')
    parser.add_argument('--no-gt', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-run jobs even when the output json already exists and is valid.')
    parser.add_argument('--csv-name', default='bbfr_sensitivity_table.csv')
    parser.add_argument('--md-name', default='bbfr_sensitivity_table.md')
    return parser.parse_args()


def main():
    args = parse_args()
    rows = make_rows(args)
    if not args.dry_run:
        save_table(rows, args)


if __name__ == '__main__':
    main()
