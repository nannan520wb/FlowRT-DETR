import argparse
import contextlib
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig


def load_checkpoint_state(path):
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'ema' in checkpoint and isinstance(checkpoint['ema'], dict):
            return checkpoint['ema'].get('module', checkpoint['ema'])
        if 'model' in checkpoint:
            return checkpoint['model']
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
    return checkpoint


def load_detector(config, resume, device, deploy=True):
    cfg = YAMLConfig(config, resume=resume)
    model = cfg.model
    postprocessor = cfg.postprocessor

    if resume:
        state = load_checkpoint_state(resume)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f'[Detector] Loaded checkpoint: {resume}')
        if missing:
            print(f'[Detector] Missing keys: {len(missing)}')
        if unexpected:
            print(f'[Detector] Unexpected keys: {len(unexpected)}')

    if deploy and hasattr(model, 'deploy'):
        model = model.deploy()
    if deploy and hasattr(postprocessor, 'deploy'):
        postprocessor = postprocessor.deploy()

    model = model.to(device).eval()
    postprocessor = postprocessor.to(device).eval() if hasattr(postprocessor, 'to') else postprocessor
    return model, postprocessor


def load_flow_model(config_path, ckpt_path, device):
    if not config_path or not ckpt_path:
        raise ValueError('Flow benchmark needs both --flow-config and --flow-ckpt.')

    from optical_flow.core.raft import RAFT
    from optical_flow.core.utils.utils import load_ckpt
    from optical_flow.parser import json_to_args

    flow_args = json_to_args(config_path)
    model = RAFT(flow_args)
    load_ckpt(model, ckpt_path)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f'[SEA-RAFT] Loaded checkpoint: {ckpt_path}')
    return model


def get_flow_output(outputs):
    if isinstance(outputs, dict):
        if 'flow' in outputs:
            flow = outputs['flow']
        elif 'up_flow' in outputs:
            flow = outputs['up_flow']
        else:
            flow = list(outputs.values())[-1]
    elif isinstance(outputs, (list, tuple)):
        flow = outputs[-1]
    else:
        flow = outputs

    if isinstance(flow, (list, tuple)):
        flow = flow[-1]
    if flow.ndim == 3:
        flow = flow.unsqueeze(0)
    return flow


def cuda_sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def timed_loop(fn, warmup, repeat, device):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
    cuda_sync(device)

    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(repeat):
            fn()
    cuda_sync(device)
    elapsed = time.perf_counter() - start
    return elapsed / repeat


def maybe_amp(enabled, device):
    if enabled and torch.device(device).type == 'cuda':
        return torch.cuda.amp.autocast()
    return contextlib.nullcontext()


def make_inputs(batch_size, det_h, det_w, device):
    # Detector tensors mimic normalized images; flow tensors mimic raw RGB in 0-255.
    det_curr = torch.randn(batch_size, 3, det_h, det_w, device=device)
    det_prev = torch.randn(batch_size, 3, det_h, det_w, device=device)
    raw_curr = torch.rand(batch_size, 3, det_h, det_w, device=device) * 255.0
    raw_prev = torch.rand(batch_size, 3, det_h, det_w, device=device) * 255.0
    orig_sizes = torch.tensor([[det_w, det_h]], device=device).repeat(batch_size, 1)
    return det_curr, det_prev, raw_curr, raw_prev, orig_sizes


def build_flow_3ch(flow_2ch, out_hw, scale_x=1.0, scale_y=1.0):
    out_h, out_w = out_hw
    if flow_2ch.shape[-2:] != (out_h, out_w):
        flow_2ch = F.interpolate(flow_2ch, size=(out_h, out_w), mode='bilinear', align_corners=False)
        flow_2ch[:, 0:1] *= scale_x
        flow_2ch[:, 1:2] *= scale_y

    mag = torch.pow(torch.norm(flow_2ch, dim=1, keepdim=True) + 1e-6, 0.5)
    return torch.cat([flow_2ch[:, 0:1], flow_2ch[:, 1:2], mag], dim=1)


def compute_flow(flow_model, raw_prev, raw_curr, flow_h, flow_w, det_h, det_w, iters):
    from optical_flow.core.utils.utils import InputPadder

    if (flow_h, flow_w) != (det_h, det_w):
        prev_small = F.interpolate(raw_prev, size=(flow_h, flow_w), mode='bilinear', align_corners=False)
        curr_small = F.interpolate(raw_curr, size=(flow_h, flow_w), mode='bilinear', align_corners=False)
    else:
        prev_small = raw_prev
        curr_small = raw_curr

    padder = InputPadder(curr_small.shape)
    prev_pad, curr_pad = padder.pad(prev_small, curr_small)
    outputs = flow_model(prev_pad, curr_pad, iters=iters, test_mode=True)
    flow_2ch = padder.unpad(get_flow_output(outputs))

    scale_x = det_w / float(flow_w)
    scale_y = det_h / float(flow_h)
    return build_flow_3ch(flow_2ch, (det_h, det_w), scale_x=scale_x, scale_y=scale_y)


def detector_forward(model, postprocessor, det_curr, det_prev, flow_3ch, orig_sizes,
                     include_postprocess=False, amp=False, device='cuda'):
    with maybe_amp(amp, device):
        if flow_3ch is None:
            outputs = model(det_curr)
        else:
            outputs = model(det_curr, x_prev=det_prev, flow=flow_3ch)
        if include_postprocess:
            outputs = postprocessor(outputs, orig_sizes)
    return outputs


def find_asg_module(model, name_hint='fusion_s3'):
    matches = []
    for name, module in model.named_modules():
        is_asg = module.__class__.__name__ == 'ScaleAwareGatedFusion'
        has_asg_parts = all(hasattr(module, attr) for attr in ('compress', 'gate_generator', 'flow_transform'))
        if is_asg or has_asg_parts:
            matches.append((name, module))

    if not matches:
        raise RuntimeError(
            'No ASG module found in this config. Use a FlowRTDETR config with '
            "flow_fusion_mode='asg' and fuse_s3=True.")

    if name_hint:
        hinted = [(name, module) for name, module in matches if name_hint in name]
        if hinted:
            return hinted[0]
    return matches[0]


def get_asg_channels(asg_module):
    flow_transform = getattr(asg_module, 'flow_transform', None)
    if hasattr(flow_transform, 'in_channels'):
        return flow_transform.in_channels
    compress = getattr(asg_module, 'compress', None)
    if compress is not None and len(compress) > 0 and hasattr(compress[0], 'in_channels'):
        return compress[0].in_channels // 2
    raise RuntimeError('Could not infer ASG channel count.')


def make_asg_inputs(asg_module, batch_size, det_h, det_w, flow_h, flow_w, stride, device):
    channels = get_asg_channels(asg_module)
    feat_h = max(1, int(round(det_h / float(stride))))
    feat_w = max(1, int(round(det_w / float(stride))))
    feat_curr = torch.randn(batch_size, channels, feat_h, feat_w, device=device)
    feat_prev = torch.randn(batch_size, channels, feat_h, feat_w, device=device)
    flow = torch.randn(batch_size, 3, flow_h, flow_w, device=device)
    return feat_curr, feat_prev, flow


def asg_forward(asg_module, feat_curr, feat_prev, flow, amp=False, device='cuda'):
    with maybe_amp(amp, device):
        return asg_module(feat_curr, feat_prev, flow)


def print_result(title, avg_s, repeat, device, det_size, flow_size=None, flow_iters=None,
                 batch_size=1, include_postprocess=False, amp=False):
    ms = avg_s * 1000.0
    fps = batch_size / avg_s
    print(f'\n[{title}]')
    if torch.device(device).type == 'cuda':
        print(f' - Device: {torch.cuda.get_device_name(torch.device(device))}')
    else:
        print(f' - Device: {device}')
    print(f' - Detector input: {det_size[1]}x{det_size[0]}')
    if flow_size is not None:
        print(f' - Flow input: {flow_size[1]}x{flow_size[0]}')
    if flow_iters is not None:
        print(f' - RAFT iters: {flow_iters}')
    print(f' - Batch size: {batch_size}')
    print(f' - Repeat: {repeat}')
    print(f' - AMP: {amp}')
    print(f' - Postprocess included: {include_postprocess}')
    print(f' - Latency: {ms:.2f} ms/frame')
    print(f' - FPS: {fps:.2f}')


def main(args):
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    det_h, det_w = args.det_size
    if args.flow_size:
        flow_h, flow_w = args.flow_size
    else:
        flow_h = int(round(det_h * args.flow_scale))
        flow_w = int(round(det_w * args.flow_scale))

    det_curr, det_prev, raw_curr, raw_prev, orig_sizes = make_inputs(
        args.batch_size, det_h, det_w, device)

    need_detector = args.mode in ('baseline', 'e2e', 'asg', 'all')
    need_flow = args.mode in ('flow', 'e2e', 'all')

    detector = None
    postprocessor = None
    if need_detector:
        detector, postprocessor = load_detector(
            args.config, args.resume, device, deploy=not args.no_deploy)

    flow_model = None
    if need_flow:
        flow_model = load_flow_model(args.flow_config, args.flow_ckpt, device)

    if args.mode in ('baseline', 'all'):
        def run_baseline():
            return detector_forward(
                detector, postprocessor, det_curr, None, None, orig_sizes,
                include_postprocess=args.include_postprocess,
                amp=args.amp,
                device=device)

        avg = timed_loop(run_baseline, args.warmup, args.repeat, device)
        print_result(
            'Baseline RT-DETR',
            avg,
            args.repeat,
            device,
            det_size=(det_h, det_w),
            batch_size=args.batch_size,
            include_postprocess=args.include_postprocess,
            amp=args.amp)

    if args.mode in ('flow', 'all'):
        def run_flow_only():
            return compute_flow(
                flow_model, raw_prev, raw_curr, flow_h, flow_w, det_h, det_w, args.flow_iters)

        avg = timed_loop(run_flow_only, args.warmup, args.repeat, device)
        print_result(
            'SEA-RAFT Flow Only',
            avg,
            args.repeat,
            device,
            det_size=(det_h, det_w),
            flow_size=(flow_h, flow_w),
            flow_iters=args.flow_iters,
            batch_size=args.batch_size,
            include_postprocess=False,
            amp=False)

    if args.mode in ('e2e', 'all'):
        def run_e2e():
            flow_3ch = compute_flow(
                flow_model, raw_prev, raw_curr, flow_h, flow_w, det_h, det_w, args.flow_iters)
            return detector_forward(
                detector, postprocessor, det_curr, det_prev, flow_3ch, orig_sizes,
                include_postprocess=args.include_postprocess,
                amp=args.amp,
                device=device)

        avg = timed_loop(run_e2e, args.warmup, args.repeat, device)
        print_result(
            'FlowRT-DETR End-to-End',
            avg,
            args.repeat,
            device,
            det_size=(det_h, det_w),
            flow_size=(flow_h, flow_w),
            flow_iters=args.flow_iters,
            batch_size=args.batch_size,
            include_postprocess=args.include_postprocess,
            amp=args.amp)

    if args.mode in ('asg', 'all'):
        asg_name, asg_module = find_asg_module(detector, args.asg_module)
        feat_curr, feat_prev, flow_3ch = make_asg_inputs(
            asg_module, args.batch_size, det_h, det_w, flow_h, flow_w,
            args.asg_stride, device)

        def run_asg_only():
            return asg_forward(
                asg_module, feat_curr, feat_prev, flow_3ch,
                amp=args.amp,
                device=device)

        avg = timed_loop(run_asg_only, args.warmup, args.repeat, device)
        print_result(
            f'ASG Only ({asg_name})',
            avg,
            args.repeat,
            device,
            det_size=(det_h, det_w),
            flow_size=(flow_h, flow_w),
            flow_iters=None,
            batch_size=args.batch_size,
            include_postprocess=False,
            amp=args.amp)
        print(f' - ASG feature: {feat_curr.shape[-1]}x{feat_curr.shape[-2]}, C={feat_curr.shape[1]}')
        print(' - Includes: flow resize/downscale, warp, gate, residual fusion')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark RT-DETR / FlowRT-DETR FPS with dummy tensors.')
    parser.add_argument('-c', '--config', required=True, help='RT-DETR/FlowRT-DETR yaml config.')
    parser.add_argument('-r', '--resume', default=None, help='Detector checkpoint.')
    parser.add_argument('--flow-config', default=None, help='SEA-RAFT json config.')
    parser.add_argument('--flow-ckpt', default=None, help='SEA-RAFT checkpoint.')
    parser.add_argument('--mode', default='all', choices=['baseline', 'flow', 'e2e', 'asg', 'all'])
    parser.add_argument('-d', '--device', default='cuda:0')
    parser.add_argument('--det-size', nargs=2, type=int, default=[544, 960],
                        metavar=('H', 'W'), help='Detector input size. Default: 544 960.')
    parser.add_argument('--flow-size', nargs=2, type=int, default=None,
                        metavar=('H', 'W'), help='Flow input size. Overrides --flow-scale.')
    parser.add_argument('--flow-scale', type=float, default=0.5,
                        help='Flow input scale relative to detector input. Default: 0.5.')
    parser.add_argument('--flow-iters', type=int, default=4)
    parser.add_argument('--asg-module', default='fusion_s3',
                        help='ASG module name substring. Default: fusion_s3.')
    parser.add_argument('--asg-stride', type=int, default=8,
                        help='S3 feature stride relative to detector input. Default: 8.')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeat', type=int, default=100)
    parser.add_argument('--amp', action='store_true', help='Use CUDA autocast for detector.')
    parser.add_argument('--include-postprocess', action='store_true',
                        help='Include RT-DETR postprocessor in detector timing.')
    parser.add_argument('--no-deploy', action='store_true',
                        help='Do not call model.deploy()/postprocessor.deploy().')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())