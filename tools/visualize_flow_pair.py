import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from optical_flow.core.raft import RAFT
from optical_flow.core.utils.utils import InputPadder, load_ckpt
from optical_flow.parser import json_to_args


def pil_to_raw_tensor(img, device):
    arr = np.asarray(img.convert('RGB'), dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def normalize_map(x, p_low=2.0, p_high=98.0):
    x = np.asarray(x, dtype=np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x)
    lo = np.percentile(finite, p_low)
    hi = np.percentile(finite, p_high)
    if hi <= lo + 1e-12:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def turbo_like(gray):
    gray = np.asarray(gray, dtype=np.float32)
    stops = np.array([
        [48, 18, 59],
        [50, 95, 168],
        [37, 165, 130],
        [230, 221, 72],
        [213, 62, 79],
    ], dtype=np.float32) / 255.0
    scaled = np.clip(gray, 0, 1) * (len(stops) - 1)
    idx = np.floor(scaled).astype(np.int32)
    idx = np.clip(idx, 0, len(stops) - 2)
    t = scaled - idx
    rgb = stops[idx] * (1 - t[..., None]) + stops[idx + 1] * t[..., None]
    return (rgb * 255).astype(np.uint8)


def flow_to_rgb(flow):
    if flow.ndim == 4:
        flow = flow[0]
    flow = flow[:2].detach().float().cpu().numpy()
    u, v = flow[0], flow[1]
    mag = np.sqrt(u * u + v * v)
    ang = np.arctan2(v, u)
    hue = (ang + np.pi) / (2 * np.pi)
    val = normalize_map(mag)

    h = (hue * 6.0).astype(np.int32) % 6
    f = hue * 6.0 - np.floor(hue * 6.0)
    p = np.zeros_like(val)
    q = val * (1.0 - f)
    t = val * f

    rgb = np.zeros((*hue.shape, 3), dtype=np.float32)
    choices = [
        np.stack([val, t, p], -1),
        np.stack([q, val, p], -1),
        np.stack([p, val, t], -1),
        np.stack([p, q, val], -1),
        np.stack([t, p, val], -1),
        np.stack([val, p, q], -1),
    ]
    for i in range(6):
        rgb[h == i] = choices[i][h == i]
    return (rgb * 255).astype(np.uint8), mag


def add_label(img, text):
    img = img.convert('RGB') if isinstance(img, Image.Image) else Image.fromarray(img).convert('RGB')
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 30], fill=(255, 255, 255))
    draw.text((8, 8), text, fill=(20, 30, 40))
    return img


def make_panel(paths, out_path):
    imgs = [Image.open(p).convert('RGB') for p in paths]
    h = max(img.height for img in imgs)
    resized = []
    for img in imgs:
        if img.height != h:
            w = int(round(img.width * h / img.height))
            img = img.resize((w, h), Image.BILINEAR)
        resized.append(img)
    margin = 10
    canvas = Image.new('RGB', (sum(i.width for i in resized) + margin * (len(resized) - 1), h), (255, 255, 255))
    x = 0
    for img in resized:
        canvas.paste(img, (x, 0))
        x += img.width + margin
    canvas.save(out_path)


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


def maybe_resize(tensor, size_hw):
    if size_hw is None:
        return tensor, (1.0, 1.0)
    orig_h, orig_w = tensor.shape[-2:]
    target_h, target_w = size_hw
    resized = F.interpolate(tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)
    return resized, (orig_w / float(target_w), orig_h / float(target_h))


@torch.no_grad()
def main(args):
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device.index if device.index is not None else 0)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prev_img = Image.open(args.prev).convert('RGB')
    curr_img = Image.open(args.curr).convert('RGB')
    if prev_img.size != curr_img.size:
        curr_img = curr_img.resize(prev_img.size, Image.BILINEAR)

    prev = pil_to_raw_tensor(prev_img, device)
    curr = pil_to_raw_tensor(curr_img, device)
    orig_h, orig_w = curr.shape[-2:]

    flow_size = tuple(args.flow_size) if args.flow_size else None
    prev_in, (scale_x, scale_y) = maybe_resize(prev, flow_size)
    curr_in, _ = maybe_resize(curr, flow_size)

    flow_args = json_to_args(args.flow_config)
    flow_model = RAFT(flow_args)
    load_ckpt(flow_model, args.flow_ckpt)
    flow_model.to(device).eval()
    for p in flow_model.parameters():
        p.requires_grad = False

    padder = InputPadder(curr_in.shape)
    prev_pad, curr_pad = padder.pad(prev_in, curr_in)
    outputs = flow_model(prev_pad, curr_pad, iters=args.iters, test_mode=True)
    flow = padder.unpad(get_flow_output(outputs))

    if flow.shape[-2:] != (orig_h, orig_w):
        flow = F.interpolate(flow, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        flow[:, 0:1] *= scale_x
        flow[:, 1:2] *= scale_y

    flow_rgb, mag = flow_to_rgb(flow)
    mag_rgb = turbo_like(normalize_map(mag))
    curr_np = np.asarray(curr_img, dtype=np.float32)
    overlay = (0.58 * curr_np + 0.42 * mag_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)

    prev_path = out_dir / 'prev_frame.png'
    curr_path = out_dir / 'curr_frame.png'
    flow_path = out_dir / 'flow_color.png'
    mag_path = out_dir / 'flow_magnitude.png'
    overlay_path = out_dir / 'flow_overlay_on_current.png'
    panel_path = out_dir / 'flow_pair_panel.png'
    npy_path = out_dir / 'flow_dxdy.npy'

    add_label(prev_img, 'Previous frame I_{t-1}').save(prev_path)
    add_label(curr_img, 'Current frame I_t').save(curr_path)
    add_label(flow_rgb, 'SEA-RAFT optical flow F_{t-1->t}').save(flow_path)
    add_label(mag_rgb, 'Flow magnitude').save(mag_path)
    add_label(overlay, 'Flow magnitude overlay').save(overlay_path)
    np.save(npy_path, flow[0, :2].detach().float().cpu().numpy())
    make_panel([prev_path, curr_path, flow_path, mag_path, overlay_path], panel_path)

    stats = {
        'prev': str(args.prev),
        'curr': str(args.curr),
        'image_size': [orig_w, orig_h],
        'flow_size': [curr_in.shape[-1], curr_in.shape[-2]],
        'iters': args.iters,
        'mag_mean': float(np.mean(mag)),
        'mag_p95': float(np.percentile(mag, 95)),
        'mag_max': float(np.max(mag)),
    }
    with open(out_dir / 'summary.txt', 'w', encoding='utf-8') as f:
        for k, v in stats.items():
            f.write(f'{k}: {v}\n')

    print(f'Saved to: {out_dir}')
    print(f' - {flow_path}')
    print(f' - {mag_path}')
    print(f' - {overlay_path}')
    print(f' - {panel_path}')
    print(f' - {npy_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize SEA-RAFT optical flow for two consecutive frames.')
    parser.add_argument('--prev', required=True, help='Previous frame path, e.g. img00619.jpg.')
    parser.add_argument('--curr', required=True, help='Current frame path, e.g. img00620.jpg.')
    parser.add_argument('--flow-config', required=True, help='SEA-RAFT json config.')
    parser.add_argument('--flow-ckpt', required=True, help='SEA-RAFT checkpoint.')
    parser.add_argument('-o', '--output-dir', required=True)
    parser.add_argument('-d', '--device', default='cuda:0')
    parser.add_argument('--iters', type=int, default=4)
    parser.add_argument('--flow-size', nargs=2, type=int, default=None, metavar=('H', 'W'),
                        help='Optional RAFT input size. Omit for full-resolution flow.')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
