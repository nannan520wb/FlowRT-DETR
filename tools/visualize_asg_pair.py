""""
1. prev.png   前一帧 I_{t-k}
2. curr.png                当前帧 I_t
3. flow.png                SEA-RAFT 光流可视化
4. flow_magnitude.png      光流幅值图
5. warped.png              warped_feat_prev 可视化
6. gate.png                ASG gate G 可视化
7 fused.png               ASG 处理后的 S3_fused 可视化
8. contrib.png             G * Delta 贡献图
9. delta_heat.png          Delta 热力图
10. asg_process_panel.png  汇总大图，类似 FGFA 风格
11. summary.txt            张量形状和模块信息
"""

import argparse
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from optical_flow.core.raft import RAFT
from optical_flow.core.utils.utils import InputPadder, load_ckpt
from optical_flow.parser import json_to_args
from src.core import YAMLConfig
from src.zoo.rtdetr.rtdetr import ScaleAwareGatedFusion


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


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


def pil_to_raw_tensor(img, device):
    arr = np.asarray(img.convert('RGB'), dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def pad_to_size(img, size_hw, fill=(0, 0, 0)):
    target_h, target_w = size_hw
    img = img.convert('RGB')
    w, h = img.size
    if h > target_h or w > target_w:
        scale = min(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        w, h = img.size
    canvas = Image.new('RGB', (target_w, target_h), fill)
    canvas.paste(img, (0, 0))
    return canvas


def normalize_img_tensor(img, device):
    arr = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(STD, dtype=torch.float32).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device)


def denorm_to_uint8(x):
    x = x.detach().float().cpu()[0]
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    x = (x * std + mean).clamp(0, 1)
    return (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def normalize_map(x, p_low=2.0, p_high=98.0):
    x = np.asarray(x, dtype=np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x)
    lo = np.percentile(finite, p_low)
    hi = np.percentile(finite, p_high)
    if hi <= lo + 1e-12:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


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


def feature_to_heat(feat, out_hw):
    if feat.ndim == 4:
        feat = feat[0]
    heat = feat.detach().float().abs().mean(dim=0, keepdim=True).unsqueeze(0)
    heat = F.interpolate(heat, size=out_hw, mode='bilinear', align_corners=False)[0, 0]
    return heat.cpu().numpy()


def single_map_to_heat(x, out_hw):
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        x = x.mean(dim=0, keepdim=True)
    x = x.detach().float().unsqueeze(0)
    x = F.interpolate(x, size=out_hw, mode='bilinear', align_corners=False)[0, 0]
    return x.cpu().numpy()


def overlay(base_rgb, heat, alpha=0.48):
    heat_rgb = turbo_like(normalize_map(heat))
    return (base_rgb.astype(np.float32) * (1 - alpha) + heat_rgb.astype(np.float32) * alpha).astype(np.uint8)


def flow_to_rgb(flow):
    if flow.ndim == 4:
        flow = flow[0]
    flow = flow[:2].detach().float().cpu().numpy()
    u, v = flow[0], flow[1]
    mag = np.sqrt(u * u + v * v)
    ang = np.arctan2(v, u)
    hue = (ang + np.pi) / (2 * np.pi)
    sat = np.ones_like(hue)
    val = normalize_map(mag)

    h = (hue * 6.0).astype(np.int32)
    f = hue * 6.0 - h
    p = val * (1.0 - sat)
    q = val * (1.0 - f * sat)
    t = val * (1.0 - (1.0 - f) * sat)
    h = h % 6
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
    img = img.convert('RGB') if isinstance(img, Image.Image) else Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 30], fill=(255, 255, 255))
    draw.text((8, 7), text, fill=(20, 30, 40), font=ImageFont.load_default())
    return img


def arrow(draw, start, end, fill=(40, 40, 40), width=3):
    draw.line([start, end], fill=fill, width=width)
    sx, sy = start
    ex, ey = end
    vec = np.array([ex - sx, ey - sy], dtype=np.float32)
    norm = np.linalg.norm(vec) + 1e-6
    vec = vec / norm
    perp = np.array([-vec[1], vec[0]])
    tip = np.array([ex, ey])
    left = tip - 12 * vec + 6 * perp
    right = tip - 12 * vec - 6 * perp
    draw.polygon([tuple(tip), tuple(left), tuple(right)], fill=fill)


def compute_flow(prev_img, curr_img, args, device):
    flow_args = json_to_args(args.flow_config)
    flow_model = RAFT(flow_args)
    load_ckpt(flow_model, args.flow_ckpt)
    flow_model.to(device).eval()
    for p in flow_model.parameters():
        p.requires_grad = False

    prev_raw = pil_to_raw_tensor(prev_img, device)
    curr_raw = pil_to_raw_tensor(curr_img, device)
    padder = InputPadder(curr_raw.shape)
    prev_p, curr_p = padder.pad(prev_raw, curr_raw)
    with torch.no_grad():
        outputs = flow_model(prev_p, curr_p, iters=args.flow_iters, test_mode=True)
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
    return padder.unpad(flow)


def patch_asg_capture(model, wanted_name=None):
    captures = {}

    def make_forward(name):
        def forward(self, feat_curr, feat_prev, flow):
            b, c, h, w = feat_curr.shape
            _, _, h_orig, w_orig = flow.shape
            stride_x = w_orig / float(w)
            stride_y = h_orig / float(h)
            flow_dxdy = flow[:, :2]
            flow_resized = F.interpolate(flow_dxdy, size=(h, w), mode='bilinear', align_corners=False)
            flow_resized[:, 0:1] /= stride_x
            flow_resized[:, 1:2] /= stride_y

            xx = torch.arange(0, w, device=feat_curr.device).view(1, -1).repeat(h, 1)
            yy = torch.arange(0, h, device=feat_curr.device).view(-1, 1).repeat(1, w)
            grid = torch.stack((xx, yy), 2).float().unsqueeze(0).repeat(b, 1, 1, 1)

            # Keep the same ASG behavior as the model code currently uses.
            vgrid = grid - flow_resized.permute(0, 2, 3, 1)
            vgrid[..., 0] = 2.0 * vgrid[..., 0] / max(w - 1, 1) - 1.0
            vgrid[..., 1] = 2.0 * vgrid[..., 1] / max(h - 1, 1) - 1.0
            warped_feat_prev = F.grid_sample(
                feat_prev, vgrid, mode='bilinear', padding_mode='zeros', align_corners=True)

            cat_feat = torch.cat([feat_curr, warped_feat_prev], dim=1)
            compressed_feat = self.compress(cat_feat)
            raw_gate = torch.sigmoid(self.gate_generator(compressed_feat))
            flow_mag = torch.norm(flow_resized, dim=1, keepdim=True)
            motion_mask = torch.sigmoid((flow_mag - self.motion_thr) / max(self.motion_temp, 1e-6))

            if hasattr(self, 'sim_thr'):
                feat_curr_norm = F.normalize(feat_curr.detach(), dim=1, eps=1e-6)
                warped_norm = F.normalize(warped_feat_prev.detach(), dim=1, eps=1e-6)
                sim = (feat_curr_norm * warped_norm).sum(dim=1, keepdim=True)
                sim_mask = torch.sigmoid((sim - self.sim_thr) / max(self.sim_temp, 1e-6))
            else:
                sim = None
                sim_mask = 1.0

            gate = self.max_gate * raw_gate * motion_mask * sim_mask
            if getattr(self, 'flow_transform', None) is not None:
                try:
                    delta = self.flow_transform(warped_feat_prev - feat_curr)
                except Exception:
                    delta = self.flow_transform(warped_feat_prev)
            else:
                delta = warped_feat_prev - feat_curr
            contribution = gate * delta
            s_fused = feat_curr + contribution

            captures[name] = {
                'feat_curr': feat_curr.detach(),
                'feat_prev': feat_prev.detach(),
                'flow_resized': flow_resized.detach(),
                'warped_feat_prev': warped_feat_prev.detach(),
                'raw_gate': raw_gate.detach(),
                'gate': gate.detach(),
                'delta': delta.detach(),
                'contribution': contribution.detach(),
                's_fused': s_fused.detach(),
                'sim': None if sim is None else sim.detach(),
            }
            return s_fused
        return forward

    for name, module in model.named_modules():
        if isinstance(module, ScaleAwareGatedFusion) and (wanted_name is None or wanted_name in name):
            module.forward = types.MethodType(make_forward(name), module)
    return captures


def save_panel(out_dir, images):
    tile_w, tile_h = 260, 180
    margin = 28
    panel_w = tile_w * 3 + margin * 4
    panel_h = tile_h * 3 + margin * 4
    panel = Image.new('RGB', (panel_w, panel_h), (255, 255, 255))
    draw = ImageDraw.Draw(panel)

    positions = {
        'prev': (margin, margin),
        'curr': (margin, margin * 2 + tile_h),
        'flow': (margin, margin * 3 + tile_h * 2),
        'warped': (margin * 2 + tile_w, margin),
        'gate': (margin * 2 + tile_w, margin * 2 + tile_h),
        'fused': (margin * 3 + tile_w * 2, margin),
        'contrib': (margin * 3 + tile_w * 2, margin * 2 + tile_h),
        'det': (margin * 3 + tile_w * 2, margin * 3 + tile_h * 2),
    }
    for key, img in images.items():
        if key not in positions:
            continue
        img = img.resize((tile_w, tile_h), Image.BILINEAR)
        panel.paste(img, positions[key])

    arrow(draw, (margin + tile_w, margin + tile_h // 2), (margin * 2 + tile_w, margin + tile_h // 2))
    arrow(draw, (margin + tile_w, margin * 3 + tile_h * 2 + tile_h // 2),
          (margin * 2 + tile_w, margin + tile_h // 2), fill=(210, 100, 20))
    arrow(draw, (margin * 2 + tile_w + tile_w, margin + tile_h // 2),
          (margin * 3 + tile_w * 2, margin + tile_h // 2))
    arrow(draw, (margin * 2 + tile_w + tile_w, margin * 2 + tile_h + tile_h // 2),
          (margin * 3 + tile_w * 2, margin + tile_h // 2), fill=(80, 70, 190))
    arrow(draw, (margin * 3 + tile_w * 2 + tile_w // 2, margin + tile_h),
          (margin * 3 + tile_w * 2 + tile_w // 2, margin * 2 + tile_h), fill=(80, 70, 190))
    panel.save(out_dir / 'asg_process_panel.png')


def main(args):
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prev_img = Image.open(args.prev).convert('RGB')
    curr_img = Image.open(args.curr).convert('RGB')
    target_size = tuple(args.input_size)
    prev_pad = pad_to_size(prev_img, target_size)
    curr_pad = pad_to_size(curr_img, target_size)

    flow = compute_flow(prev_pad, curr_pad, args, device)
    dx, dy = flow[:, 0:1], flow[:, 1:2]
    mag = torch.norm(flow[:, :2], dim=1, keepdim=True)
    flow_3ch = torch.cat([dx, dy, mag], dim=1).to(device)

    cfg = YAMLConfig(args.config, resume=args.resume)
    state = load_checkpoint_state(args.resume)
    cfg.model.load_state_dict(state, strict=False)
    model = cfg.model.to(device).eval()
    captures = patch_asg_capture(model, args.module)

    curr_tensor = normalize_img_tensor(curr_pad, device)
    prev_tensor = normalize_img_tensor(prev_pad, device)
    with torch.no_grad():
        outputs = model(curr_tensor, x_prev=prev_tensor, flow=flow_3ch)

    if not captures:
        raise RuntimeError("No ASG capture found. Check that the config uses flow_fusion_mode='asg' and fuse_s3=True.")
    module_name = sorted(captures.keys())[0]
    rec = captures[module_name]

    base = denorm_to_uint8(curr_tensor)
    h, w = base.shape[:2]
    flow_rgb, flow_mag = flow_to_rgb(flow)
    flow_rgb = np.asarray(Image.fromarray(flow_rgb).resize((w, h), Image.BILINEAR))

    warped_heat = feature_to_heat(rec['warped_feat_prev'], (h, w))
    fused_heat = feature_to_heat(rec['s_fused'], (h, w))
    gate_heat = single_map_to_heat(rec['gate'].mean(dim=1, keepdim=True), (h, w))
    contrib_heat = feature_to_heat(rec['contribution'], (h, w))
    delta_heat = feature_to_heat(rec['delta'], (h, w))

    images = {
        'prev': add_label(prev_pad, 'Previous frame I_{t-k}'),
        'curr': add_label(curr_pad, 'Current frame I_t'),
        'flow': add_label(flow_rgb, 'SEA-RAFT flow F_{t-k->t}'),
        'warped': add_label(overlay(base, warped_heat), 'warped_feat_prev'),
        'gate': add_label(overlay(base, gate_heat), 'ASG gate G'),
        'fused': add_label(overlay(base, fused_heat), 'ASG output S3_fused'),
        'contrib': add_label(overlay(base, contrib_heat), 'motion contribution G*Delta'),
        'det': add_label(base, 'current frame / detection input'),
    }
    for name, img in images.items():
        img.save(out_dir / f'{name}.png')
    Image.fromarray(turbo_like(normalize_map(delta_heat))).save(out_dir / 'delta_heat.png')
    Image.fromarray(turbo_like(normalize_map(flow_mag))).save(out_dir / 'flow_magnitude.png')
    save_panel(out_dir, images)

    with open(out_dir / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write(f'captured_module: {module_name}\n')
        f.write(f'prev_image: {args.prev}\n')
        f.write(f'curr_image: {args.curr}\n')
        f.write(f'flow_shape: {tuple(flow_3ch.shape)}\n')
        f.write(f'warped_feat_prev_shape: {tuple(rec["warped_feat_prev"].shape)}\n')
        f.write(f'gate_shape: {tuple(rec["gate"].shape)}\n')
        f.write(f'fused_shape: {tuple(rec["s_fused"].shape)}\n')
        f.write(f'pred_keys: {list(outputs.keys()) if isinstance(outputs, dict) else type(outputs)}\n')

    print(f'Saved ASG pair visualization to: {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prev', required=True, help='Previous/reference frame image.')
    parser.add_argument('--curr', required=True, help='Current frame image.')
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-r', '--resume', required=True)
    parser.add_argument('--flow-ckpt', required=True)
    parser.add_argument('--flow-config', required=True)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-o', '--output-dir', default='output/asg_pair_vis')
    parser.add_argument('--input-size', type=int, nargs=2, default=[544, 960], metavar=('H', 'W'))
    parser.add_argument('--flow-iters', type=int, default=12)
    parser.add_argument('--module', default='fusion_s3', help='ASG module name substring to capture.')
    args = parser.parse_args()
    main(args)
