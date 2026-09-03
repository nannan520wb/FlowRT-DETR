import argparse
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig
from src.zoo.rtdetr.rtdetr import MotionGuidedFeatureRecalibration, ScaleAwareGatedFusion


DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


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


def tensor_to_rgb_uint8(image, mean=DEFAULT_MEAN, std=DEFAULT_STD):
    image = image.detach().float().cpu()
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    image = image * std + mean
    image = image.clamp(0, 1)
    return (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def normalize_map(x, p_low=2.0, p_high=98.0):
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    vals = x[finite]
    lo = np.percentile(vals, p_low)
    hi = np.percentile(vals, p_high)
    if hi <= lo + 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0)


def turbo_like_colormap(gray):
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
    rgb = stops[idx] * (1.0 - t[..., None]) + stops[idx + 1] * t[..., None]
    return (rgb * 255.0).astype(np.uint8)


def resize_map_to_image(x, size_hw):
    tensor = torch.as_tensor(x, dtype=torch.float32)[None, None]
    tensor = F.interpolate(tensor, size=size_hw, mode='bilinear', align_corners=False)[0, 0]
    return tensor.numpy()


def save_heatmap_and_overlay(base_rgb, heat, out_prefix, title):
    h, w = base_rgb.shape[:2]
    heat = resize_map_to_image(heat, (h, w))
    heat_norm = normalize_map(heat)
    heat_rgb = turbo_like_colormap(heat_norm)
    overlay = (0.55 * base_rgb.astype(np.float32) + 0.45 * heat_rgb.astype(np.float32)).astype(np.uint8)

    heat_img = Image.fromarray(heat_rgb)
    overlay_img = Image.fromarray(overlay)
    for img in (heat_img, overlay_img):
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, min(w, 520), 32], fill=(255, 255, 255))
        draw.text((10, 8), title, fill=(20, 30, 40))

    heat_path = f'{out_prefix}_heat.png'
    overlay_path = f'{out_prefix}_overlay.png'
    heat_img.save(heat_path)
    overlay_img.save(overlay_path)
    return heat_path, overlay_path


def make_panel(paths, out_path):
    images = [Image.open(p).convert('RGB') for p in paths]
    if not images:
        return
    w, h = images[0].size
    cols = 3 if len(images) > 4 else 2
    rows = int(np.ceil(len(images) / cols))
    canvas = Image.new('RGB', (w * cols, h * rows), (255, 255, 255))
    for i, img in enumerate(images):
        img = img.resize((w, h))
        canvas.paste(img, ((i % cols) * w, (i // cols) * h))
    canvas.save(out_path)


def patch_asg_forward(model):
    records = {}

    def make_forward(module_name):
        def forward(self, feat_curr, feat_prev, flow):
            b, c, h, w = feat_curr.shape
            _, _, h_orig, w_orig = flow.shape

            stride_x = w_orig / float(w)
            stride_y = h_orig / float(h)
            flow_dxdy = flow[:, :2, :, :]
            flow_resized = F.interpolate(flow_dxdy, size=(h, w), mode='bilinear', align_corners=False)
            flow_resized[:, 0:1, :, :] /= stride_x
            flow_resized[:, 1:2, :, :] /= stride_y

            xx = torch.arange(0, w, device=feat_curr.device).view(1, -1).repeat(h, 1)
            yy = torch.arange(0, h, device=feat_curr.device).view(-1, 1).repeat(1, w)
            grid = torch.stack((xx, yy), 2).float().unsqueeze(0).repeat(b, 1, 1, 1)
            vgrid = grid - flow_resized.permute(0, 2, 3, 1)
            valid_mask = (
                (vgrid[..., 0:1] >= 0) & (vgrid[..., 0:1] <= w - 1) &
                (vgrid[..., 1:2] >= 0) & (vgrid[..., 1:2] <= h - 1)
            ).permute(0, 3, 1, 2).to(feat_curr.dtype)
            vgrid[..., 0] = 2.0 * vgrid[..., 0] / max(w - 1, 1) - 1.0
            vgrid[..., 1] = 2.0 * vgrid[..., 1] / max(h - 1, 1) - 1.0
            warped_feat_prev = F.grid_sample(
                feat_prev, vgrid, mode='bilinear', padding_mode='zeros', align_corners=True)

            cat_feat = torch.cat([feat_curr, warped_feat_prev], dim=1)
            compressed_feat = self.compress(cat_feat)
            raw_gate = torch.sigmoid(self.gate_generator(compressed_feat))
            flow_mag = torch.norm(flow_resized, dim=1, keepdim=True)
            motion_mask = torch.sigmoid((flow_mag - self.motion_thr) / max(self.motion_temp, 1e-6))
            feat_curr_norm = F.normalize(feat_curr.detach(), dim=1, eps=1e-6)
            warped_norm = F.normalize(warped_feat_prev.detach(), dim=1, eps=1e-6)
            sim = (feat_curr_norm * warped_norm).sum(dim=1, keepdim=True)
            sim_mask = torch.sigmoid((sim - self.sim_thr) / max(self.sim_temp, 1e-6))
            gate = self.max_gate * raw_gate * motion_mask * sim_mask * valid_mask
            delta = self.flow_transform(warped_feat_prev - feat_curr)
            contribution = gate * delta
            s_fused = feat_curr + contribution

            with torch.no_grad():
                feat_norm = feat_curr.detach().float().norm(dim=1)
                contrib_norm = contribution.detach().float().norm(dim=1)
                rec = {
                    'flow_mag': flow_resized.detach().float().norm(dim=1).cpu(),
                    'gate': gate.detach().float().mean(dim=1).cpu(),
                    'sim': sim.detach().float().squeeze(1).cpu(),
                    'contribution': contrib_norm.cpu(),
                    'ratio': (contrib_norm / (feat_norm + 1e-6)).clamp(0, 10).cpu(),
                }
                records.setdefault(module_name, []).append(rec)
            return s_fused
        return forward

    for name, module in model.named_modules():
        if isinstance(module, ScaleAwareGatedFusion):
            module.forward = types.MethodType(make_forward(name), module)
    return records


def patch_recalibration_forward(model, records):
    def make_forward(module_name):
        def forward(self, feat_curr, flow):
            b, c, h, w = feat_curr.shape
            _, _, h_orig, w_orig = flow.shape

            stride_x = w_orig / float(w)
            stride_y = h_orig / float(h)
            flow_dxdy = flow[:, :2, :, :]
            flow_resized = F.interpolate(flow_dxdy, size=(h, w), mode='bilinear', align_corners=False)
            flow_resized[:, 0:1, :, :] /= stride_x
            flow_resized[:, 1:2, :, :] /= stride_y

            flow_mag = torch.norm(flow_resized, dim=1, keepdim=True)
            flow_input = torch.cat([flow_resized, torch.log1p(flow_mag)], dim=1)
            flow_feat = self.flow_proj(flow_input)

            raw_gate = torch.sigmoid(self.gate_net(torch.cat([feat_curr, flow_feat], dim=1)))
            motion_mask = torch.sigmoid((flow_mag - self.motion_thr) / max(self.motion_temp, 1e-6))
            gate = self.alpha * raw_gate * motion_mask
            delta = self.delta(feat_curr)
            contribution = gate * delta
            out = feat_curr + contribution

            with torch.no_grad():
                feat_norm = feat_curr.detach().float().norm(dim=1)
                contrib_norm = contribution.detach().float().norm(dim=1)
                rec = {
                    'flow_mag': flow_mag.detach().float().squeeze(1).cpu(),
                    'gate': gate.detach().float().mean(dim=1).cpu(),
                    'contribution': contrib_norm.cpu(),
                    'ratio': (contrib_norm / (feat_norm + 1e-6)).clamp(0, 10).cpu(),
                }
                records.setdefault(module_name, []).append(rec)
            return out
        return forward

    for name, module in model.named_modules():
        if isinstance(module, MotionGuidedFeatureRecalibration):
            module.forward = types.MethodType(make_forward(name), module)
    return records


def iter_visualization_batches(data_loader, device, max_samples):
    seen = 0
    for (samples, prev_samples, flow_img), targets in data_loader:
        samples = samples.to(device)
        prev_samples = prev_samples.to(device)
        flow_img = flow_img.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        yield samples, prev_samples, flow_img, targets, seen
        seen += samples.shape[0]
        if seen >= max_samples:
            break


def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = YAMLConfig(args.config, resume=args.resume)
    state = load_checkpoint_state(args.resume)
    missing, unexpected = cfg.model.load_state_dict(state, strict=False)
    if missing:
        print(f'[Warn] Missing keys: {len(missing)}')
    if unexpected:
        print(f'[Warn] Unexpected keys: {len(unexpected)}')

    model = cfg.model.to(args.device).eval()
    records = patch_asg_forward(model)
    records = patch_recalibration_forward(model, records)

    # Visualization only needs a few samples. On NFS/CPFS, multi-worker
    # DataLoader finalizers can leave busy .nfs temp files, so keep it single
    # process unless the user explicitly asks otherwise.
    if args.num_workers is not None and hasattr(cfg.val_dataloader, 'num_workers'):
        cfg.val_dataloader.num_workers = args.num_workers

    saved = 0
    with torch.no_grad():
        for samples, prev_samples, flow_img, targets, start_index in iter_visualization_batches(
                cfg.val_dataloader, args.device, args.max_samples):
            before_counts = {k: len(v) for k, v in records.items()}
            _ = model(samples, x_prev=prev_samples, flow=flow_img)

            batch_size = samples.shape[0]
            for module_name, module_records in records.items():
                new_records = module_records[before_counts.get(module_name, 0):]
                if not new_records:
                    continue
                rec = new_records[-1]
                scale_name = module_name.replace('.', '_')
                for bi in range(batch_size):
                    global_idx = start_index + bi
                    if global_idx >= args.max_samples:
                        break
                    base_rgb = tensor_to_rgb_uint8(samples[bi])
                    sample_dir = out_dir / f'sample_{global_idx:04d}'
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(base_rgb).save(sample_dir / 'current_frame.png')

                    panel_inputs = []
                    all_metrics = [
                        ('flow_mag', 'Flow magnitude at feature scale'),
                        ('gate', 'Gate weight G'),
                        ('sim', 'Current-warped feature similarity'),
                        ('contribution', 'Motion contribution ||G * Delta||'),
                        ('ratio', 'Contribution ratio ||G * Delta|| / ||S_t||'),
                    ]
                    metrics = [(key, title) for key, title in all_metrics if key in rec]
                    for key, title in metrics:
                        heat = rec[key][bi].numpy()
                        prefix = sample_dir / f'{scale_name}_{key}'
                        _, overlay_path = save_heatmap_and_overlay(
                            base_rgb, heat, str(prefix), f'{module_name}: {title}')
                        panel_inputs.append(overlay_path)

                    make_panel(panel_inputs, sample_dir / f'{scale_name}_panel.png')
                    saved += 1

    print(f'Saved ASG visualizations for {saved} module-samples under: {out_dir}')
    if not records:
        print('[Warn] No ScaleAwareGatedFusion module was found. Check that the config uses FlowRTDETR.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='RT-DETR/FlowRT-DETR yaml config.')
    parser.add_argument('-r', '--resume', required=True, help='Trained 5.3 checkpoint path.')
    parser.add_argument('-d', '--device', default='cuda', help='cuda or cpu.')
    parser.add_argument('-o', '--output-dir', default='output/asg_vis_5_3')
    parser.add_argument('--max-samples', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()
    main(args)
