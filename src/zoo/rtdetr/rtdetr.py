import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.core import register


class ScaleAwareGatedFusion(nn.Module):
    """Identity-initialized, scale-aware gated temporal residual."""
    def __init__(self, channels, max_gate=0.2, motion_thr=0.5, motion_temp=0.25,
                 sim_thr=0.0, sim_temp=0.1):
        super().__init__()
        self.max_gate = max_gate
        self.motion_thr = motion_thr
        self.motion_temp = motion_temp
        self.sim_thr = sim_thr
        self.sim_temp = sim_temp

        self.compress = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(32, channels), num_channels=channels),
            nn.ReLU(inplace=True)
        )

        gate_gn_groups = min(16, channels // 2)
        self.gate_generator = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=gate_gn_groups, num_channels=channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, kernel_size=1)
        )

        # Near-zero gate and zero temporal projection give an exact identity
        # at initialization.
        nn.init.constant_(self.gate_generator[-1].weight, 0)
        nn.init.constant_(self.gate_generator[-1].bias, -7.0)
        self.flow_transform = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        nn.init.constant_(self.flow_transform.weight, 0)
        nn.init.constant_(self.flow_transform.bias, 0)
    def forward(self, feat_curr, feat_prev, flow):
        B, C, H, W = feat_curr.shape
        _, _, H_orig, W_orig = flow.shape
        stride_x = W_orig / float(W)
        stride_y = H_orig / float(H)
        flow_dxdy = flow[:, :2, :, :]
        flow_resized = F.interpolate(flow_dxdy, size=(H, W), mode='bilinear', align_corners=False)
        flow_resized[:, 0:1, :, :] /= stride_x
        flow_resized[:, 1:2, :, :] /= stride_y
        xx = torch.arange(0, W, device=feat_curr.device).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device=feat_curr.device).view(-1, 1).repeat(1, W)
        grid = torch.stack((xx, yy), 2).float().unsqueeze(0).repeat(B, 1, 1, 1)
        # RAFT(img_prev, img_curr) estimates forward flow prev -> curr.
        # grid_sample needs previous-frame source coordinates for each current
        # location, so use an approximate backward lookup: current grid - flow.
        vgrid = grid - flow_resized.permute(0, 2, 3, 1)
        valid_mask = (
            (vgrid[..., 0:1] >= 0) & (vgrid[..., 0:1] <= W - 1) &
            (vgrid[..., 1:2] >= 0) & (vgrid[..., 1:2] <= H - 1)
        ).permute(0, 3, 1, 2).to(feat_curr.dtype)
        vgrid[..., 0] = 2.0 * vgrid[..., 0] / max(W - 1, 1) - 1.0
        vgrid[..., 1] = 2.0 * vgrid[..., 1] / max(H - 1, 1) - 1.0
        warped_feat_prev = F.grid_sample(feat_prev, vgrid, mode='bilinear', padding_mode='zeros', align_corners=True)
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
        s_fused = feat_curr + gate * delta

        return s_fused


class MotionGuidedFeatureRecalibration(nn.Module):
    """
    Scheme B: optical-flow-guided current-feature recalibration.

    Flow decides where to enhance, while the enhanced content still comes from
    the current-frame feature. This avoids injecting misaligned previous-frame
    features into high-resolution S3.
    """
    def __init__(self, channels, alpha=0.1, motion_thr=0.5, motion_temp=0.25, flow_hidden=32):
        super().__init__()
        self.alpha = alpha
        self.motion_thr = motion_thr
        self.motion_temp = motion_temp

        flow_groups = min(8, flow_hidden)
        self.flow_proj = nn.Sequential(
            nn.Conv2d(3, flow_hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=flow_groups, num_channels=flow_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(flow_hidden, flow_hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=flow_groups, num_channels=flow_hidden),
            nn.ReLU(inplace=True),
        )

        gate_channels = max(channels // 2, 32)
        gate_groups = min(16, gate_channels)
        self.gate_net = nn.Sequential(
            nn.Conv2d(channels + flow_hidden, gate_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=gate_groups, num_channels=gate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_channels, channels, kernel_size=1, bias=True),
        )

        self.delta = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        nn.init.constant_(self.gate_net[-1].weight, 0)
        nn.init.constant_(self.gate_net[-1].bias, -7.0)
        nn.init.constant_(self.delta.weight, 0)
        nn.init.constant_(self.delta.bias, 0)

    def forward(self, feat_curr, flow):
        B, C, H, W = feat_curr.shape
        _, _, H_orig, W_orig = flow.shape

        stride_x = W_orig / float(W)
        stride_y = H_orig / float(H)
        flow_dxdy = flow[:, :2, :, :]
        flow_resized = F.interpolate(flow_dxdy, size=(H, W), mode='bilinear', align_corners=False)
        flow_resized[:, 0:1, :, :] /= stride_x
        flow_resized[:, 1:2, :, :] /= stride_y

        flow_mag = torch.norm(flow_resized, dim=1, keepdim=True)
        flow_input = torch.cat([flow_resized, torch.log1p(flow_mag)], dim=1)
        flow_feat = self.flow_proj(flow_input)

        raw_gate = torch.sigmoid(self.gate_net(torch.cat([feat_curr, flow_feat], dim=1)))
        motion_mask = torch.sigmoid((flow_mag - self.motion_thr) / max(self.motion_temp, 1e-6))
        gate = self.alpha * raw_gate * motion_mask

        delta = self.delta(feat_curr)
        return feat_curr + gate * delta


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.multi_scale = multi_scale

    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)
        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


@register
class FlowRTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None,
                 fuse_s3=True, fuse_s4=False, fuse_s5=False,
                 asg_max_gate=0.2, asg_motion_thr=0.5, asg_motion_temp=0.25,
                 asg_sim_thr=0.0, asg_sim_temp=0.1,
                 flow_fusion_mode='asg', recalib_alpha=0.1,
                 recalib_motion_thr=0.5, recalib_motion_temp=0.25,
                 recalib_flow_hidden=32):
        """Build a flow-guided RT-DETR with configurable feature levels."""
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.multi_scale = multi_scale

        self.fuse_s3 = fuse_s3
        self.fuse_s4 = fuse_s4
        self.fuse_s5 = fuse_s5
        self.asg_max_gate = asg_max_gate
        self.asg_motion_thr = asg_motion_thr
        self.asg_motion_temp = asg_motion_temp
        self.asg_sim_thr = asg_sim_thr
        self.asg_sim_temp = asg_sim_temp
        self.flow_fusion_mode = flow_fusion_mode
        self.recalib_alpha = recalib_alpha
        self.recalib_motion_thr = recalib_motion_thr
        self.recalib_motion_temp = recalib_motion_temp
        self.recalib_flow_hidden = recalib_flow_hidden

        if flow_fusion_mode not in ('asg', 'recalibration'):
            raise ValueError(f"Unsupported flow_fusion_mode: {flow_fusion_mode}")

        channels = getattr(self.backbone, 'out_channels', None)
        if channels is None or len(channels) != 3:
            raise ValueError(
                'FlowRTDETR expects a backbone exposing exactly three out_channels')
        self.s3_ch, self.s4_ch, self.s5_ch = channels

        if self.fuse_s3:
            self.fusion_s3 = self._build_flow_fusion(self.s3_ch)
        if self.fuse_s4:
            self.fusion_s4 = self._build_flow_fusion(self.s4_ch)
        if self.fuse_s5:
            self.fusion_s5 = self._build_flow_fusion(self.s5_ch)

    def _build_flow_fusion(self, channels):
        if self.flow_fusion_mode == 'recalibration':
            return MotionGuidedFeatureRecalibration(
                channels,
                alpha=self.recalib_alpha,
                motion_thr=self.recalib_motion_thr,
                motion_temp=self.recalib_motion_temp,
                flow_hidden=self.recalib_flow_hidden)

        return ScaleAwareGatedFusion(
            channels,
            self.asg_max_gate,
            self.asg_motion_thr,
            self.asg_motion_temp,
            self.asg_sim_thr,
            self.asg_sim_temp)

    def _apply_flow_fusion(self, module, feat_curr, feat_prev, flow):
        if self.flow_fusion_mode == 'recalibration':
            return module(feat_curr, flow)
        return module(feat_curr, feat_prev, flow)

    def forward(self, x, x_prev=None, flow=None, prev_boxes=None, targets=None):
        if self.multi_scale and self.training:
            target_h = np.random.choice(self.multi_scale)
            B, C, H, W = x.shape
            ratio = target_h / H
            target_w = int(W * ratio)
            target_w = int(round(target_w / 32) * 32)

            x = F.interpolate(x, size=[target_h, target_w], mode='bilinear', align_corners=False)
            if x_prev is not None:
                x_prev = F.interpolate(x_prev, size=[target_h, target_w], mode='bilinear', align_corners=False)

            if flow is not None:
                orig_h, orig_w = flow.shape[-2:]
                scale_x = target_w / orig_w
                scale_y = target_h / orig_h

                flow = F.interpolate(flow, size=[target_h, target_w], mode='bilinear', align_corners=False)
                if flow.size(1) >= 2:
                    flow_u = flow[:, 0:1, :, :] * scale_x
                    flow_v = flow[:, 1:2, :, :] * scale_y
                    if flow.size(1) == 3:
                        scale_avg = (scale_x + scale_y) / 2.0
                        flow_mag = flow[:, 2:3, :, :] * scale_avg
                        flow = torch.cat([flow_u, flow_v, flow_mag], dim=1)
                    else:
                        flow = torch.cat([flow_u, flow_v], dim=1)

        s3_curr, s4_curr, s5_curr = self.backbone(x)

        if flow is not None and (self.flow_fusion_mode == 'recalibration' or x_prev is not None):
            if self.flow_fusion_mode == 'asg':
                s3_prev, s4_prev, s5_prev = self.backbone(x_prev)
            else:
                s3_prev, s4_prev, s5_prev = None, None, None

            s3_out = self._apply_flow_fusion(self.fusion_s3, s3_curr, s3_prev, flow) if self.fuse_s3 else s3_curr
            s4_out = self._apply_flow_fusion(self.fusion_s4, s4_curr, s4_prev, flow) if self.fuse_s4 else s4_curr
            s5_out = self._apply_flow_fusion(self.fusion_s5, s5_curr, s5_prev, flow) if self.fuse_s5 else s5_curr

            encoder_input = [s3_out, s4_out, s5_out]
        else:
            encoder_input = [s3_curr, s4_curr, s5_curr]

        x_enc = self.encoder(encoder_input)
        x_out = self.decoder(x_enc, targets)

        return x_out

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


__all__ = ['FlowRTDETR', 'RTDETR']
