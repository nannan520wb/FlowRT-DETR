"""
BBFR (Bounding Box Flicker Rate) — 时序稳定性指标
适用于 FlowRT-DETR 评估 (T-ITS submission)

定义 (论文口径):
    给定一个视频序列, 一个 "flicker" 事件指: 一个真实存在的目标轨迹中,
    在中间某些帧出现了短暂的检测中断 (漏检), 但前后帧均检测到了同一目标.
    BBFR 衡量这种"闪烁"的频率, 反映检测器的时序稳定性.

两种实现:
    1. BBFR-Det (主指标, 无需 GT): 基于检测结果做 IoU tracking,
       统计每条 track 中的 fragmentation (短暂消失再出现) 次数.
    2. BBFR-GT (辅助验证, 需要 GT): 基于 GT 轨迹, 统计"GT 中存在但
       检测器漏检"的孤立帧数 (前后帧都检中, 当前帧漏掉).

归一化 (使指标可跨视频比较):
    BBFR = (总 flicker 事件数) / (总 track-frames) * 1000
    单位: flickers per 1000 detection-frames

Author: for FlowRT-DETR / T-ITS
"""

import os
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


# =====================================================================
# 1. 基础几何工具
# =====================================================================

def box_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """单对 box 的 IoU. box: [x1, y1, x2, y2]"""
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter <= 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def iou_matrix_xyxy(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """批量 IoU. 返回 [N_a, N_b] 矩阵."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = boxes_a[:, None, :]   # [N_a, 1, 4]
    b = boxes_b[None, :, :]   # [1, N_b, 4]
    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter + 1e-9
    return (inter / union).astype(np.float32)


# =====================================================================
# 2. 轻量 IoU Tracker (用于 BBFR-Det)
# =====================================================================

class SimpleIoUTracker:
    """
    最简 IoU tracker, 用于将连续帧的检测串成轨迹.
    关键参数:
        iou_thresh: 帧间匹配 IoU 阈值
        max_lost  : 一条 track 在丢失多少帧后判定为终止 (此值决定了 flicker 的最大间隔)
    设计要点:
        - 同类别 (label) 才允许匹配, 避免 car/bus 跳变.
        - 用 Hungarian-style 贪心匹配 (按 IoU 降序), 简单稳定.
        - 终止后的 track 不再复活 (避免假阳性 flicker).
    """

    def __init__(self, iou_thresh: float = 0.5, max_lost: int = 5,
                 same_class_only: bool = True):
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost
        self.same_class_only = same_class_only
        self.next_track_id = 0
        self.active_tracks: Dict[int, dict] = {}    # track_id -> dict
        self.finished_tracks: List[dict] = []       # 已结束的轨迹

    def _create_track(self, frame_idx: int, box: np.ndarray, label: int, score: float):
        tid = self.next_track_id
        self.next_track_id += 1
        self.active_tracks[tid] = {
            'track_id': tid,
            'label': int(label),
            'frames': [frame_idx],          # 检中的帧序号 (按时间顺序)
            'boxes': [box.astype(np.float32)],
            'scores': [float(score)],
            'last_seen': frame_idx,         # 最后一次检中的帧
            'last_box': box.astype(np.float32),
            'gaps': [],                     # 记录该 track 中所有的 gap 长度 (>0 即 flicker)
            'first_frame': frame_idx,
        }
        return tid

    def update(self, frame_idx: int,
               boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray):
        """
        输入当前帧的检测, 更新 tracker 状态.
        boxes : [N,4] xyxy
        labels: [N]
        scores: [N]
        """
        # ---- Step 1: 终止超过 max_lost 没更新的 track ----
        to_remove = []
        for tid, tr in self.active_tracks.items():
            if frame_idx - tr['last_seen'] > self.max_lost:
                self.finished_tracks.append(tr)
                to_remove.append(tid)
        for tid in to_remove:
            del self.active_tracks[tid]

        if len(boxes) == 0:
            return

        # ---- Step 2: 与活跃 track 做 IoU 匹配 ----
        active_ids = list(self.active_tracks.keys())
        if len(active_ids) == 0:
            for i in range(len(boxes)):
                self._create_track(frame_idx, boxes[i], labels[i], scores[i])
            return

        active_boxes = np.stack([self.active_tracks[tid]['last_box'] for tid in active_ids], axis=0)
        active_labels = np.array([self.active_tracks[tid]['label'] for tid in active_ids])
        ious = iou_matrix_xyxy(boxes, active_boxes)        # [N_det, N_track]

        # 同类约束
        if self.same_class_only:
            same_cls = (labels[:, None] == active_labels[None, :])
            ious = ious * same_cls.astype(np.float32)

        # 贪心匹配
        matched_det = set()
        matched_trk = set()
        # 把所有候选对按 IoU 降序排
        cand = []
        for i in range(ious.shape[0]):
            for j in range(ious.shape[1]):
                if ious[i, j] >= self.iou_thresh:
                    cand.append((ious[i, j], i, j))
        cand.sort(key=lambda x: -x[0])

        for iou, i, j in cand:
            if i in matched_det or j in matched_trk:
                continue
            # 匹配成功: 更新对应 track
            tid = active_ids[j]
            tr = self.active_tracks[tid]
            gap = frame_idx - tr['last_seen']     # 中间空了几帧
            if gap > 1:
                # 出现了 flicker, 记录间隔长度
                tr['gaps'].append(gap - 1)
            tr['frames'].append(frame_idx)
            tr['boxes'].append(boxes[i].astype(np.float32))
            tr['scores'].append(float(scores[i]))
            tr['last_seen'] = frame_idx
            tr['last_box'] = boxes[i].astype(np.float32)
            matched_det.add(i)
            matched_trk.add(j)

        # 未匹配的检测 → 新建 track
        for i in range(len(boxes)):
            if i not in matched_det:
                self._create_track(frame_idx, boxes[i], labels[i], scores[i])

    def finalize(self):
        """把所有未关闭的 track 推入 finished_tracks."""
        for tid, tr in list(self.active_tracks.items()):
            self.finished_tracks.append(tr)
        self.active_tracks.clear()
        return self.finished_tracks


# =====================================================================
# 3. BBFR-Det: 基于检测结果的 BBFR (主指标)
# =====================================================================

def compute_bbfr_det(
    video_detections: Dict[int, dict],
    score_thresh: float = 0.3,
    iou_thresh: float = 0.5,
    max_lost: int = 5,
    min_track_len: int = 3,
    same_class_only: bool = True,
) -> dict:
    """
    输入单个视频序列的检测结果, 计算 BBFR-Det.

    Args:
        video_detections: {frame_idx (int) -> {'boxes':[N,4] xyxy,
                                              'labels':[N],
                                              'scores':[N]}}
            注意: frame_idx 必须能反映时间顺序 (从小到大对应 t1, t2, ...)
        score_thresh    : 低于此分数的检测忽略 (默认 0.3, 与论文常用阈值一致)
        iou_thresh      : tracker 帧间匹配 IoU 阈值
        max_lost        : 一条 track 允许的最大丢失帧数 (即 flicker 最大跨度)
        min_track_len   : 太短的轨迹 (<3 帧) 不计入统计, 避免噪声主导
        same_class_only : tracker 是否要求同类匹配 (建议 True)

    Returns:
        dict: {
            'BBFR'                  : float, 主指标 (per 1000 track-frames),
            'flicker_events'        : int  , 总 flicker 事件数,
            'flicker_frames'        : int  , 因 flicker 丢失的总帧数 (sum of gaps),
            'total_track_frames'    : int  , 所有有效 track 的总检中帧数,
            'num_tracks'            : int  , 有效 track 数,
            'mean_track_length'     : float,
            'fragmentation_per_track': float, 平均每条 track 的 flicker 次数,
        }
    """
    if len(video_detections) == 0:
        return {'BBFR': 0.0, 'flicker_events': 0, 'flicker_frames': 0,
                'total_track_frames': 0, 'num_tracks': 0,
                'mean_track_length': 0.0, 'fragmentation_per_track': 0.0}

    tracker = SimpleIoUTracker(iou_thresh=iou_thresh,
                               max_lost=max_lost,
                               same_class_only=same_class_only)

    # 严格按帧序遍历
    sorted_frames = sorted(video_detections.keys())
    for fidx in sorted_frames:
        det = video_detections[fidx]
        boxes  = np.asarray(det['boxes'],  dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(det['labels'], dtype=np.int64).reshape(-1)
        scores = np.asarray(det['scores'], dtype=np.float32).reshape(-1)

        keep = scores >= score_thresh
        tracker.update(fidx, boxes[keep], labels[keep], scores[keep])

    tracks = tracker.finalize()

    # 过滤太短的 track
    valid_tracks = [t for t in tracks if len(t['frames']) >= min_track_len]

    flicker_events = sum(len(t['gaps']) for t in valid_tracks)
    flicker_frames = sum(sum(t['gaps']) for t in valid_tracks)
    total_track_frames = sum(len(t['frames']) for t in valid_tracks)
    num_tracks = len(valid_tracks)

    if total_track_frames == 0:
        bbfr = 0.0
        frag_per_track = 0.0
        mean_len = 0.0
    else:
        bbfr = flicker_events / total_track_frames * 1000.0
        frag_per_track = flicker_events / max(num_tracks, 1)
        mean_len = total_track_frames / max(num_tracks, 1)

    return {
        'BBFR'                   : float(bbfr),
        'flicker_events'         : int(flicker_events),
        'flicker_frames'         : int(flicker_frames),
        'total_track_frames'     : int(total_track_frames),
        'num_tracks'             : int(num_tracks),
        'mean_track_length'      : float(mean_len),
        'fragmentation_per_track': float(frag_per_track),
    }


# =====================================================================
# 4. BBFR-GT: 基于 GT 的辅助验证指标
# =====================================================================

def compute_bbfr_gt(
    video_detections: Dict[int, dict],
    video_gts:        Dict[int, dict],
    score_thresh: float = 0.3,
    iou_match:    float = 0.5,
) -> dict:
    """
    基于 GT 轨迹的 BBFR (需要数据集提供 track_id, 例如 UA-DETRAC).

    定义:
        对每条 GT 轨迹, 在它存在的帧序列上, 检测器是否每帧都能匹配到一个
        IoU >= iou_match 的预测. 若某帧匹配不到, 但其前一帧和后一帧均能
        匹配到, 则记一次 GT-flicker.

    Args:
        video_detections: 同 compute_bbfr_det
        video_gts: {frame_idx -> {'boxes':[M,4] xyxy,
                                  'labels':[M],
                                  'track_ids':[M]}}  # track_ids 是 GT 提供的物体 ID
    Returns:
        dict: BBFR-GT 及相关统计.
    """
    # 重组成 per-track 的 GT 序列
    track_seq: Dict[int, Dict[int, dict]] = defaultdict(dict)
    for fidx, gt in video_gts.items():
        boxes = np.asarray(gt['boxes'], dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(gt['labels'], dtype=np.int64).reshape(-1)
        tids = np.asarray(gt['track_ids'], dtype=np.int64).reshape(-1)
        for k in range(len(boxes)):
            track_seq[int(tids[k])][fidx] = {
                'box': boxes[k], 'label': int(labels[k])
            }

    total_gt_frames = 0
    flicker_events = 0
    flicker_frames = 0
    num_gt_tracks = 0

    for tid, frames in track_seq.items():
        sorted_f = sorted(frames.keys())
        if len(sorted_f) < 3:
            continue
        num_gt_tracks += 1
        # 每一帧, 检查检测器是否覆盖到该 GT
        hits: Dict[int, bool] = {}
        for fidx in sorted_f:
            gt_box = frames[fidx]['box']
            gt_lbl = frames[fidx]['label']
            det = video_detections.get(fidx, None)
            if det is None:
                hits[fidx] = False
                continue
            d_boxes  = np.asarray(det['boxes'],  dtype=np.float32).reshape(-1, 4)
            d_labels = np.asarray(det['labels'], dtype=np.int64).reshape(-1)
            d_scores = np.asarray(det['scores'], dtype=np.float32).reshape(-1)
            keep = d_scores >= score_thresh
            d_boxes, d_labels = d_boxes[keep], d_labels[keep]
            same = d_labels == gt_lbl
            if same.sum() == 0:
                hits[fidx] = False
                continue
            ious = iou_matrix_xyxy(gt_box[None, :], d_boxes[same]).reshape(-1)
            hits[fidx] = bool(ious.max() >= iou_match) if len(ious) > 0 else False
            total_gt_frames += 1

        # 在该 GT 轨迹上扫描 flicker: 形如 [True, False*k, True], k>=1
        keys = sorted_f
        i = 0
        while i < len(keys):
            if hits[keys[i]]:
                # 找下一个 True
                j = i + 1
                while j < len(keys) and not hits[keys[j]]:
                    j += 1
                if j < len(keys) and (j - i) >= 2:
                    # 中间至少有 1 帧 False, 且两端都是 True → flicker
                    miss = (j - i) - 1
                    flicker_events += 1
                    flicker_frames += miss
                    i = j
                else:
                    i += 1
            else:
                i += 1

    if total_gt_frames == 0:
        bbfr_gt = 0.0
    else:
        bbfr_gt = flicker_events / total_gt_frames * 1000.0

    return {
        'BBFR_GT'          : float(bbfr_gt),
        'gt_flicker_events': int(flicker_events),
        'gt_flicker_frames': int(flicker_frames),
        'total_gt_frames'  : int(total_gt_frames),
        'num_gt_tracks'    : int(num_gt_tracks),
    }


# =====================================================================
# 5. 多视频聚合
# =====================================================================

def aggregate_bbfr(per_video_results: Dict[str, dict], key: str = 'BBFR') -> dict:
    """
    多视频聚合. 用 micro-average (按 track-frame 加权) 而不是 macro-average
    更符合 T-ITS / MOT 领域惯例 (避免短视频主导结果).

    Returns:
        dict: 包含 micro 和 macro 两种平均, 以及 per-video 列表.
    """
    if len(per_video_results) == 0:
        return {'BBFR_micro': 0.0, 'BBFR_macro': 0.0, 'per_video': {}}

    if key == 'BBFR':
        flick_total = sum(r['flicker_events']     for r in per_video_results.values())
        frame_total = sum(r['total_track_frames'] for r in per_video_results.values())
        macro = float(np.mean([r['BBFR'] for r in per_video_results.values()]))
        micro = (flick_total / frame_total * 1000.0) if frame_total > 0 else 0.0
        return {
            'BBFR_micro': micro,
            'BBFR_macro': macro,
            'total_flicker_events': int(flick_total),
            'total_track_frames'  : int(frame_total),
            'num_videos'          : len(per_video_results),
            'per_video'           : per_video_results,
        }
    elif key == 'BBFR_GT':
        flick_total = sum(r['gt_flicker_events'] for r in per_video_results.values())
        frame_total = sum(r['total_gt_frames']   for r in per_video_results.values())
        macro = float(np.mean([r['BBFR_GT'] for r in per_video_results.values()]))
        micro = (flick_total / frame_total * 1000.0) if frame_total > 0 else 0.0
        return {
            'BBFR_GT_micro'        : micro,
            'BBFR_GT_macro'        : macro,
            'total_gt_flicker_events': int(flick_total),
            'total_gt_frames'      : int(frame_total),
            'num_videos'           : len(per_video_results),
            'per_video'            : per_video_results,
        }
    else:
        raise ValueError(f"Unknown key: {key}")