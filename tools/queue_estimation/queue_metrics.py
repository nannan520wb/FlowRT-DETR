"""
排队估计核心模块
================
包含:
    1. 简易 SORT tracker (用于 RT-DETR + SORT baseline 对比)
    2. 速度估计 (基于 tracker 输出)
    3. 排队长度计算
    4. 稳定性指标 (Std, Jitter)

设计要点:
    - SORT 实现遵循 Bewley et al. 2016 原作思路, 但简化为纯 numpy 实现, 避免依赖 filterpy
    - 速度用相邻帧 box 中心位移估计, 单位是 pixel/frame (后续可换算为 km/h 如果有相机标定)
    - 排队 = 视野内 + 速度 < 阈值, 不强制要求空间连续 (UA-DETRAC 大部分场景排队会自然连续)



queue_metrics.py — 核心算法库（无 PyTorch 依赖）

KalmanBoxTracker + SORT：完整的 Kalman + IoU 跟踪器
compute_queue_length：基于 ROI + 速度阈值算排队车辆数
compute_queue_stability_metrics：std / jitter / peak_jitter / smoothness
compute_queue_accuracy：mae / rmse / mape vs GT
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional


# =====================================================================
# 1. Kalman Filter (用于 SORT)
# =====================================================================

class KalmanBoxTracker:
    """
    单个 box 的 Kalman 滤波器, 状态向量 [cx, cy, s, r, vx, vy, vs]:
        cx, cy: 中心
        s     : box 面积
        r     : 长宽比 (近似不变)
        vx,vy : 中心速度
        vs    : 面积变化率
    遵循 Bewley 2016 SORT 论文设置.
    """
    count = 0

    def __init__(self, bbox_xyxy: np.ndarray, label: int):
        # 状态转移矩阵 F: 线性匀速运动模型
        self.F = np.eye(7, dtype=np.float32)
        for i in range(4):
            if i < 3:
                self.F[i, i + 4] = 1.0
        # 观测矩阵 H: 只观测 [cx, cy, s, r]
        self.H = np.zeros((4, 7), dtype=np.float32)
        self.H[:4, :4] = np.eye(4)
        # 过程噪声 / 观测噪声 (经验值, 与原版 SORT 一致)
        self.Q = np.eye(7, dtype=np.float32)
        self.Q[4:, 4:] *= 0.01
        self.Q[6, 6] *= 0.01
        self.R = np.eye(4, dtype=np.float32)
        self.R[2:, 2:] *= 10.0
        # 协方差初值
        self.P = np.eye(7, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 1000.0

        # 状态初值
        self.x = np.zeros((7, 1), dtype=np.float32)
        self.x[:4, 0] = self._xyxy_to_z(bbox_xyxy)

        # 元数据
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.label = int(label)
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        # 历史 (用于速度估计)
        self.history = deque(maxlen=10)         # [(frame_idx, cx, cy)]

    @staticmethod
    def _xyxy_to_z(b: np.ndarray) -> np.ndarray:
        w = b[2] - b[0]
        h = b[3] - b[1]
        cx = b[0] + w / 2
        cy = b[1] + h / 2
        s = w * h
        r = w / max(h, 1e-6)
        return np.array([cx, cy, s, r], dtype=np.float32)

    @staticmethod
    def _z_to_xyxy(z: np.ndarray) -> np.ndarray:
        cx, cy, s, r = z[:4]
        s = max(s, 1.0)
        r = max(r, 1e-6)
        w = np.sqrt(s * r)
        h = s / max(w, 1e-6)
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dtype=np.float32)

    def predict(self):
        # x_{k|k-1} = F x_{k-1|k-1}
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.age += 1
        return self._z_to_xyxy(self.x[:, 0])

    def update(self, bbox_xyxy: np.ndarray, frame_idx: int):
        z = self._xyxy_to_z(bbox_xyxy).reshape(4, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7, dtype=np.float32) - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        cx, cy = float(self.x[0, 0]), float(self.x[1, 0])
        self.history.append((frame_idx, cx, cy))

    def get_state_xyxy(self) -> np.ndarray:
        return self._z_to_xyxy(self.x[:, 0])

    def get_velocity(self, window: int = 5) -> float:
        """估计 pixel/frame 速度. 用最近 window 个观测的位移平均."""
        if len(self.history) < 2:
            return 0.0
        recent = list(self.history)[-window:]
        if len(recent) < 2:
            return 0.0
        f0, x0, y0 = recent[0]
        f1, x1, y1 = recent[-1]
        dt = max(f1 - f0, 1)
        return float(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / dt)


# =====================================================================
# 2. 简易 SORT
# =====================================================================

def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[N,4] x [M,4] -> [N,M] IoU"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    aa = a[:, None]
    bb = b[None, :]
    x1 = np.maximum(aa[..., 0], bb[..., 0])
    y1 = np.maximum(aa[..., 1], bb[..., 1])
    x2 = np.minimum(aa[..., 2], bb[..., 2])
    y2 = np.minimum(aa[..., 3], bb[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (aa[..., 2] - aa[..., 0]) * (aa[..., 3] - aa[..., 1])
    area_b = (bb[..., 2] - bb[..., 0]) * (bb[..., 3] - bb[..., 1])
    return (inter / (area_a + area_b - inter + 1e-9)).astype(np.float32)


class SORT:
    """
    简易 SORT tracker. 接口设计:
        update(frame_idx, dets_xyxy, dets_labels, dets_scores)
            -> list of (track_id, xyxy, label, velocity)
    """
    def __init__(self, iou_thresh: float = 0.3, max_age: int = 5, min_hits: int = 3,
                 same_class_only: bool = True):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.same_class_only = same_class_only
        self.trackers: List[KalmanBoxTracker] = []

    def update(self, frame_idx: int,
               dets_xyxy: np.ndarray, dets_labels: np.ndarray, dets_scores: np.ndarray):
        # Step 1: 所有 tracker 预测一步
        predicted = []
        for tr in self.trackers:
            pred = tr.predict()
            predicted.append(pred)
        predicted = np.array(predicted, dtype=np.float32) if len(predicted) > 0 \
                    else np.zeros((0, 4), dtype=np.float32)
        trk_labels = np.array([tr.label for tr in self.trackers], dtype=np.int64)

        # Step 2: IoU 匹配
        N, M = len(dets_xyxy), len(predicted)
        matched_d, matched_t = set(), set()
        if N > 0 and M > 0:
            iou_mat = _iou_xyxy(dets_xyxy, predicted)
            if self.same_class_only:
                same_cls = (dets_labels[:, None] == trk_labels[None, :]).astype(np.float32)
                iou_mat = iou_mat * same_cls
            # 贪心匹配
            cand = []
            for i in range(N):
                for j in range(M):
                    if iou_mat[i, j] >= self.iou_thresh:
                        cand.append((iou_mat[i, j], i, j))
            cand.sort(key=lambda x: -x[0])
            for _, i, j in cand:
                if i in matched_d or j in matched_t:
                    continue
                self.trackers[j].update(dets_xyxy[i], frame_idx)
                matched_d.add(i)
                matched_t.add(j)

        # Step 3: 未匹配的检测 -> 新 tracker
        for i in range(N):
            if i not in matched_d:
                self.trackers.append(KalmanBoxTracker(dets_xyxy[i], int(dets_labels[i])))

        # Step 4: 移除过老的 tracker
        self.trackers = [tr for tr in self.trackers if tr.time_since_update <= self.max_age]

        # Step 5: 输出当前活跃 tracker
        out = []
        for tr in self.trackers:
            # 只输出 hit_streak >= min_hits 或 帧 < min_hits 的 (避免开头丢失)
            if tr.hit_streak >= self.min_hits or frame_idx < self.min_hits:
                if tr.time_since_update == 0:
                    # 仅当前帧匹配上的才输出 (避免 SORT 强行插值带来的假阳性)
                    # 注: 如果想包含 SORT 插值结果以体现"消除 flicker"效果, 改成 <= max_age
                    out.append((tr.id, tr.get_state_xyxy(), tr.label, tr.get_velocity()))
        return out

    def update_with_interp(self, frame_idx: int,
                           dets_xyxy: np.ndarray, dets_labels: np.ndarray, dets_scores: np.ndarray):
        """
        和 update 的区别: 即使当前帧没匹配上, 只要 tracker 还活着 (time_since_update <= max_age),
        就用 Kalman 预测的位置作为输出. 这是 RT-DETR + SORT baseline 在 BBFR 上"作弊变低"的关键.
        """
        # 主流程同 update
        predicted = []
        for tr in self.trackers:
            pred = tr.predict()
            predicted.append(pred)
        predicted = np.array(predicted, dtype=np.float32) if len(predicted) > 0 \
                    else np.zeros((0, 4), dtype=np.float32)
        trk_labels = np.array([tr.label for tr in self.trackers], dtype=np.int64)

        N, M = len(dets_xyxy), len(predicted)
        matched_d, matched_t = set(), set()
        if N > 0 and M > 0:
            iou_mat = _iou_xyxy(dets_xyxy, predicted)
            if self.same_class_only:
                same_cls = (dets_labels[:, None] == trk_labels[None, :]).astype(np.float32)
                iou_mat = iou_mat * same_cls
            cand = []
            for i in range(N):
                for j in range(M):
                    if iou_mat[i, j] >= self.iou_thresh:
                        cand.append((iou_mat[i, j], i, j))
            cand.sort(key=lambda x: -x[0])
            for _, i, j in cand:
                if i in matched_d or j in matched_t:
                    continue
                self.trackers[j].update(dets_xyxy[i], frame_idx)
                matched_d.add(i)
                matched_t.add(j)

        for i in range(N):
            if i not in matched_d:
                self.trackers.append(KalmanBoxTracker(dets_xyxy[i], int(dets_labels[i])))

        self.trackers = [tr for tr in self.trackers if tr.time_since_update <= self.max_age]

        out = []
        for tr in self.trackers:
            if tr.hit_streak >= self.min_hits or tr.hits >= self.min_hits:
                # 关键差异: 不要求 time_since_update == 0
                # SORT 用 Kalman 预测的 box 顶替漏检, 看起来轨迹连续了
                out.append((tr.id, tr.get_state_xyxy(), tr.label, tr.get_velocity()))
        return out


# =====================================================================
# 3. 排队估计核心
# =====================================================================

def compute_queue_length(
    tracks: List[Tuple[int, np.ndarray, int, float]],
    roi_polygon: Optional[np.ndarray] = None,
    speed_thresh: float = 2.0,
) -> Tuple[int, List[int]]:
    """
    计算当前帧的排队车辆数.

    Args:
        tracks: [(track_id, xyxy, label, velocity), ...]
        roi_polygon: ROI 多边形 [N, 2] (x, y), 在像素坐标系下. None 则全图都算.
        speed_thresh: 速度阈值 (pixel/frame), 低于此值视为"排队"

    Returns:
        queue_count: 排队车辆数
        queue_track_ids: 这些车的 track id
    """
    queue_ids = []
    for tid, xyxy, lbl, v in tracks:
        # 用 box 底边中点判定是否在 ROI (车辆在地面的位置)
        cx = (xyxy[0] + xyxy[2]) / 2
        cy_bot = xyxy[3]
        if roi_polygon is not None:
            if not _point_in_polygon(cx, cy_bot, roi_polygon):
                continue
        # 速度判定
        if v <= speed_thresh:
            queue_ids.append(tid)
    return len(queue_ids), queue_ids


def _point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    """Ray casting 算法判断点是否在多边形内. polygon: [N, 2]"""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


# =====================================================================
# 4. 稳定性指标
# =====================================================================

def compute_queue_stability_metrics(queue_lengths: List[int]) -> dict:
    """
    输入一个视频的逐帧排队长度序列, 计算稳定性指标.

    指标:
        std         : 标准差, 整体波动
        jitter      : 平均相邻帧绝对差, 高频抖动
        peak_jitter : 最大相邻帧差
        smoothness  : 二阶差分的均方 (加速度), 越小越平滑
    """
    if len(queue_lengths) < 2:
        return {'mean': 0.0, 'std': 0.0, 'jitter': 0.0,
                'peak_jitter': 0.0, 'smoothness': 0.0, 'n_frames': len(queue_lengths)}
    arr = np.array(queue_lengths, dtype=np.float32)
    diff1 = np.abs(np.diff(arr))
    diff2 = np.diff(arr, n=2) if len(arr) >= 3 else np.array([0.0])
    return {
        'mean'       : float(arr.mean()),
        'std'        : float(arr.std()),
        'jitter'     : float(diff1.mean()),
        'peak_jitter': float(diff1.max()) if len(diff1) > 0 else 0.0,
        'smoothness' : float((diff2 ** 2).mean()) if len(diff2) > 0 else 0.0,
        'n_frames'   : int(len(queue_lengths)),
    }


def compute_queue_accuracy(pred_lengths: List[int], gt_lengths: List[int]) -> dict:
    """
    Pred vs GT 的精度指标. 长度需一一对应 (按帧).
    """
    n = min(len(pred_lengths), len(gt_lengths))
    if n == 0:
        return {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0, 'n_frames': 0}
    p = np.array(pred_lengths[:n], dtype=np.float32)
    g = np.array(gt_lengths[:n], dtype=np.float32)
    err = p - g
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    # MAPE: 仅对 g > 0 的帧计算, 避免除零
    mask = g > 0
    if mask.sum() > 0:
        mape = float(np.abs(err[mask] / g[mask]).mean() * 100)
    else:
        mape = 0.0
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'n_frames': n}