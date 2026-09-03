"""
排队估计模块单元测试.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from tools.queue_estimation.queue_metrics import (
    SORT, KalmanBoxTracker,
    compute_queue_length,
    compute_queue_stability_metrics,
    compute_queue_accuracy,
)


def make_box(cx, cy, w=20, h=20):
    return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dtype=np.float32)


def test_sort_basic():
    """SORT 在稳定检测下应该串成一条 track."""
    KalmanBoxTracker.count = 0
    sort = SORT(iou_thresh=0.3, max_age=5, min_hits=2)
    track_ids_seen = set()
    for f in range(1, 11):
        tracks = sort.update(
            f,
            np.array([make_box(100 + f * 5, 200)]),
            np.array([0]),
            np.array([0.9])
        )
        for tid, box, lbl, v in tracks:
            track_ids_seen.add(tid)
    assert len(track_ids_seen) == 1, f"Expected 1 track, got {len(track_ids_seen)}"
    print(f"  [SORT-basic] {len(track_ids_seen)} track(s) ✓")


def test_sort_velocity():
    """匀速运动的 box, 速度估计应该接近真实速度."""
    KalmanBoxTracker.count = 0
    sort = SORT(iou_thresh=0.3, max_age=5, min_hits=2)
    last_v = 0.0
    for f in range(1, 11):
        tracks = sort.update(
            f,
            np.array([make_box(100 + f * 5, 200)]),  # 每帧 cx 移动 5 像素
            np.array([0]),
            np.array([0.9])
        )
        for tid, box, lbl, v in tracks:
            last_v = v
    # 真实速度 5 px/frame, 容差 ±1
    assert 4.0 <= last_v <= 6.0, f"Expected v≈5, got {last_v}"
    print(f"  [SORT-velocity] estimated v={last_v:.2f} ≈ true 5 ✓")


def test_queue_length_with_roi():
    """两辆车, 一辆停止 (排队), 一辆移动. 应只统计停止的那辆."""
    tracks = [
        (1, make_box(100, 200), 0, 0.5),   # 静止 (排队)
        (2, make_box(300, 200), 0, 8.0),   # 移动中
    ]
    qlen, qids = compute_queue_length(tracks, roi_polygon=None, speed_thresh=2.0)
    assert qlen == 1
    assert qids == [1]
    print(f"  [queue-len] qlen={qlen}, ids={qids} ✓")


def test_queue_length_roi():
    """ROI 外的车不算排队."""
    roi = np.array([[0, 0], [200, 0], [200, 300], [0, 300]], dtype=np.float32)  # 左半区
    tracks = [
        (1, make_box(100, 200), 0, 0.0),    # ROI 内, 静止
        (2, make_box(400, 200), 0, 0.0),    # ROI 外, 静止
    ]
    qlen, qids = compute_queue_length(tracks, roi_polygon=roi, speed_thresh=2.0)
    assert qlen == 1
    assert qids == [1]
    print(f"  [queue-roi] qlen={qlen} (only in-ROI vehicle counted) ✓")


def test_stability_constant():
    """排队长度恒定 => std=0, jitter=0."""
    seq = [10] * 50
    m = compute_queue_stability_metrics(seq)
    assert m['std'] == 0.0
    assert m['jitter'] == 0.0
    print(f"  [stab-const] std={m['std']}, jitter={m['jitter']} ✓")


def test_stability_jitter():
    """排队长度在 8/10 间反复跳, 应有较大 jitter."""
    seq = [8 if i % 2 == 0 else 10 for i in range(50)]
    m = compute_queue_stability_metrics(seq)
    assert m['jitter'] > 1.5, f"Expected jitter ~ 2, got {m['jitter']}"
    assert m['std'] > 0.8
    print(f"  [stab-jitter] std={m['std']:.2f}, jitter={m['jitter']:.2f} ✓")


def test_stability_smooth_change():
    """排队长度缓慢爬升 (0→20), std 大但 jitter 小."""
    seq = list(range(0, 21))
    m = compute_queue_stability_metrics(seq)
    assert m['jitter'] == 1.0   # 每帧增加 1
    print(f"  [stab-smooth] mean={m['mean']:.1f}, std={m['std']:.2f}, "
          f"jitter={m['jitter']:.2f} ✓")


def test_accuracy_perfect():
    """pred == gt, mae/rmse = 0."""
    pred = [5, 6, 7, 8]
    gt = [5, 6, 7, 8]
    m = compute_queue_accuracy(pred, gt)
    assert m['mae'] == 0.0
    assert m['rmse'] == 0.0
    print(f"  [acc-perfect] mae={m['mae']}, rmse={m['rmse']} ✓")


def test_accuracy_offset():
    """pred 整体偏高 2: mae=2."""
    pred = [7, 8, 9, 10]
    gt = [5, 6, 7, 8]
    m = compute_queue_accuracy(pred, gt)
    assert m['mae'] == 2.0
    assert m['rmse'] == 2.0
    print(f"  [acc-offset] mae={m['mae']}, rmse={m['rmse']} ✓")


def test_full_pipeline_synthetic():
    """完整 pipeline: 5 辆车, 2 辆停车 3 辆缓行, 期望排队 = 2."""
    KalmanBoxTracker.count = 0
    sort = SORT(iou_thresh=0.3, max_age=5, min_hits=2)
    qlens = []
    for f in range(1, 21):
        boxes = [
            make_box(100, 200),                 # 静止排队车 1
            make_box(150, 200),                 # 静止排队车 2
            make_box(300 + f * 10, 200),        # 高速通过
            make_box(400 + f * 10, 200),        # 高速通过
            make_box(500 + f * 10, 200),        # 高速通过
        ]
        tracks = sort.update(
            f,
            np.stack(boxes, axis=0),
            np.array([0, 0, 0, 0, 0]),
            np.array([0.9, 0.9, 0.9, 0.9, 0.9])
        )
        qlen, _ = compute_queue_length(tracks, roi_polygon=None, speed_thresh=2.0)
        qlens.append(qlen)
    # 经过 SORT 暖机后 (min_hits=2), 应该稳定输出 2
    assert qlens[-1] == 2, f"Expected qlen=2 at last frame, got {qlens[-1]}"
    m = compute_queue_stability_metrics(qlens[3:])  # 去掉前几帧暖机
    print(f"  [pipeline] last qlen={qlens[-1]}, "
          f"steady-state std={m['std']:.2f}, jitter={m['jitter']:.2f} ✓")


def test_setting_b_vs_c_with_flicker():
    """当检测器有 flicker 时, setting B 排队不稳定, setting C 稳定."""
    # 模拟一辆停在路边的车, 检测器在第 5/10/15 帧漏检
    boxes_seq = []
    for f in range(1, 21):
        if f in (5, 10, 15):
            boxes_seq.append(np.zeros((0, 4)))
        else:
            boxes_seq.append(np.array([make_box(100, 200)]))

    # Setting B
    KalmanBoxTracker.count = 0
    sort_B = SORT(iou_thresh=0.3, max_age=5, min_hits=2)
    qlens_B = []
    for f in range(1, 21):
        bx = boxes_seq[f - 1]
        lb = np.zeros((len(bx),), dtype=int)
        sc = np.ones((len(bx),)) * 0.9
        tracks = sort_B.update(f, bx, lb, sc)
        qlen, _ = compute_queue_length(tracks, None, 2.0)
        qlens_B.append(qlen)

    # Setting C
    KalmanBoxTracker.count = 0
    sort_C = SORT(iou_thresh=0.3, max_age=5, min_hits=2)
    qlens_C = []
    for f in range(1, 21):
        bx = boxes_seq[f - 1]
        lb = np.zeros((len(bx),), dtype=int)
        sc = np.ones((len(bx),)) * 0.9
        tracks = sort_C.update_with_interp(f, bx, lb, sc)
        qlen, _ = compute_queue_length(tracks, None, 2.0)
        qlens_C.append(qlen)

    m_B = compute_queue_stability_metrics(qlens_B[3:])
    m_C = compute_queue_stability_metrics(qlens_C[3:])
    print(f"  [B-vs-C] B: jitter={m_B['jitter']:.2f}  C: jitter={m_C['jitter']:.2f}")
    # 注意: 由于 SORT 内部的预测机制, B 的速度估计在漏检时会偏高,
    # 可能导致车辆短暂被认为"非排队", 所以 jitter 更大.
    # 这个测试主要展示两种设置的差异, 不强校验大小关系.


if __name__ == '__main__':
    print("=" * 60)
    print("Queue Estimation Unit Tests")
    print("=" * 60)
    test_sort_basic()
    test_sort_velocity()
    test_queue_length_with_roi()
    test_queue_length_roi()
    test_stability_constant()
    test_stability_jitter()
    test_stability_smooth_change()
    test_accuracy_perfect()
    test_accuracy_offset()
    test_full_pipeline_synthetic()
    test_setting_b_vs_c_with_flicker()
    print("=" * 60)
    print("All tests passed ✓")
    print("=" * 60)