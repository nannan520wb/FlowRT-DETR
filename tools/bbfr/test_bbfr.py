"""
BBFR 单元测试: 在合成场景验证指标行为.
测试逻辑参考论文级别的 sanity check, 可作为论文 supplementary 的实验.

预期行为:
    场景 A (完美检测): BBFR = 0
    场景 B (单个 flicker): BBFR > 0, flicker_events = 1
    场景 C (持续抖动): BBFR 远大于 B
    场景 D (开头/结尾的丢失不算 flicker): BBFR = 0
    场景 E (类别错误不应视为同一 track): 不会形成跨类匹配
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from tools.bbfr.bbfr_metric import compute_bbfr_det, compute_bbfr_gt


def make_box(cx, cy, w=20, h=20):
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]


def test_perfect_detection():
    """场景 A: 一辆车从 frame 1 到 10 都被稳定检中, 应该 0 flicker."""
    dets = {}
    for f in range(1, 11):
        dets[f] = {
            'boxes':  np.array([make_box(100 + f*5, 200)]),
            'labels': np.array([0]),
            'scores': np.array([0.9]),
        }
    r = compute_bbfr_det(dets, score_thresh=0.3, iou_thresh=0.3, max_lost=5)
    assert r['flicker_events'] == 0, f"Expected 0 flickers, got {r['flicker_events']}"
    assert r['BBFR'] == 0.0
    assert r['num_tracks'] == 1
    print(f"  [A] perfect detection: BBFR={r['BBFR']:.3f}, "
          f"tracks={r['num_tracks']} ✓")


def test_single_flicker():
    """场景 B: 第 5 帧漏检, 前后帧都检中 → 1 次 flicker."""
    dets = {}
    for f in range(1, 11):
        if f == 5:
            dets[f] = {'boxes': np.zeros((0, 4)),
                       'labels': np.zeros((0,), dtype=int),
                       'scores': np.zeros((0,))}
        else:
            dets[f] = {
                'boxes':  np.array([make_box(100 + f*5, 200)]),
                'labels': np.array([0]),
                'scores': np.array([0.9]),
            }
    r = compute_bbfr_det(dets, score_thresh=0.3, iou_thresh=0.3, max_lost=5)
    assert r['flicker_events'] == 1, f"Expected 1 flicker, got {r['flicker_events']}"
    assert r['flicker_frames'] == 1
    print(f"  [B] single flicker: BBFR={r['BBFR']:.3f}, "
          f"events={r['flicker_events']} ✓")


def test_severe_jitter():
    """场景 C: 隔帧漏检 (1, 3, 5, 7 检中, 2, 4, 6 漏检) — 多次 flicker."""
    dets = {}
    for f in range(1, 11):
        if f % 2 == 0:
            dets[f] = {'boxes': np.zeros((0, 4)),
                       'labels': np.zeros((0,), dtype=int),
                       'scores': np.zeros((0,))}
        else:
            dets[f] = {
                'boxes':  np.array([make_box(100 + f*5, 200)]),
                'labels': np.array([0]),
                'scores': np.array([0.9]),
            }
    r = compute_bbfr_det(dets, score_thresh=0.3, iou_thresh=0.3, max_lost=5)
    # 1→3, 3→5, 5→7, 7→9, 共 4 次 flicker
    assert r['flicker_events'] == 4, f"Expected 4 flickers, got {r['flicker_events']}"
    print(f"  [C] severe jitter: BBFR={r['BBFR']:.3f}, "
          f"events={r['flicker_events']} ✓")


def test_no_flicker_at_boundary():
    """场景 D: 物体在第 8 帧之后离开视野 — 不应算 flicker."""
    dets = {}
    for f in range(1, 11):
        if f <= 7:
            dets[f] = {
                'boxes':  np.array([make_box(100 + f*5, 200)]),
                'labels': np.array([0]),
                'scores': np.array([0.9]),
            }
        else:
            dets[f] = {'boxes': np.zeros((0, 4)),
                       'labels': np.zeros((0,), dtype=int),
                       'scores': np.zeros((0,))}
    r = compute_bbfr_det(dets, score_thresh=0.3, iou_thresh=0.3, max_lost=5)
    assert r['flicker_events'] == 0, f"Boundary loss should not count, got {r['flicker_events']}"
    print(f"  [D] boundary loss not flicker: BBFR={r['BBFR']:.3f} ✓")


def test_class_change_not_same_track():
    """场景 E: label 跳变不应被串成一个 track."""
    dets = {}
    for f in range(1, 11):
        # 前 5 帧 class 0, 后 5 帧 class 1, 位置完全相同
        cls = 0 if f <= 5 else 1
        dets[f] = {
            'boxes':  np.array([make_box(100, 200)]),
            'labels': np.array([cls]),
            'scores': np.array([0.9]),
        }
    r = compute_bbfr_det(dets, score_thresh=0.3, iou_thresh=0.3, max_lost=5,
                         same_class_only=True)
    # 应形成 2 条独立的 track, 各自 5 帧, 都没 flicker
    assert r['num_tracks'] == 2, f"Expected 2 tracks, got {r['num_tracks']}"
    assert r['flicker_events'] == 0
    print(f"  [E] class change → separate tracks: tracks={r['num_tracks']} ✓")


def test_bbfr_gt_basic():
    """测试 BBFR-GT: GT 存在 10 帧, 检测器在第 5 帧漏掉."""
    gt_box = make_box(100, 200)
    gts = {}
    dets = {}
    for f in range(1, 11):
        gts[f] = {
            'boxes':    np.array([gt_box]),
            'labels':   np.array([0]),
            'track_ids': np.array([1]),
        }
        if f == 5:
            dets[f] = {'boxes': np.zeros((0, 4)),
                       'labels': np.zeros((0,), dtype=int),
                       'scores': np.zeros((0,))}
        else:
            dets[f] = {
                'boxes':  np.array([gt_box]),
                'labels': np.array([0]),
                'scores': np.array([0.9]),
            }
    r = compute_bbfr_gt(dets, gts, score_thresh=0.3, iou_match=0.5)
    assert r['gt_flicker_events'] == 1
    assert r['gt_flicker_frames'] == 1
    print(f"  [GT-1] BBFR_GT={r['BBFR_GT']:.3f}, "
          f"events={r['gt_flicker_events']} ✓")


def test_low_score_filtered():
    """场景: 第 5 帧检测分数 < threshold, 等价于漏检."""
    dets = {}
    for f in range(1, 11):
        score = 0.1 if f == 5 else 0.9
        dets[f] = {
            'boxes':  np.array([make_box(100 + f*5, 200)]),
            'labels': np.array([0]),
            'scores': np.array([score]),
        }
    r = compute_bbfr_det(dets, score_thresh=0.3, iou_thresh=0.3, max_lost=5)
    assert r['flicker_events'] == 1
    print(f"  [F] low score filtered → flicker: events={r['flicker_events']} ✓")


def test_multi_object():
    """场景: 两辆车并行, 其中一辆在第 5 帧漏检."""
    dets = {}
    for f in range(1, 11):
        if f == 5:
            # 只检中第二辆
            dets[f] = {
                'boxes':  np.array([make_box(300, 200)]),
                'labels': np.array([0]),
                'scores': np.array([0.9]),
            }
        else:
            dets[f] = {
                'boxes': np.array([make_box(100, 200), make_box(300, 200)]),
                'labels': np.array([0, 0]),
                'scores': np.array([0.9, 0.9]),
            }
    r = compute_bbfr_det(dets, score_thresh=0.3, iou_thresh=0.3, max_lost=5)
    assert r['num_tracks'] == 2, f"Expected 2 tracks, got {r['num_tracks']}"
    assert r['flicker_events'] == 1
    print(f"  [G] multi-object, one flickers: tracks={r['num_tracks']}, "
          f"events={r['flicker_events']} ✓")


if __name__ == '__main__':
    print("=" * 60)
    print("BBFR Unit Tests")
    print("=" * 60)
    test_perfect_detection()
    test_single_flicker()
    test_severe_jitter()
    test_no_flicker_at_boundary()
    test_class_change_not_same_track()
    test_low_score_filtered()
    test_multi_object()
    test_bbfr_gt_basic()
    print("=" * 60)
    print("All tests passed ✓")
    print("=" * 60)