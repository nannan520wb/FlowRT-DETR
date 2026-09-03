# Bounding-Box Flicker Rate

BBFR measures short detection interruptions inside otherwise continuous object
tracks. It complements frame-wise AP by focusing on temporal instability.

For a set of tracks, the implementation reports flicker events per 1,000 track
frames:

```text
BBFR = number_of_flicker_events / number_of_track_frames × 1000
```

Two modes are provided:

- **BBFR-D:** builds tracks from detections with class-aware IoU association;
  no ground-truth identity is required.
- **BBFR-T / GT-assisted:** aligns detections to annotated tracks and measures
  short internal misses.

The manuscript defaults are score threshold `0.3`, IoU threshold `0.5`, maximum
gap `5`, and minimum track length `3`. Always report these parameters because
they define the metric operating point.

```bash
python tools/bbfr/eval_bbfr.py \
  -c configs/flowrtdetr/flowrtdetr_r18_uadetrac.yml \
  -r path/to/checkpoint.pth \
  --score-thresh 0.3 --iou-thresh 0.5 \
  --max-lost 5 --min-track-len 3 \
  --out-json output/bbfr.json
```

Synthetic unit tests cover stable detections, isolated flickers, repeated
flickers, boundary disappearance, class changes, and multiple objects:

```bash
pytest -q tools/bbfr/test_bbfr.py
```
