# Reproducibility notes

This repository is a cleaned research-code snapshot, not a bit-for-bit archive
of the training machine. The source tree was separated from local datasets,
large outputs, and checkpoints; paths and interfaces were made portable while
preserving the surviving model behavior.

## Reported versus verified

- README metrics are transcribed from the manuscript and are not CI results.
- The manuscript reports an A800-SXM4-80GB speed measurement. Use
  `tools/benchmark.py` and report hardware, precision, batch size, input size,
  SEA-RAFT iterations, warmup, repeats, and whether postprocessing is included.
- Detector and optical-flow checkpoints are not included. Publish checksums
  alongside any future model release.
- The original converted UA-DETRAC JSON used for training was not present when
  this repository was prepared, so data ordering could not be independently
  audited.

## Snapshot-defining behavior

- The surviving dataset code used a two-record historical offset. This is now
  explicit as `frame_offset: 2`; the manuscript describes adjacent frames.
- SEA-RAFT is called as `(previous, current)`, producing previous-to-current
  forward flow. ASG samples the previous feature with an approximate
  `current_grid - forward_flow` lookup and masks invalid coordinates.
- The released ASG implementation uses an identity-initialized residual,
  `max_gate=0.2`, gate bias `-7`, motion/similarity masks, zero padding, and a
  learned transform of `warped_previous - current`.
- The release optimizer uses cosine annealing for 72 epochs. No warmup argument
  is claimed because the underlying scheduler factory does not implement one.

These details must be treated as experimental configuration, not incidental
implementation. If the manuscript specification is chosen as authoritative,
change them deliberately, retrain, and report a new result table rather than
mixing checkpoints across definitions.

## Minimum release record

For each published checkpoint, archive:

1. Git commit and complete YAML configuration.
2. Dataset conversion script, annotation checksum, and split manifest.
3. Detector and SEA-RAFT checkpoint checksums.
4. Python, PyTorch, torchvision, CUDA, cuDNN, and GPU versions.
5. Random seed, world size, effective batch size, and exact launch command.
6. COCO metrics, BBFR parameters/results, and benchmark protocol.

## Lightweight checks

```bash
PYTHONPYCACHEPREFIX=/tmp/flowrtdetr_pycache \
  python -m compileall -q src optical_flow tools
pytest -q tools/bbfr/test_bbfr.py tools/queue_estimation/test_queue.py
```

CI runs these checks without downloading datasets or model weights.
