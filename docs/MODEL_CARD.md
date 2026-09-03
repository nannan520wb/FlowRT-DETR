# FlowRT-DETR model card

## Model summary

FlowRT-DETR is a two-frame traffic-surveillance detector derived from RT-DETR.
A frozen SEA-RAFT network estimates motion between an earlier frame and the
current frame. Scale-aware gated residual fusion is applied to the S3 backbone
feature before the RT-DETR hybrid encoder and decoder.

## Intended use

- Research on video object detection and temporal detection stability.
- Offline evaluation on fixed traffic-surveillance cameras.
- Ablations involving optical flow, feature fusion, BBFR, and downstream queue
  stability.

The model is not validated for safety-critical traffic control, autonomous
driving, law enforcement, biometric identification, or decisions about
individuals.

## Training data

The manuscript reports experiments on the official UA-DETRAC video-level
split: 60 training sequences (approximately 84,000 frames) and 40 test
sequences (approximately 56,000 frames), with car, bus, van, and others classes.
The dataset is not distributed in this repository.

## Metrics

The manuscript reports COCO-style mAP, fixed-threshold precision/recall/F1,
BBFR-D, BBFR-T, parameter count, and FPS. See the root README for the reported
table and `REPRODUCIBILITY.md` for its verification status.

## Limitations

- Accuracy can degrade under extreme occlusion, abrupt motion, night/weather
  shifts, camera motion, or optical-flow failure.
- SEA-RAFT materially increases latency and memory compared with single-frame
  RT-DETR.
- Frame pairing and flow direction are part of the model definition; changing
  them invalidates direct checkpoint comparisons.
- Performance and bias outside UA-DETRAC have not been established.
- BBFR depends on score, matching, maximum-gap, and minimum-track settings.

## Responsible evaluation

Report per-class performance, failure cases, temporal metrics, hardware and
latency protocol. Validate on the deployment camera distribution, protect
privacy in recorded footage, and retain human oversight for consequential
uses.
