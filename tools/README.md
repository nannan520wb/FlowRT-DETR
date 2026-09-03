# Tools

Run commands from the repository root.

| Tool | Purpose |
|:--|:--|
| `train.py` | Train, resume, tune, or evaluate a detector |
| `convert_uadetrac_to_coco.py` | Convert UA-DETRAC XML to video-aware COCO JSON |
| `infer_pair.py` | Run FlowRT-DETR on an earlier/current image pair |
| `bbfr/eval_bbfr.py` | Evaluate BBFR-D and GT-assisted BBFR |
| `evaluate_pr.py` | Fixed-threshold precision/recall evaluation |
| `benchmark.py` | Benchmark detector, SEA-RAFT, ASG, and end-to-end latency |
| `visualize_flow_pair.py` | Visualize flow direction, magnitude, and overlay |
| `visualize_asg_pair.py` | Inspect ASG gates and fused features |
| `find_deteriorated_uadetrac_cases.py` | Mine blur, occlusion, and unusual cases |
| `queue_estimation/eval_queue.py` | Evaluate downstream queue stability |

Examples are documented in the root [README](../README.md). Use `-h` on an
individual tool for its complete parameter list. Outputs belong under
`output/`, which is excluded from Git.
