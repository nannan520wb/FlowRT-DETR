# Data preparation

FlowRT-DETR expects COCO-style detection annotations augmented with reliable
video ordering.

## Directory layout

```text
data/ua_detrac/
├── annotations/
│   ├── train.json
│   └── val.json
└── images/
    ├── train/
    │   ├── MVI_20011/img00001.jpg
    │   └── ...
    └── val/
        ├── MVI_39031/img00001.jpg
        └── ...
```

## Image records

Recommended COCO `images` entry:

```json
{
  "id": 1,
  "file_name": "MVI_20011/img00001.jpg",
  "width": 960,
  "height": 540,
  "video_id": "MVI_20011",
  "frame_id": 1
}
```

`VideoCocoDetection` groups images by `video_id` when it exists. Otherwise it
uses the parent directory of `file_name`. Frames are sorted by `frame_id`, then
by the last number in the filename, and finally by image id. Do not flatten all
videos into one folder without adding `video_id`.

Convert each official split independently:

```bash
python tools/convert_uadetrac_to_coco.py \
  --annotation-root path/to/DETRAC-Train-Annotations-XML \
  --image-root data/ua_detrac/images/train \
  --output data/ua_detrac/annotations/train.json
```

Use the corresponding test/validation directories for `val.json`. The script
maps `car`, `bus`, `van`, and `others` to category ids `1`, `2`, `3`, and `4`,
and preserves target identities as `track_id`.

## Categories and tracks

The release configuration preserves `car=1`, `bus=2`, `van=3`, and `others=4` with
`remap_mscoco_category: False` and `num_classes: 5`; index 0 is unused. If your
conversion remaps labels to `0..3`, update both fields consistently and verify
postprocessing before training.

BBFR-D does not require ground-truth track ids. BBFR-T/GT-assisted evaluation
requires a stable per-object field such as `track_id` in each annotation.

## Frame offset

`frame_offset` is the number of ordered records between the current and earlier
frame. The release snapshot defaults to `2`. Set it to `1` for adjacent frames.
At the start of a sequence, the current frame is reused rather than pairing
across videos.

UA-DETRAC is not redistributed by this repository. Download and use it under
the dataset provider's terms.
