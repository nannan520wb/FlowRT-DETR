"""Convert one UA-DETRAC split from XML annotations to COCO detection JSON."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image


CATEGORIES = [
    {"id": 1, "name": "car"},
    {"id": 2, "name": "bus"},
    {"id": 3, "name": "van"},
    {"id": 4, "name": "others"},
]
CATEGORY_ID = {item["name"]: item["id"] for item in CATEGORIES}


def find_frame(image_root, sequence, frame_number):
    directory = image_root / sequence
    for suffix in (".jpg", ".jpeg", ".png"):
        path = directory / f"img{frame_number:05d}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Missing frame {frame_number} for {sequence} under {directory}")


def parse_box(element, width, height):
    left = max(0.0, float(element.attrib["left"]))
    top = max(0.0, float(element.attrib["top"]))
    right = min(float(width), left + float(element.attrib["width"]))
    bottom = min(float(height), top + float(element.attrib["height"]))
    return left, top, max(0.0, right - left), max(0.0, bottom - top)


def convert(annotation_root, image_root, sequence_names=None, strict_categories=False):
    allowed = set(sequence_names) if sequence_names else None
    output = {
        "info": {"description": "UA-DETRAC converted to COCO detection format"},
        "licenses": [],
        "categories": CATEGORIES,
        "videos": [],
        "images": [],
        "annotations": [],
    }
    image_id = 1
    annotation_id = 1

    xml_files = sorted(annotation_root.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML annotations found under {annotation_root}")

    for xml_path in xml_files:
        root = ET.parse(xml_path).getroot()
        sequence = root.attrib.get("name", xml_path.stem)
        if allowed is not None and sequence not in allowed:
            continue
        video_id = len(output["videos"]) + 1
        output["videos"].append({"id": video_id, "name": sequence})

        frames = sorted(root.findall("frame"), key=lambda item: int(item.attrib["num"]))
        for frame in frames:
            frame_number = int(frame.attrib["num"])
            image_path = find_frame(image_root, sequence, frame_number)
            with Image.open(image_path) as image:
                width, height = image.size

            current_image_id = image_id
            output["images"].append({
                "id": current_image_id,
                "file_name": image_path.relative_to(image_root).as_posix(),
                "width": width,
                "height": height,
                "video_id": video_id,
                "frame_id": frame_number,
            })
            image_id += 1

            target_list = frame.find("target_list")
            if target_list is None:
                continue
            for target in target_list.findall("target"):
                box_element = target.find("box")
                attribute = target.find("attribute")
                if box_element is None or attribute is None:
                    continue
                vehicle_type = attribute.attrib.get("vehicle_type", "others").lower()
                if vehicle_type not in CATEGORY_ID:
                    if strict_categories:
                        raise ValueError(
                            f"Unknown vehicle_type={vehicle_type!r} in {xml_path}")
                    vehicle_type = "others"
                box = parse_box(box_element, width, height)
                if box[2] <= 0 or box[3] <= 0:
                    continue
                output["annotations"].append({
                    "id": annotation_id,
                    "image_id": current_image_id,
                    "category_id": CATEGORY_ID[vehicle_type],
                    "bbox": list(box),
                    "area": box[2] * box[3],
                    "iscrowd": 0,
                    "track_id": int(target.attrib["id"]),
                })
                annotation_id += 1

    if not output["images"]:
        raise RuntimeError("No matching sequences were converted")
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sequence-list",
        type=Path,
        help="Optional text file containing one sequence name per line.")
    parser.add_argument("--strict-categories", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    sequence_names = None
    if args.sequence_list:
        sequence_names = [
            line.strip() for line in args.sequence_list.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    result = convert(
        args.annotation_root,
        args.image_root,
        sequence_names=sequence_names,
        strict_categories=args.strict_categories,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"Wrote {len(result['images'])} images and "
        f"{len(result['annotations'])} annotations to {args.output}")


if __name__ == "__main__":
    main()
