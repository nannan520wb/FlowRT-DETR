#motion_blur，video_defocus，part_occlusion，unusual_vehicle_appearance
# COCO 格式 UA-DETRAC 标注里自动筛

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def gray_np(img):
    arr = np.asarray(img.convert('L'), dtype=np.float32) / 255.0
    return arr


def laplacian_var(img):
    g = gray_np(img)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = (
        -4 * g[1:-1, 1:-1]
        + g[:-2, 1:-1] + g[2:, 1:-1]
        + g[1:-1, :-2] + g[1:-1, 2:]
    )
    return float(np.var(lap))


def gradient_anisotropy(img):
    g = gray_np(img)
    gx = np.diff(g, axis=1)
    gy = np.diff(g, axis=0)
    ex = float(np.mean(np.abs(gx)))
    ey = float(np.mean(np.abs(gy)))
    return max(ex, ey) / (min(ex, ey) + 1e-6)


def frame_difference(img_a, img_b):
    a = np.asarray(img_a.convert('L').resize((160, 90)), dtype=np.float32)
    b = np.asarray(img_b.convert('L').resize((160, 90)), dtype=np.float32)
    return float(np.mean(np.abs(a - b)) / 255.0)


def crop_box(img, bbox, pad=8):
    x, y, w, h = bbox
    x1 = max(0, int(x - pad))
    y1 = max(0, int(y - pad))
    x2 = min(img.width, int(x + w + pad))
    y2 = min(img.height, int(y + h + pad))
    return img.crop((x1, y1, x2, y2))


def bbox_area_ratio(bbox, width, height):
    return float(max(0, bbox[2]) * max(0, bbox[3]) / max(width * height, 1))


def boundary_truncation_score(bbox, width, height, margin=3):
    x, y, w, h = bbox
    x2, y2 = x + w, y + h
    hits = 0
    hits += x <= margin
    hits += y <= margin
    hits += x2 >= width - margin
    hits += y2 >= height - margin
    return hits / 4.0


def bbox_iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def occlusion_score(ann, same_image_anns):
    explicit = []
    for key in ('occluded', 'occlusion', 'is_occluded'):
        if key in ann:
            try:
                explicit.append(float(ann[key]))
            except Exception:
                pass
    if explicit:
        return max(explicit)
    return max((bbox_iou(ann['bbox'], other['bbox']) for other in same_image_anns if other is not ann), default=0.0)


def open_image(img_root, file_name):
    return Image.open(Path(img_root) / file_name).convert('RGB')


def find_prev_image(images_sorted, idx):
    if idx <= 0:
        return None
    curr = images_sorted[idx]
    prev = images_sorted[idx - 1]
    curr_video = curr['file_name'].split('/')[0]
    prev_video = prev['file_name'].split('/')[0]
    if curr_video == prev_video:
        return prev
    return None


def save_case(out_dir, category, rank, image, crop, meta):
    cat_dir = out_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{rank:03d}_img{meta["image_id"]}_ann{meta.get("ann_id", "frame")}'
    image_path = cat_dir / f'{stem}_frame.jpg'
    crop_path = cat_dir / f'{stem}_crop.jpg'
    txt_path = cat_dir / f'{stem}.txt'

    frame = image.copy()
    draw = ImageDraw.Draw(frame)
    if 'bbox' in meta and meta['bbox'] is not None:
        x, y, w, h = meta['bbox']
        draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=3)
    frame.save(image_path, quality=92)
    crop.save(crop_path, quality=92)
    with open(txt_path, 'w', encoding='utf-8') as f:
        for k, v in meta.items():
            f.write(f'{k}: {v}\n')
    return crop_path


def make_sheet(paths, out_path, title, thumb=(220, 130), cols=4):
    if not paths:
        return
    rows = int(np.ceil(len(paths) / cols))
    title_h = 38
    pad = 12
    w = cols * thumb[0] + (cols + 1) * pad
    h = title_h + rows * thumb[1] + (rows + 1) * pad
    sheet = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 10), title, fill=(20, 30, 40), font=ImageFont.load_default())
    for i, path in enumerate(paths):
        img = Image.open(path).convert('RGB')
        img.thumbnail(thumb)
        x = pad + (i % cols) * (thumb[0] + pad)
        y = title_h + pad + (i // cols) * (thumb[1] + pad)
        canvas = Image.new('RGB', thumb, (245, 245, 245))
        canvas.paste(img, ((thumb[0] - img.width) // 2, (thumb[1] - img.height) // 2))
        sheet.paste(canvas, (x, y))
    sheet.save(out_path, quality=95)


def main(args):
    out_dir = Path(args.output_dir)
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = load_json(args.ann_file)
    images = {img['id']: img for img in coco['images']}
    images_sorted = sorted(coco['images'], key=lambda x: (x['file_name'].split('/')[0], x.get('frame_id', x['id'])))
    image_to_idx = {img['id']: i for i, img in enumerate(images_sorted)}

    anns_by_img = defaultdict(list)
    for ann in coco['annotations']:
        if ann.get('iscrowd', 0):
            continue
        anns_by_img[ann['image_id']].append(ann)

    motion_blur = []
    defocus = []
    occlusion = []
    unusual = []

    for img_id, img_info in images.items():
        try:
            img = open_image(args.img_root, img_info['file_name'])
        except Exception:
            continue
        anns = anns_by_img.get(img_id, [])
        if not anns:
            continue

        frame_sharp = laplacian_var(img)
        frame_aniso = gradient_anisotropy(img)
        prev_info = find_prev_image(images_sorted, image_to_idx.get(img_id, 0))
        diff = 0.0
        if prev_info is not None:
            try:
                prev_img = open_image(args.img_root, prev_info['file_name'])
                diff = frame_difference(prev_img, img)
            except Exception:
                diff = 0.0

        # Defocus: whole frame is globally soft.
        defocus.append((frame_sharp, img_id, None, img_info, None, dict(
            image_id=img_id, file_name=img_info['file_name'], frame_sharpness=frame_sharp)))

        # Motion blur: low sharpness + large frame difference or directional blur.
        motion_score = (1.0 / (frame_sharp + 1e-6)) * (1.0 + diff) * min(frame_aniso, 8.0)
        motion_blur.append((motion_score, img_id, None, img_info, None, dict(
            image_id=img_id, file_name=img_info['file_name'], frame_sharpness=frame_sharp,
            frame_difference=diff, gradient_anisotropy=frame_aniso)))

        for ann in anns:
            bbox = ann['bbox']
            if bbox[2] < args.min_box or bbox[3] < args.min_box:
                continue
            crop = crop_box(img, bbox)
            crop_sharp = laplacian_var(crop)
            occ = occlusion_score(ann, anns)
            trunc = max(boundary_truncation_score(bbox, img.width, img.height), float(ann.get('truncation', 0)))
            area = bbox_area_ratio(bbox, img.width, img.height)
            aspect = bbox[2] / max(bbox[3], 1)

            occ_score = occ + trunc
            occlusion.append((occ_score, img_id, ann, img_info, crop, dict(
                image_id=img_id, ann_id=ann['id'], file_name=img_info['file_name'],
                bbox=bbox, occlusion_score=occ, truncation_score=trunc, crop_sharpness=crop_sharp)))

            # UA-DETRAC is vehicles, so "rare pose" should be interpreted as unusual appearance:
            # tiny, extreme aspect ratio, boundary truncation, or heavily blurred crop.
            unusual_score = 0.0
            unusual_score += 2.0 if area < args.small_area_ratio else 0.0
            unusual_score += min(abs(np.log(max(aspect, 1e-6) / 2.2)), 2.0)
            unusual_score += 2.0 * trunc
            unusual_score += 1.0 / (crop_sharp + 1e-6)
            unusual.append((unusual_score, img_id, ann, img_info, crop, dict(
                image_id=img_id, ann_id=ann['id'], file_name=img_info['file_name'],
                bbox=bbox, area_ratio=area, aspect_ratio=aspect, truncation_score=trunc,
                crop_sharpness=crop_sharp)))

    saved = defaultdict(list)

    for rank, (_, img_id, ann, img_info, crop, meta) in enumerate(sorted(defocus, key=lambda x: x[0])[:args.topk], 1):
        img = open_image(args.img_root, img_info['file_name'])
        saved['video_defocus'].append(save_case(out_dir, 'video_defocus', rank, img, img.resize((240, 135)), meta))

    for rank, (_, img_id, ann, img_info, crop, meta) in enumerate(sorted(motion_blur, key=lambda x: x[0], reverse=True)[:args.topk], 1):
        img = open_image(args.img_root, img_info['file_name'])
        saved['motion_blur'].append(save_case(out_dir, 'motion_blur', rank, img, img.resize((240, 135)), meta))

    for rank, (_, img_id, ann, img_info, crop, meta) in enumerate(sorted(occlusion, key=lambda x: x[0], reverse=True)[:args.topk], 1):
        img = open_image(args.img_root, img_info['file_name'])
        saved['part_occlusion'].append(save_case(out_dir, 'part_occlusion', rank, img, crop, meta))

    for rank, (_, img_id, ann, img_info, crop, meta) in enumerate(sorted(unusual, key=lambda x: x[0], reverse=True)[:args.topk], 1):
        img = open_image(args.img_root, img_info['file_name'])
        saved['unusual_vehicle_appearance'].append(save_case(out_dir, 'unusual_vehicle_appearance', rank, img, crop, meta))

    for cat, paths in saved.items():
        make_sheet(paths, out_dir / f'{cat}_sheet.jpg', cat.replace('_', ' '), cols=args.sheet_cols)

    print(f'Saved candidate cases to: {out_dir}')
    for cat, paths in saved.items():
        print(f'{cat}: {len(paths)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-root', required=True, help='COCO-format UA-DETRAC image root.')
    parser.add_argument('--ann-file', required=True, help='COCO-format annotation json.')
    parser.add_argument('-o', '--output-dir', default='output/deteriorated_cases')
    parser.add_argument('--topk', type=int, default=24)
    parser.add_argument('--min-box', type=float, default=8)
    parser.add_argument('--small-area-ratio', type=float, default=0.002)
    parser.add_argument('--sheet-cols', type=int, default=4)
    parser.add_argument('--clean', action='store_true')
    args = parser.parse_args()
    main(args)
