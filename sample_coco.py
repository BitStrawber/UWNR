"""Randomly sample N images from COCO train annotations.

Usage:
    python tools/sample_coco.py \
        --ann /path/to/instances_train2017.json \
        --img-dir /path/to/train2017 \
        --output-dir /path/to/coco_uwnr \
        --num 50000
"""
import argparse
import json
import os
import random
import shutil
from pycocotools.coco import COCO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', required=True, help='COCO train annotation file')
    parser.add_argument('--img-dir', required=True, help='COCO train image directory')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--num', type=int, default=50000, help='Number of images to sample')
    parser.add_argument('--copy', action='store_true', help='Copy images instead of symlink')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f'Loading annotations from {args.ann} ...')
    coco = COCO(args.ann)

    all_img_ids = list(coco.imgs.keys())
    print(f'Total images: {len(all_img_ids)}')

    sampled_ids = set(random.sample(all_img_ids, min(args.num, len(all_img_ids))))
    print(f'Sampled: {len(sampled_ids)}')

    images = [img for img in coco.dataset['images'] if img['id'] in sampled_ids]
    annotations = [ann for ann in coco.dataset['annotations'] if ann['image_id'] in sampled_ids]
    categories = coco.dataset['categories']

    new_dataset = {
        'images': images,
        'annotations': annotations,
        'categories': categories,
        'info': coco.dataset.get('info', {}),
        'licenses': coco.dataset.get('licenses', []),
    }

    ann_out_dir = os.path.join(args.output_dir, 'annotations')
    os.makedirs(ann_out_dir, exist_ok=True)
    ann_out = os.path.join(ann_out_dir, f'instances_train{args.num}.json')
    with open(ann_out, 'w') as f:
        json.dump(new_dataset, f)
    print(f'Saved filtered annotation to {ann_out}')

    img_out_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(img_out_dir, exist_ok=True)

    for i, img in enumerate(images):
        src = os.path.join(args.img_dir, img['file_name'])
        dst = os.path.join(img_out_dir, img['file_name'])
        if os.path.exists(dst):
            continue
        if args.copy:
            shutil.copy2(src, dst)
        else:
            os.symlink(os.path.abspath(src), dst)
        if (i + 1) % 10000 == 0:
            print(f'  Linked {i+1}/{len(images)}')

    print(f'Done. {len(images)} images linked to {img_out_dir}')


if __name__ == '__main__':
    main()