"""Convert clean COCO images to underwater images using UWNR.

Usage:
    python tools/convert_coco_uwnr.py \
        --ann /path/to/instances_train50k.json \
        --img-dir /path/to/train2017 \
        --output-dir /path/to/coco_uwnr \
        --uwnr-dir /path/to/UWNR \
        --uwnr-model /path/to/uwnr_epoch200.pth \
        [--depth-dir /path/to/depth_maps]
"""
import argparse
import os
import sys
import json
import shutil
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm


def load_uwnr_generator(model_path, uwnr_dir, device):
    sys.path.insert(0, uwnr_dir)
    from model.FSU2 import Generator
    netG = Generator()
    ckpt = torch.load(model_path, map_location='cpu')
    state = ckpt['G1'] if 'G1' in ckpt else ckpt
    from collections import OrderedDict
    new_state = OrderedDict()
    for k, v in state.items():
        new_state[k.replace('module.', '', 1)] = v
    netG.load_state_dict(new_state)
    netG.to(device)
    netG.eval()
    return netG


def _compute_a_map(img_rgb):
    from myutils.dcp import MutiScaleLuminanceEstimation
    return MutiScaleLuminanceEstimation(img_rgb)


def load_midas(device):
    model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    return midas, transform


def estimate_depth(img_np, midas_model, midas_transform, device):
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    input_batch = midas_transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = midas_model(input_batch)
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=img_np.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = prediction.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth.astype(np.float32)


def process_single_image(img_path, netG, device, size, midas_model=None,
                         midas_transform=None, depth_dir=None):
    img = cv2.imread(img_path)
    if img is None:
        return None
    h_orig, w_orig = img.shape[:2]

    img_resized = cv2.resize(img, (size, size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    A_map = _compute_a_map(img_rgb)
    A_map_tensor = transforms.ToTensor()(np.float32(A_map) / 255.0)

    basename = os.path.splitext(os.path.basename(img_path))[0]
    if depth_dir and os.path.exists(os.path.join(depth_dir, basename + '.png')):
        depth = cv2.imread(os.path.join(depth_dir, basename + '.png'), cv2.IMREAD_GRAYSCALE)
        depth = cv2.resize(depth, (size, size)).astype(np.float32) / 255.0
    elif depth_dir and os.path.exists(os.path.join(depth_dir, basename + '.npy')):
        depth = np.load(os.path.join(depth_dir, basename + '.npy'))
        depth = cv2.resize(depth, (size, size)).astype(np.float32)
        if depth.max() > 1.0:
            depth = depth / 255.0
    elif midas_model is not None:
        depth = estimate_depth(img_resized, midas_model, midas_transform, device)
    else:
        depth = np.ones((size, size), dtype=np.float32) * 0.5

    depth_tensor = torch.from_numpy(depth).unsqueeze(0)
    img_tensor = transforms.ToTensor()(img_rgb)

    x = torch.cat([img_tensor, depth_tensor, A_map_tensor], dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = netG(x)

    output = output.squeeze(0).cpu()
    output = (output + 1.0) / 2.0
    output = torch.clamp(output, 0, 1)
    output_np = (output.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    output_np = cv2.resize(output_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    return output_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', required=True)
    parser.add_argument('--img-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--uwnr-dir', required=True)
    parser.add_argument('--uwnr-model', required=True)
    parser.add_argument('--depth-dir', default=None)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    with open(args.ann, 'r') as f:
        coco = json.load(f)
    images = coco['images']
    print(f'Images to process: {len(images)}')

    ann_out_dir = os.path.join(args.output_dir, 'annotations')
    os.makedirs(ann_out_dir, exist_ok=True)
    ann_out = os.path.join(ann_out_dir, os.path.basename(args.ann))
    if not os.path.exists(ann_out):
        shutil.copy2(args.ann, ann_out)
        print(f'Copied annotation to {ann_out}')

    img_out_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(img_out_dir, exist_ok=True)

    print(f'Loading UWNR model from {args.uwnr_model} ...')
    netG = load_uwnr_generator(args.uwnr_model, args.uwnr_dir, device)

    midas_model, midas_transform = None, None
    if args.depth_dir is None:
        print('Loading MiDaS for on-the-fly depth estimation...')
        midas_model, midas_transform = load_midas(device)

    skipped = 0
    for i, img_info in enumerate(tqdm(images, desc='UWNR converting')):
        filename = img_info['file_name']
        src_path = os.path.join(args.img_dir, filename)
        dst_path = os.path.join(img_out_dir, filename)

        if os.path.exists(dst_path):
            continue

        result = process_single_image(
            src_path, netG, device, args.size,
            midas_model=midas_model,
            midas_transform=midas_transform,
            depth_dir=args.depth_dir
        )

        if result is None:
            print(f'Warning: failed to read {src_path}')
            skipped += 1
            continue

        cv2.imwrite(dst_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    print(f'Done. Processed: {len(images) - skipped}, Skipped: {skipped}')
    print(f'Output: {img_out_dir}')


if __name__ == '__main__':
    main()