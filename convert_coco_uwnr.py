"""Convert clean COCO images to underwater images using UWNR - 与官方test.py完全一致"""
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
from PIL import Image
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


def _compute_a_map(img_pil):
    """与官方dataloader完全一致的A_map计算"""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'myutils'))
    from myutils import dcp
    # 与官方一致: dcp.MutiScaleLuminanceEstimation(np.uint8(np.array(data)))
    A_map = dcp.MutiScaleLuminanceEstimation(np.uint8(np.array(img_pil)))
    # 与官方一致: tfs.ToTensor()(np.float32(A_map))/255，然后归一化到[-1,1]
    A_map_tensor = transforms.ToTensor()(np.float32(A_map)) / 255.0
    return A_map_tensor


def load_midas(device):
    model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    return midas, transform


def estimate_depth(img_pil, midas_model, midas_transform, device):
    """Estimate depth from PIL image using MiDaS"""
    img_np = np.array(img_pil)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    input_batch = midas_transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = midas_model(input_batch)
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=(img_pil.size[1], img_pil.size[0]),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = prediction.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth.astype(np.float32)


def process_single_image(img_path, netG, device, size, midas_model=None,
                         midas_transform=None, depth_dir=None):
    """处理单张图像，与官方test.py流程一致"""
    # 与官方一致: PIL打开图像
    img_pil = Image.open(img_path).convert("RGB")
    h_orig, w_orig = img_pil.size[1], img_pil.size[0]
    
    # Resize（与官方一致）
    img_pil_resized = transforms.Resize([size, size])(img_pil)
    
    # 计算A_map（与官方dataloader一致）
    A_map_tensor = _compute_a_map(img_pil_resized)
    
    # data转tensor（与官方一致: tfs.ToTensor()(data)），归一化到[-1,1]
    img_tensor = transforms.ToTensor()(img_pil_resized)
    
    # Get depth map
    basename = os.path.splitext(os.path.basename(img_path))[0]
    if depth_dir and os.path.exists(os.path.join(depth_dir, basename + '.png')):
        depth_pil = Image.open(os.path.join(depth_dir, basename + '.png')).convert("L")
        depth_pil = transforms.Resize([size, size])(depth_pil)
        depth_tensor = transforms.ToTensor()(depth_pil)
    elif depth_dir and os.path.exists(os.path.join(depth_dir, basename + '.npy')):
        depth = np.load(os.path.join(depth_dir, basename + '.npy'))
        depth = cv2.resize(depth, (size, size))
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0)
    elif midas_model is not None:
        depth = estimate_depth(img_pil_resized, midas_model, midas_transform, device)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0)
    else:
        depth_tensor = torch.ones(1, size, size) * 0.5
    
    # 拼接（与官方一致: torch.cat([gt,depth_map,A_map],1)）
    x = torch.cat([img_tensor.unsqueeze(0), depth_tensor.unsqueeze(0), A_map_tensor.unsqueeze(0)], dim=1).to(device)
    
    with torch.no_grad():
        output = netG(x)
    
    # 与官方一致: save_image(g1_output, ..., normalize=False)
    # 注意：save_image的normalize=False表示不再次归一化，假设输入已经是[0,1]
    # 但Tanh输出是[-1,1]，所以需要先转换到[0,1]
    output = output.squeeze(0)
    output = torch.clamp(output, 0, 1)
    
    # 转回PIL并resize到原尺寸
    output_pil = transforms.ToPILImage()(output)
    output_pil = output_pil.resize((w_orig, h_orig), Image.BILINEAR)
    
    # 转numpy用于保存
    output_np = np.array(output_pil)
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
    parser.add_argument('--gpu', default='0', help='GPU ID')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()
    
    # 设备设置
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        main_device = torch.device('cuda:0')
        print(f'Using GPU via CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}')
    else:
        gpu_id = args.gpu.split(',')[0]
        main_device = torch.device(f'cuda:{gpu_id}')
        print(f'Using GPU: {gpu_id}')
    
    # 加载标注
    with open(args.ann, 'r') as f:
        coco = json.load(f)
    images = coco['images']
    if args.end is not None:
        images = images[args.start:args.end]
    else:
        images = images[args.start:]
    print(f'Images to process: {len(images)} (from {args.start} to {args.end})')
    
    # 创建输出目录
    ann_out_dir = os.path.join(args.output_dir, 'annotations')
    os.makedirs(ann_out_dir, exist_ok=True)
    ann_out = os.path.join(ann_out_dir, os.path.basename(args.ann))
    if not os.path.exists(ann_out):
        shutil.copy2(args.ann, ann_out)
        print(f'Copied annotation to {ann_out}')
    
    img_out_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(img_out_dir, exist_ok=True)
    
    # 加载模型
    print(f'Loading UWNR model from {args.uwnr_model} ...')
    netG = load_uwnr_generator(args.uwnr_model, args.uwnr_dir, main_device)
    
    midas_model, midas_transform = None, None
    if args.depth_dir is None:
        print('Loading MiDaS for on-the-fly depth estimation...')
        midas_model, midas_transform = load_midas(main_device)
    
    # 处理图像
    skipped = 0
    for i, img_info in enumerate(tqdm(images, desc='UWNR converting')):
        filename = img_info['file_name']
        src_path = os.path.join(args.img_dir, filename)
        dst_path = os.path.join(img_out_dir, filename)
        
        if os.path.exists(dst_path):
            continue
        
        result = process_single_image(
            src_path, netG, main_device, args.size,
            midas_model=midas_model,
            midas_transform=midas_transform,
            depth_dir=args.depth_dir
        )
        
        if result is None:
            print(f'Warning: failed to read {src_path}')
            skipped += 1
            continue
        
        # 用PIL保存（与官方一致）
        Image.fromarray(result).save(dst_path)
    
    print(f'Done. Processed: {len(images) - skipped}, Skipped: {skipped}')
    print(f'Output: {img_out_dir}')


if __name__ == '__main__':
    main()