"""简化版官方测试 - 处理单张图片对比"""
import argparse
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys
import cv2

# 添加UWNR路径
sys.path.insert(0, '.')
from model.FSU2 import Generator
from myutils import dcp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='输入图片路径')
    parser.add_argument('--depth', required=True, help='深度图路径(.png)')
    parser.add_argument('--output', required=True, help='输出图片路径')
    parser.add_argument('--model', default='./UWNR.pk', help='模型权重路径')
    parser.add_argument('--size', type=int, default=256, help='处理尺寸')
    parser.add_argument('--gpu', default='0', help='GPU ID')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}')
    
    # 加载模型
    print(f'Loading model from {args.model}...')
    netG = Generator()
    ckpt = torch.load(args.model, map_location='cpu')
    state = ckpt['G1'] if 'G1' in ckpt else ckpt
    from collections import OrderedDict
    new_state = OrderedDict()
    for k, v in state.items():
        new_state[k.replace('module.', '', 1)] = v
    netG.load_state_dict(new_state)
    netG.to(device)
    netG.eval()
    
    # 1. 加载输入图像 (与官方一致: PIL打开)
    print(f'Processing {args.input}...')
    img_pil = Image.open(args.input).convert("RGB")
    h_orig, w_orig = img_pil.size[1], img_pil.size[0]
    
    # 2. Resize (与官方一致)
    img_pil_resized = transforms.Resize([args.size, args.size])(img_pil)
    
    # 3. 转Tensor (与官方一致: transforms.ToTensor())
    img_tensor = transforms.ToTensor()(img_pil_resized).unsqueeze(0)
    
    # 4. 加载深度图 (与官方一致: PIL打开L模式)
    depth_pil = Image.open(args.depth).convert("L")
    depth_pil = transforms.Resize([args.size, args.size])(depth_pil)
    depth_tensor = transforms.ToTensor()(depth_pil).unsqueeze(0)
    
    # 5. 计算A_map (与官方一致)
    A_map = dcp.MutiScaleLuminanceEstimation(np.uint8(np.array(img_pil_resized)))
    A_map_tensor = transforms.ToTensor()(np.float32(A_map)) / 255.0
    
    # 6. 拼接 (与官方一致: [img, depth, A_map])
    x = torch.cat([img_tensor, depth_tensor, A_map_tensor.unsqueeze(0)], dim=1).to(device)
    
    print(f'Input shape: {x.shape}')
    print(f'  Image: {img_tensor.shape}')
    print(f'  Depth: {depth_tensor.shape}')
    print(f'  A_map: {A_map_tensor.unsqueeze(0).shape}')
    
    # 7. 推理
    with torch.no_grad():
        output = netG(x)
    
    print(f'Output range: [{output.min():.3f}, {output.max():.3f}]')
    
    # 8. 后处理 (与官方save_image一致: 假设输出是[-1,1])
    output = output.squeeze(0)
    output = (output + 1.0) / 2.0  # [-1,1] -> [0,1]
    output = torch.clamp(output, 0, 1)
    
    # 9. 转回PIL并resize到原尺寸
    output_pil = transforms.ToPILImage()(output)
    output_pil = output_pil.resize((w_orig, h_orig), Image.BILINEAR)
    
    # 10. 保存
    output_pil.save(args.output)
    print(f'Saved to {args.output}')
    
    # 同时保存resize后的原图用于对比
    img_pil_resized.save(args.output.replace('.png', '_input.png'))
    print(f'Saved input (resized) to {args.output.replace(".png", "_input.png")}')

if __name__ == '__main__':
    main()