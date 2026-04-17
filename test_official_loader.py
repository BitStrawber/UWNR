"""使用官方DataLoader方式测试UWNR - 确保与训练时完全一致"""
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys
import os

sys.path.insert(0, '.')
from model.FSU2 import Generator
from myutils.dataloader import UWS_Dataset_Retinex_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='输入图片路径')
    parser.add_argument('--output', required=True, help='输出图片路径')
    parser.add_argument('--model', default='./UWNR.pk', help='模型权重路径')
    parser.add_argument('--gpu', default='0', help='GPU ID')
    args = parser.parse_args()
    
    # 创建设备
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
    
    # 复制图片到临时目录结构（模拟官方DataLoader期望的结构）
    import tempfile
    import shutil
    
    tmp_dir = tempfile.mkdtemp()
    qingxi_dir = os.path.join(tmp_dir, 'qingxi')
    os.makedirs(qingxi_dir)
    
    # 复制图片
    tmp_img_path = os.path.join(qingxi_dir, 'test.jpg')
    shutil.copy(args.input, tmp_img_path)
    
    print(f'Created temp structure: {tmp_dir}')
    
    # 使用官方DataLoader
    dataset = UWS_Dataset_Retinex_test(tmp_dir, train=False, size=256, dcp=False)
    
    print(f'Dataset length: {len(dataset)}')
    
    # 获取数据
    A_map, data = dataset[0]
    
    print(f'A_map shape: {A_map.shape}, range: [{A_map.min():.3f}, {A_map.max():.3f}]')
    print(f'data shape: {data.shape}, range: [{data.min():.3f}, {data.max():.3f}]')
    
    # DataLoader输出是A_map, data（图像）
    # 但没有depth，我们需要自己生成或使用默认
    
    # 创建默认depth（与官方一致）
    depth_tensor = torch.ones(1, 256, 256) * 0  # [-1,1]范围，0对应0.5
    
    # 添加batch维度并移到GPU
    A_map = A_map.unsqueeze(0).to(device)
    data = data.unsqueeze(0).to(device)
    depth_tensor = depth_tensor.unsqueeze(0).to(device)
    
    print(f'Input shapes - A_map: {A_map.shape}, data: {data.shape}, depth: {depth_tensor.shape}')
    
    # 拼接（注意顺序：[data, depth, A_map]）
    x = torch.cat([data, depth_tensor, A_map], dim=1)
    print(f'Concatenated input shape: {x.shape}')
    
    # 推理
    with torch.no_grad():
        output = netG(x)
    
    print(f'Output range: [{output.min():.3f}, {output.max():.3f}]')
    
    # 后处理
    output = output.squeeze(0)
    output = (output + 1.0) / 2.0  # [-1,1] -> [0,1]
    output = torch.clamp(output, 0, 1)
    
    # 转PIL并保存
    output_pil = transforms.ToPILImage()(output)
    output_pil.save(args.output)
    print(f'Saved to {args.output}')
    
    # 清理临时目录
    shutil.rmtree(tmp_dir)
    print('Cleaned up temp directory')

if __name__ == '__main__':
    main()