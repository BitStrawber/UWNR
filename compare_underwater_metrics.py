"""对比修复前后生成图像的水下特征指标

水下图像典型特征:
  - 蓝绿通道主导 (B>G>R)
  - 暗通道值偏高 (雾化/散射)
  - 对比度偏低
  - 饱和度偏低
  - UCIQE相关指标

用法:
    python compare_underwater_metrics.py --old /path/to/old_output --new /path/to/new_output
    python compare_underwater_metrics.py --old /path/to/old_img.png --new /path/to/new_img.png
    python compare_underwater_metrics.py --old /path/to/old_dir --new /path/to/new_dir --num 100
"""
import argparse
import os
import sys
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def calc_dark_channel(img, patch=15):
    h, w = img.shape[:2]
    min_rgb = img.min(axis=2)
    padded = np.pad(min_rgb, ((patch // 2, patch // 2), (patch // 2, patch // 2)), mode='edge')
    dc = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            dc[i, j] = padded[i:i + patch, j:j + patch].min()
    return dc


def calc_metrics(img_bgr):
    img = img_bgr.astype(np.float64) / 255.0
    B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    r_mean = R.mean()
    g_mean = G.mean()
    b_mean = B.mean()

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    contrast = gray.std()

    dark_ch = calc_dark_channel(img_bgr).mean() / 255.0

    max_rgb = img.max(axis=2)
    min_rgb = img.min(axis=2)
    saturation = np.where(max_rgb > 1e-6, (max_rgb - min_rgb) / max_rgb, 0)
    sat_mean = saturation.mean()

    hue_shift = b_mean - r_mean

    return {
        'R_mean': r_mean,
        'G_mean': g_mean,
        'B_mean': b_mean,
        'B-R': b_mean - r_mean,
        'B-G': b_mean - g_mean,
        'contrast': contrast,
        'dark_channel': dark_ch,
        'saturation': sat_mean,
    }


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return img


def collect_images(path, num=None):
    p = Path(path)
    if p.is_file():
        img = load_image(str(p))
        return [img] if img is not None else []
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    files = sorted([f for f in p.iterdir() if f.suffix.lower() in exts])
    if num:
        files = files[:num]
    images = []
    for f in files:
        img = load_image(str(f))
        if img is not None:
            images.append(img)
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old', required=True, help='修复前的输出(目录或单图)')
    parser.add_argument('--new', required=True, help='修复后的输出(目录或单图)')
    parser.add_argument('--num', type=int, default=None, help='最多采样图片数')
    args = parser.parse_args()

    old_imgs = collect_images(args.old, args.num)
    new_imgs = collect_images(args.new, args.num)

    if not old_imgs or not new_imgs:
        print('错误: 无法加载图片')
        sys.exit(1)

    n = min(len(old_imgs), len(new_imgs))
    print(f'采样图片数: {n}')

    old_all = {k: [] for k in calc_metrics(old_imgs[0])}
    new_all = {k: [] for k in calc_metrics(new_imgs[0])}

    for i in tqdm(range(n), desc='计算指标中'):
        om = calc_metrics(old_imgs[i])
        nm = calc_metrics(new_imgs[i])
        for k in om:
            old_all[k].append(om[k])
            new_all[k].append(nm[k])

    labels = {
        'R_mean':    'R通道均值 (水下应偏低)',
        'G_mean':    'G通道均值',
        'B_mean':    'B通道均值 (水下应偏高)',
        'B-R':       'B-R差值 (水下应为正,越大越明显)',
        'B-G':       'B-G差值',
        'contrast':  '对比度 (水下应偏低)',
        'dark_channel': '暗通道均值 (水下雾化应偏高)',
        'saturation':   '饱和度 (水下通常偏低)',
    }

    print(f'\n{"指标":<30} {"修复前(旧)":>12} {"修复后(新)":>12} {"变化":>12}  判断')
    print('-' * 85)

    for k in labels:
        o = np.mean(old_all[k])
        n_val = np.mean(new_all[k])
        diff = n_val - o
        sign = '+' if diff > 0 else ''

        if k in ('B_mean', 'B-R', 'dark_channel'):
            judge = '✓ 更水下' if diff > 0 else '✗ 不明显'
        elif k in ('R_mean', 'contrast', 'saturation'):
            judge = '✓ 更水下' if diff < 0 else '✗ 不明显'
        else:
            judge = ''

        print(f'{labels[k]:<30} {o:>12.4f} {n_val:>12.4f} {sign}{diff:>11.4f}  {judge}')

    b_r_old = np.mean(old_all['B-R'])
    b_r_new = np.mean(new_all['B-R'])

    print('\n' + '=' * 85)
    if b_r_new > 0 and b_r_new > b_r_old:
        print('结论: 修复后图像具备水下特征 (B>R, 蓝色主导), 且比修复前更明显')
    elif b_r_new > 0:
        print('结论: 修复后图像具备水下特征 (B>R), 但改善不明显')
    else:
        print('结论: 修复后图像水下特征不明显 (R>=B), 可能仍有问题')


if __name__ == '__main__':
    main()
