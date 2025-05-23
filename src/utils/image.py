import numpy as np
import torch
import torch.nn.functional as F


def poisson_gaussian_noise_perturb(img, i, j, sigma=3, poisson_lambda=5.0, gaussian_sigma=10.0):
    """
    仅对单个像素 (i, j) 生成高斯模糊掩码，并在 GPU 上进行高斯 + 泊松扰动
    """
    C, H, W = img.shape[1:]  # (B, C, H, W)
    device = img.device

    # 生成 2D 高斯掩码 (H, W)
    x, y = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing="xy")
    gaussian_mask = torch.exp(-((x - j) ** 2 + (y - i) ** 2) / (2 * sigma ** 2))
    gaussian_mask = gaussian_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # 计算泊松噪声（光子计数误差）
    poisson_noisy = torch.poisson((img * 0.5 + 1.0) * 255.0 * poisson_lambda) / (255.0 * poisson_lambda) * 2.0 - 1.0

    # 计算高斯噪声（传感器读出噪声）
    gaussian_noise = torch.normal(mean=0, std=gaussian_sigma / 255.0 * 2.0, size=img.shape, device=device)

    # 组合噪声
    noisy_img = torch.clip(img + poisson_noisy + gaussian_noise, -1.0, 1.0)

    # 计算扰动图像
    perturbed_img = gaussian_mask * noisy_img + (1 - gaussian_mask) * img
    return perturbed_img


def gaussian_blur_perturb(img, i, j, sigma=3.0, kernel_size=11):
    """
    仅对单个像素 (i, j) 生成高斯模糊掩码，并在 GPU 上进行高斯扰动
    """
    C, H, W = img.shape[1:]  # (B, C, H, W)
    device = img.device

    # 生成 2D 高斯掩码 (H, W)
    x, y = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing="xy")
    gaussian_mask = torch.exp(-((x - j) ** 2 + (y - i) ** 2) / (2 * sigma ** 2))

    gaussian_mask = gaussian_mask.unsqueeze(0).unsqueeze(0)

    # 计算全局模糊图像
    blurred_img = F.conv2d(img, create_gaussian_kernel(C, kernel_size, sigma).to(device),
                           padding=kernel_size // 2,
                           groups=C)

    # 计算扰动图像
    perturbed_img = gaussian_mask * blurred_img + (1 - gaussian_mask) * img
    return perturbed_img


def create_gaussian_kernel(C, kernel_size=11, sigma=3.0):
    """
    生成一个 2D 高斯卷积核 (C, 1, k, k) 以适用于 F.conv2d
    """
    x = torch.arange(kernel_size).float()
    x = x - (kernel_size - 1) / 2
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()

    kernel_2d = torch.outer(gauss, gauss).unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)
    return kernel_2d.expand(C, 1, kernel_size, kernel_size)  # (C, 1, k, k)
