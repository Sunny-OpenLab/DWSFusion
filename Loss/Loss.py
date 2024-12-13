# -*-coding:utf-8 -*-

"""
# File       : Loss.py
# Time       ：2024/12/12 15:59
# Author     ：Qiang42
# version    ：python 3.8
# Description：
"""

import torch
import torch.nn as nn
import utils
from pytorch_msssim import ms_ssim
import torch.autograd as autograd

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
    def forward(self, fused, ir, vi):
        loss_fuse_int = nn.L1Loss()(fused, torch.max(ir, vi))

        # Texture loss
        fused_grad = torch.abs(utils.gradient(fused))
        ir_grad = torch.abs(utils.gradient(ir))
        vi_gard = torch.abs(utils.gradient(vi))
        loss_fuse_grad = nn.L1Loss()(fused_grad, torch.max(ir_grad, vi_gard))

        # MS-SSIM loss
        msssim_loss = 1 - ms_ssim(fused, torch.max(ir, vi), data_range=1.0)

        # Final loss
        loss_fuse =  loss_fuse_int + loss_fuse_grad + msssim_loss
        return loss_fuse

def g_loss_wgan(fake_output):
    return -torch.mean(fake_output)

def d_loss_wgan_gp(D, real_samples, fake_samples, device, lambda_gp=10):
    real_loss = -torch.mean(D(real_samples))
    fake_loss = torch.mean(D(fake_samples))
    if lambda_gp != 0:
        gp = compute_gradient_penalty(D, real_samples, fake_samples, device)
        d_loss = real_loss + fake_loss + lambda_gp * gp
    else:
        d_loss = real_loss + fake_loss
    return d_loss

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    # 生成随机插值样本
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    # 判别器对插值样本的输出
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size(), requires_grad=False, device=device)

    # 计算插值样本的梯度
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # 计算梯度惩罚项
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty