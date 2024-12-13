# -*-coding:utf-8 -*-

"""
# File       : train.py
# Time       ：2024/12/12 15:57
# Author     ：Qiang42
# version    ：python 3.8
# Description：
"""

from tqdm import tqdm
import torch.nn as nn
from collections import OrderedDict
from Loss.Loss import g_loss_wgan, d_loss_wgan_gp

def train(args, G, D_IR, D_VI, train_loader, criterion, optimizer, optimizer_D_IR, optimizer_D_VI, device):
    G.train()
    D_IR.train()
    D_VI.train()
    running_loss = 0
    train_tqdm = tqdm(train_loader, total=len(train_loader), desc="Train", leave=False, delay=1)
    for i, (_, vis_y_image, _, _, ir_image, _, _) in enumerate(train_tqdm):
        ir_image, vis_y_image = ir_image.to(device), vis_y_image.to(device)
        optimizer.zero_grad()

        fused_image, W_IR, W_VI, F_IR, F_VI, ir_w, vi_w = G(ir_image, vis_y_image)
        # D_IR
        for _ in range(1):
            optimizer_D_IR.zero_grad()
            d_loss_IR = d_loss_wgan_gp(D_IR, ir_image, fused_image.detach(), device, lambda_gp=10)
            d_loss_IR.backward()
            optimizer_D_IR.step()

        # D_VI
        for _ in range(1):
            optimizer_D_VI.zero_grad()
            d_loss_VI = d_loss_wgan_gp(D_VI, vis_y_image, fused_image.detach(), device, lambda_gp=10)
            d_loss_VI.backward()
            optimizer_D_VI.step()

        loss_D = d_loss_IR + d_loss_VI

        content_loss = criterion(fused_image, ir_image, vis_y_image)

        loss_W_IR = nn.L1Loss()(W_IR, F_IR)
        loss_W_VI = nn.L1Loss()(W_VI, F_VI)
        loss_W_IR_VI = nn.L1Loss()(W_IR, 1 - W_VI)
        loss_W = loss_W_IR + loss_W_VI + loss_W_IR_VI

        fake_output_IR = D_IR(fused_image)
        fake_output_VI = D_VI(fused_image)
        g_loss_ir = g_loss_wgan(fake_output_IR)
        g_loss_vi = g_loss_wgan(fake_output_VI)
        g_loss = g_loss_ir + g_loss_vi
        alpha = 0.1
        loss_G = 10 * content_loss + loss_W + alpha * g_loss
        totle_loss = loss_G + alpha * loss_D
        running_loss += totle_loss.item()
        train_tqdm.set_postfix(OrderedDict([
            ('loss_D', f"{loss_D.item():.6f}"),
            ('content_loss', f"{content_loss.item():.6f}"),
            ('loss_W', f"{loss_W.item():.6f}"),
            ('g_loss', f"{g_loss.item():.6f}"),
        ]))
        loss_G.backward()
        optimizer.step()

    return running_loss / len(train_loader)
