# -*-coding:utf-8 -*-

"""
# File       : val.py
# Time       ：2024/12/12 15:57
# Author     ：Qiang42
# version    ：python 3.8
# Description：
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from Loss.Loss import g_loss_wgan, d_loss_wgan_gp
from collections import OrderedDict

def val(args, G, D_IR, D_VI, val_loader, criterion, device):
    G.eval()
    D_IR.eval()
    D_VI.eval()
    running_loss = 0.0
    with torch.no_grad():
        eval_tqdm = tqdm(val_loader, total=len(val_loader), desc="eval", leave=False)
        for i, (_, vis_y_image, _, _, ir_image, _, _) in enumerate(eval_tqdm):
            ir_image, vis_y_image = ir_image.to(device), vis_y_image.to(device)
            fused_image, W_IR, W_VI, F_IR, F_VI, ir_w, vi_w  = G(ir_image, vis_y_image)

            d_loss_IR = d_loss_wgan_gp(D_IR, ir_image, fused_image.detach(), device, lambda_gp=0)
            d_loss_VI = d_loss_wgan_gp(D_VI, vis_y_image, fused_image.detach(), device, lambda_gp=0)
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
            eval_tqdm.set_postfix(OrderedDict([
                ('loss_D', f"{loss_D.item():.6f}"),
                ('content_loss', f"{content_loss.item():.6f}"),
                ('loss_W', f"{loss_W.item():.6f}"),
                ('g_loss', f"{g_loss.item():.6f}"),
            ]))

    return running_loss / len(val_loader)

