# -*-coding:utf-8 -*-

"""
# File       : __init__.py.py
# Time       ：2024/12/12 15:50
# Author     ：Qiang42
# version    ：python 3.8
# Description：
"""
import random
import numpy as np
import torch
import torch.nn as nn

def init_seeds(device, seed=0):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark = (seed != 0)
    cudnn.deterministic = (seed == 0)

def gradient(input):
    filter1 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter2 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)

    filter1.weight.data = torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.]
    ]).reshape(1, 1, 3, 3).cuda()

    filter2.weight.data = torch.tensor([
        [1., 2., 1.],
        [0., 0., 0.],
        [-1., -2., -1.]
    ]).reshape(1, 1, 3, 3).cuda()

    g1 = filter1(input)
    g2 = filter2(input)
    image_gradient = torch.abs(g1) + torch.abs(g2)

    return image_gradient

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def clamp(value, min=0.0, max=1.0):
    return torch.clamp(value, min=min, max=max)

def RGB2Y_Cr_Cb(rgb_image):
    R = rgb_image[0:1]
    G = rgb_image[1:2]
    B = rgb_image[2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = clamp(Y)
    Cr = clamp(Cr)
    Cb = clamp(Cb)

    return Y, Cr, Cb

def Y_Cr_Cb2RGB(Y, Cr, Cb):
    bias = torch.tensor([0.0, -0.5, -0.5]).to(Y.device)
    ycrcb = torch.cat([Y, Cr, Cb], dim=0)
    C, W, H = ycrcb.shape
    im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0],
         [1.403, -0.714, 0.0],
         [0.0, -0.344, 1.773]]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.transpose(0, 1).reshape(C, W, H)
    rgb_image = clamp(out)

    return rgb_image