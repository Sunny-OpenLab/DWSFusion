# -*-coding:utf-8 -*-

"""
# File       : CBAM.py
# Time       ：2024/9/27 10:39
# Author     ：Qiang42
# version    ：python 3.8
# Description：
"""

import torch
import torch.nn as nn
from utils.FADC import AdaptiveDilatedConv

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // reduction, in_planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)
        return x * out.expand_as(x)

class SpatialAttention(nn.Module):
    # todo:UserWarning: Initializing zero-element tensors is a no-op
    #   warnings.warn("Initializing zero-element tensors is a no-op")
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = AdaptiveDilatedConv(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False) #   AdaptiveDilatedConv | nn.Conv2d
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        assert len(x.shape) == 4, f"Input tensor must have 4 dimensions, but got {x.shape}"
        assert x.size(1) > 0, "Input tensor's channel dimension must be greater than 0"
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_map = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.conv1(attention_map)
        attention_map = self.sigmoid(attention_map)
        return x * attention_map

class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# 测试 CBAM 模块
if __name__ == "__main__":
    x = torch.randn(4, 1, 32, 32)
    fa2m = FA2M(in_planes=1)
    y = fa2m(x)
    print(y.shape)  # 输出: torch.Size([4, 64, 32, 32])
