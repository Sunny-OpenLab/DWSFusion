# -*-coding:utf-8 -*-

"""
# File       : DWSFusion.py
# Time       ：2024/12/12 16:00
# Author     ：Qiang42
# version    ：python 3.8
# Description：
"""

import utils
import torch.nn as nn
from utils.CBAM import *
import torch.nn.functional as F
from utils.FADC import AdaptiveDilatedConv

class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CBR, self).__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.cbr(x)

class CBR_D2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CBR_D2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.size(2) % 2 != 0 or x.size(3) % 2 != 0:
            x = F.pad(x, (0, x.size(3) % 2, 0, x.size(2) % 2))
        x = self.relu(self.bn(self.conv1(x)))
        x = self.maxpool(x)
        if x.size(2) % 2 != 0 or x.size(3) % 2 != 0:
            x = F.pad(x, (0, x.size(3) % 2, 0, x.size(2) % 2))
        x = self.relu(self.bn(self.conv2(x)))
        x = self.maxpool(x)

        return x

class CBR_U2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CBR_U2, self).__init__()
        self.upcbar = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.upcbar(x)

class WGet(nn.Module):
    def __init__(self):
        super(WGet, self).__init__()
        self.conv1 = CBR_D2(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.cbam1 = CBAM(in_planes=4)
        self.conv2 = CBR_D2(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.cbam2 = CBAM(in_planes=8)
        self.conv3 = CBR_U2(in_channels=8, out_channels=4, kernel_size=2, stride=2, padding=0)
        self.cbam3 = CBAM(in_planes=4)
        self.conv4 = CBR_U2(in_channels=8, out_channels=1, kernel_size=2, stride=2, padding=0)
        self.cbam4 = CBAM(in_planes=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        out1 = self.conv1(x)
        out1 = self.cbam1(out1)
        out2 = self.conv2(out1)
        out2 = self.cbam2(out2)
        out3 = self.conv3(out2)
        out3 = self.cbam3(out3)
        if out3.size() != out1.size():
            out3 = F.interpolate(out3, size=(out1.size(2), out1.size(3)), mode='bilinear', align_corners=False)
        cat = torch.cat((out1, out3), dim=1)
        out4 = self.conv4(cat)
        out4 = self.cbam4(out4)
        if out4.size() != identity.size():
            out4 = F.interpolate(out4, size=(identity.size(2), identity.size(3)), mode='bilinear', align_corners=False)
        out4 = out4 + identity
        out = self.sigmoid(out4)

        return out

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.feature_extract = nn.Sequential(
            AdaptiveDilatedConv(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1),
            # AdaptiveDilatedConv(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            CBAM(in_planes=4),
        )
        self.conv4 = AdaptiveDilatedConv(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.channel_match = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        features = self.feature_extract(x)
        identity = self.channel_match(x)
        skip = features + identity
        x = self.conv4(skip)
        return utils.clamp(x, min=0.0, max=1.0)

class alphaNet(nn.Module):
    def __init__(self):
        super(alphaNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(8, 1)

    def forward(self, ir, vi):
        x = torch.cat([ir, vi], dim=1)
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        return out.view(-1, 1, 1, 1)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.WGet_IR = WGet()
        self.WGet_VI = WGet()
        self.Fusion = Fusion()
        self.alphanet = alphaNet()

    def forward(self, ir, vi):
        alpha = self.alphanet(ir, vi)
        W_IR = self.WGet_IR(ir)
        W_VI = self.WGet_VI(vi)
        ir_w = ir * (alpha * W_IR + (1 - alpha)*(1 - W_VI))
        vi_w = vi * (alpha * W_VI + (1 - alpha)*(1 - W_IR))
        out_cat = torch.cat([ir_w, vi_w], dim=1)
        F = self.Fusion(out_cat)
        F_IR = self.WGet_IR(F)
        F_VI = self.WGet_VI(F)

        return F, W_IR, W_VI, F_IR, F_VI, ir_w, vi_w

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.avgpool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        x = self.flatten(x)
        fc1 = nn.Linear(x.shape[1], 1).to(x.device)
        x = fc1(x)

        return x


