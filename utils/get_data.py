# -*-coding:utf-8 -*-

"""
# File       : get_data.py
# Time       ：2024/9/23 16:47
# Author     ：Qiang42
# version    ：python 3.8
# Description：
"""

import cv2
import os
import torch
import numpy as np
import tarfile
from PIL import Image
from torchvision import transforms
from torch.utils import data

import utils

to_tensor = transforms.Compose([transforms.ToTensor()])
DATASETSNAME = 'MSRS_240_320'

class get_data(data.Dataset):
    """
    # -dataset.tar
    #     --train
    #         ---ir
    #             ---- imgs
    #             ---- ...
    #         ---vi
    #             ---- imgs
    #             ---- ...
    #     --val
    #         ---ir
    #             ---- imgs
    #             ---- ...
    #         ---vi
    #             ---- imgs
    #             ---- ...
    #             ---- ...
    """
    def __init__(self, args, train=True, transform=to_tensor):
        super().__init__()

        self.tar_path = args.dataset_path
        self.transform = transform
        self.members = []

        with tarfile.open(self.tar_path, 'r') as tar_ref:
            if train:
                print('Loading train datasets...')
                data_dir = DATASETSNAME + '/train'

            else:
                print('Loading val datasets...')
                data_dir = DATASETSNAME + '/val'

            for member in tar_ref.getmembers():
                if member.isfile() and member.name.startswith(data_dir):
                    self.members.append(member)

        self.inf_members = [m for m in self.members if 'ir/' in m.name]
        self.vis_members = [m for m in self.members if 'vi/' in m.name]

        self.inf_members.sort(key=lambda m: os.path.basename(m.name))
        self.vis_members.sort(key=lambda m: os.path.basename(m.name))

    def __getitem__(self, index):
        with tarfile.open(self.tar_path, 'r') as tar_ref:
            inf_member = self.inf_members[index]
            vis_member = self.vis_members[index]

            inf_image = Image.open(tar_ref.extractfile(inf_member)).convert('L')
            vis_image = Image.open(tar_ref.extractfile(vis_member)).convert('RGB')

            if self.transform:
                inf_image = self.transform(inf_image)
                vis_image = self.transform(vis_image)


            vis_y_image, vis_cr_image, vis_cb_image = utils.RGB2Y_Cr_Cb(vis_image)
            seg_label = 0

            return vis_image, vis_y_image, vis_cr_image, vis_cb_image, inf_image, seg_label, os.path.basename(vis_member.name)

    def __len__(self):
        return len(self.inf_members)

class get_data_test(data.Dataset):
    def __init__(self, tar_path, dataset, transform=to_tensor):
        super().__init__()
        self.tar_path = tar_path
        self.transform = transform
        self.members = []

        with tarfile.open(self.tar_path, 'r') as tar_ref:
             for member in tar_ref.getmembers():
                if member.isfile():
                    self.members.append(member)
        self.inf_members = [m for m in self.members if 'ir/' in m.name]
        self.vis_members = [m for m in self.members if 'vi/' in m.name]

        self.inf_members.sort(key=lambda m: os.path.basename(m.name))
        self.vis_members.sort(key=lambda m: os.path.basename(m.name))

    def __getitem__(self, index):
        with tarfile.open(self.tar_path, 'r') as tar_ref:
            inf_member = self.inf_members[index]
            vis_member = self.vis_members[index]

            inf_image = Image.open(tar_ref.extractfile(inf_member)).convert('L')
            vis_image = Image.open(tar_ref.extractfile(vis_member)).convert('RGB')

            if self.transform:
                inf_image = self.transform(inf_image)
                vis_image = self.transform(vis_image)

            vis_y_image, vis_cr_image, vis_cb_image = utils.RGB2Y_Cr_Cb(vis_image)


            return vis_image, vis_y_image, vis_cr_image, vis_cb_image, inf_image, os.path.basename(vis_member.name)

    def __len__(self):
        return len(self.inf_members)