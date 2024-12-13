# -*-coding:utf-8 -*-

"""
# File       : test.py
# Time       ：2024/12/12 15:57
# Author     ：Qiang42
# version    ：python 3.8
# Description：
"""


import os
from args import parse_args

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from pprint import pprint

import utils
from utils.get_data import get_data_test
from eva import evaluation_main
from Model.DWSFusion import Generator

def test(model, args, device):
    for dataset in os.listdir(args.test_dataset_path):
        test_dataset = get_data_test(os.path.join(args.test_dataset_path, dataset), dataset)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
        model.load_state_dict(torch.load(os.path.join(args.log_dir, 'best_network.pth')))
        model.eval()
        model.to(device)

        test_tqdm = tqdm(test_loader, total=len(test_loader), leave=False)
        with torch.no_grad():
            for vis_image, vis_y_image, cr, cb, ir_image, name in test_tqdm:
                vis_y_image = vis_y_image.to(device)
                cb = cb.to(device)
                cr = cr.to(device)
                ir_image = ir_image.to(device)

                fused_image, W_IR, W_VI, F_IR, F_VI, ir_w, vi_w = model(ir_image, vis_y_image)
                rgb_fused_image = utils.Y_Cr_Cb2RGB(fused_image[0], cr[0], cb[0])
                rgb_fused_image = transforms.ToPILImage(mode='RGB')(rgb_fused_image)
                if not os.path.exists(os.path.join(args.test_save_path, dataset.replace(".tar", ""))):
                    os.makedirs(os.path.join(args.test_save_path, dataset.replace(".tar", "")))
                rgb_fused_image.save(f'{args.test_save_path}/{dataset.replace(".tar", "")}/{name[0]}')
                test_tqdm.set_description("{} | {}".format(dataset.replace(".tar", ""), name[0]))

if __name__ == '__main__':
    timestamp_date = '20241122'
    timestamp_time = '175858'

    args = parse_args(timestamp_date, timestamp_time)

    pprint(vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator()
    print('testing on test set')
    test(model, args, device)
    print('evaluation_main running')
    eva = evaluation_main(args)
    eva.evaluation_main(args)
