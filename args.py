# -*-coding:utf-8 -*-

"""
# File       : args.py
# Time       ：2024/12/12 15:49
# Author     ：Qiang42
# version    ：python 3.8
# Description：
"""

import os
import argparse

def parse_args(timestamp_date, timestamp_time):
    log_root = os.path.join('logs', timestamp_date, timestamp_time)
    os.makedirs(log_root, exist_ok=True)
    print('log_root is: {}'.format(log_root))
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='DWSFusion')
    parser.add_argument('-e', '--epochs', type=int, default=2)
    parser.add_argument('-b', '--batch_size', type=int, default=16) # 64 / 16
    parser.add_argument('-l', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-w', '--workers', type=int, default=0) # 18/0

    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--lr_factor", type=float, default=0.1)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N')
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--metrics', type=bool, default=True)

    # PATH
    parser.add_argument("--dataset_path", type=str, default=r"./data/MSRS_240_320.tar") # MSRS_240_320_Seg
    parser.add_argument("--test_dataset_path", type=str, default=r"./data/test_data")
    parser.add_argument("--results_path", type=str, default=r"./results")
    parser.add_argument('--hyperparameters_path', type=str, default=log_root)
    parser.add_argument('--log_dir', type=str, default=log_root, help="Log directory")
    parser.add_argument("--test_save_path", type=str, default=os.path.join(log_root, "results/"))
    parser.add_argument("--W_save_path", type=str, default=os.path.join(log_root, "W_plt"))

    args = parser.parse_args()

    return args

