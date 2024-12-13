# -*-coding:utf-8 -*-

"""
# File       : main.py
# Time       ：2024/12/12 15:49
# Author     ：Qiang42
# version    ：python 3.8
# Description：
"""

import os
import datetime
# import yaml
import csv
from pprint import pprint

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import *
from utils.get_data import get_data
from utils.early_stopping import EarlyStopping

from args import parse_args
from eva import evaluation_main
from train import train
from val import val
from test import test

from Model.DWSFusion import Generator
from Model.DWSFusion import Discriminator
from Loss.Loss import Loss

def main(device, args):
    G = Generator().to(device)
    D_IR = Discriminator().to(device)
    D_VI = Discriminator().to(device)
    optimizer_D_IR = optim.RMSprop(D_IR.parameters(), lr=args.learning_rate)
    optimizer_D_VI = optim.RMSprop(D_VI.parameters(), lr=args.learning_rate)

    parameters_G = count_parameters(G)
    print('count_parameters(G)', parameters_G)
    criterion = Loss()
    optimizer = optim.Adam(G.parameters(), lr=args.learning_rate)

    train_dataset = get_data(args, train=True)
    val_dataset = get_data(args, train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    print('Datasets done')

    # 学习率下降策略
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.lr_patience, factor=args.lr_factor, threshold=1e-5)
    # early_stopping
    early_stopping = EarlyStopping(args.log_dir, args.early_stopping_patience, verbose=True, delta=1e-5)

    train_losses, val_losses = [], []
    best_loss = float("inf")

    for epoch in range(0, args.epochs):
        train_loss = train(args, G, D_IR, D_VI, train_loader, criterion, optimizer, optimizer_D_IR, optimizer_D_VI, device)
        train_losses.append(train_loss)
        val_loss = val(args, G, D_IR, D_VI, val_loader, criterion, device)
        val_losses.append(val_loss)
        # 更新学习率
        scheduler.step(val_loss)

        checkpoint = {
            'epoch': epoch,
            'G_state_dict': G.state_dict(),
        }

        # 保存检查点模型
        if best_loss == float("inf"):
            best_loss = val_loss
            print("pass")
        else:
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(checkpoint, os.path.join(args.log_dir, "checkpoint.pth"))

        # 打印信息
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{args.epochs} "
              f"Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}, LR: {current_lr:.8f}")

        early_stopping(val_loss, G)
        # 达到早停止条件时，early_stop会被置为True
        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练

    with open(os.path.join(args.log_dir, "train_losses.csv"), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([[value] for value in train_losses])
    with open(os.path.join(args.log_dir, "val_losses.csv"), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([[value] for value in val_losses])


    segline = "=" * 40
    train_run_timestamp = datetime.datetime.now()
    print(f"{segline}train_run_timestamp:\t\t{train_run_timestamp - timestamp}{segline}")

    if args.test:
        print('testing on test set')
        test(G, args, device)

    if args.metrics:
        print('evaluation_main running')
        eva = evaluation_main(args)
        eva.evaluation_main(args)
        if not args.resume:
            evaluation_end_timestamp = datetime.datetime.now()
            print(f"{segline}evaluation_run_timestamp:\t{evaluation_end_timestamp - evaluation_end_timestamp}{segline}")

if __name__ == '__main__':
    timestamp = datetime.datetime.now()
    timestamp_date = timestamp.strftime("%Y%m%d")
    timestamp_time = timestamp.strftime("%H%M%S")
    # timestamp_time = '20241028'
    # timestamp_time = '152312'
    args = parse_args(timestamp_date, timestamp_time)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    init_seeds(torch.cuda.is_available(), args.seed)
    pprint(vars(args), sort_dicts=False)

    # 保存超参数信息到日志文件
    # with open(os.path.join(args.hyperparameters_path, "hyperparameters.yaml"), "w", encoding='utf-8') as f:
    #     yaml.dump(args, f, default_flow_style=False, allow_unicode=True)

    main(device, args)
    endtimestamp = datetime.datetime.now()
    print(f"starttime: {timestamp}")
    print(f"endtime: {endtimestamp}")
    print(f"elapsed time: {endtimestamp - timestamp}")
