#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Only use one card
import sys

import numpy
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from my_utils import get_dataset, loss_function
from options import args_parser
from update import test_inference, SegmentationMetric
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, UNet, Unet

if __name__ == '__main__':
    args = args_parser()
    # args.gpu = 'cuda'  # delete
    if args.gpu != 'cpu':
        torch.cuda.set_device(args.gpu)
    # device = args.gpu if args.gpu else 'cpu'
    # torch.cuda.set_device(args.gpu)
    device = args.gpu

    print(device)
    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    elif args.model == 'unet':
        # Unet
        global_model = UNet(args=args)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    # print(global_model)

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    trainloader = DataLoader(train_dataset, batch_size=args.trainloder_BatchSize, shuffle=True)
    # metric = SegmentationMetric(args.Metric_num_classes, device).to(device)  # NUM_CLASSES represents it has the number of NUM_CLASSES categories

    epoch_loss = []
    epoch_mIOU = []

    print(f"start training!")
    pbar = tqdm(range(args.epochs), total=args.epochs, file=sys.stdout, unit='epoch', ncols=args.ncols)
    for epoch in pbar:
        # metric.reset()
        metric = SegmentationMetric(args.Metric_num_classes, device).to(device)  # NUM_CLASSES represents it has the number of NUM_CLASSES categories
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = global_model(images)
            loss = loss_function(args, outputs, labels, device)
            metric.addBatch(outputs.squeeze(1) > 0.5, labels)  # [batch_size, width, height]
            # print(str(epoch) + "-" + str(batch_idx), numpy.sum((outputs.squeeze(1) > 0.5).cpu().detach().numpy()), numpy.sum(labels.cpu().detach().numpy()))
            # print(str(epoch) + "-" + str(batch_idx) + "-" + str(metric.MeanIOU()))
            loss.backward()
            optimizer.step()

            # if batch_idx % 5 == 0 or batch_idx == len(trainloader) - 1:
            s = 'Train Epoch {}: \tbatch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tGPU:{} '.format(
                epoch + 1,
                # args.epochs,
                (batch_idx + 1) * len(images),
                len(train_dataset),
                100 * (batch_idx + 1) * len(images) / len(train_dataset),
                loss.item(),
                '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if device is not 'cpu' else 0)
            )
            if epoch + 1 > 1:
                pbar.set_description(
                    s + f"\t\tTrain loss of epoch_{epoch + 1}: {loss_avg:.6f}, the mIOU: {meanIOU.cpu().detach().numpy():.6f}")
            else:
                pbar.set_description(s)

            batch_loss.append(loss.item())

        IOU, meanIOU = metric.MeanIOU()
        # print("mean", meanIOU)
        loss_avg = sum(batch_loss) / len(batch_loss)
        epoch_loss.append(loss_avg)
        epoch_mIOU.append(meanIOU.cpu().detach().numpy())

    # Plot loss
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    path = '../save/baseline/'
    loss_name = path + 'nn_{}_{}_{}_loss'.format(
        args.dataset,
        args.model,
        args.epochs)
    plt.savefig(loss_name + '.png')
    f = open(loss_name + '.txt', "w")
    epoch_loss = [str(i) + '\n' for i in epoch_loss]
    f.writelines(epoch_loss)
    f.close()

    # Plot mIOU
    plt.figure()
    plt.plot(range(len(epoch_mIOU)), epoch_mIOU)
    plt.xlabel('epochs')
    plt.ylabel('Train mIOU')
    mIOU_name = path + 'nn_{}_{}_{}_mIOU'.format(
        args.dataset,
        args.model,
        args.epochs)
    plt.savefig(mIOU_name + '.png')
    f = open(mIOU_name + '.txt', "w")
    epoch_mIOU = [str(i) + '\n' for i in epoch_mIOU]
    f.writelines(epoch_mIOU)
    f.close()

    # testing
    print(f"\nstart testing!")
    test_acc, test_loss = test_inference(args, global_model, test_dataset, device)
    test_results = [
        f"Test on {len(test_dataset)} samples\n",
        "Test mIOU: {:.2f}%\n".format(100 * test_acc),
        "Test loss: {:.4f}\n".format(test_loss),
    ]
    for i in test_results:
        print(i, end=" ")

    test_name = path + 'nn_{}_{}_{}_test_results'.format(
        args.dataset,
        args.model,
        args.epochs)
    f = open(test_name + '.txt', "w")
    f.writelines(test_results)
    f.close()
