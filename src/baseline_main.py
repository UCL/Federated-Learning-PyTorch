#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import sys

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from utils import get_dataset
from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, UNet


if __name__ == '__main__':
    args = args_parser()
    # args.gpu = 'cuda:0'  # 待删除
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu is not None else 'cpu'
    print(f"device = {device}")
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
    if args.criterion == 'CE':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.criterion == 'NLLLoss':
        criterion = torch.nn.NLLLoss().to(device)
    epoch_loss = []

    if args.debug:
        args.epochs = 1


    for epoch in range(args.epochs):
        # print('Train Epoch: {:2}/{:2} '.format(epoch+1, args.epochs ))
        batch_loss = []
        pbar = tqdm(enumerate(trainloader), total=len(trainloader), file=sys.stdout, unit="batch")
        for batch_idx, (images, labels) in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)
            if args.criterion == 'CE':
                loss = criterion(outputs.float(), labels.long())
            elif args.criterion == 'NLLLoss':
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # if batch_idx % 50 == 0:
            s = 'Train Epoch: {:2}/{:2}  Batch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1,
                args.epochs,
                batch_idx * len(images),
                len(trainloader.dataset),
                100. * batch_idx * len(images) / len(trainloader.dataset),
                loss.item())
            pbar.set_description(s)

            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        epoch_loss.append(loss_avg)

    # Plot loss
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.savefig('../save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
                                                 args.epochs))

    # testing
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    if args.dataset == "carvana_image":
        print("Test meanIOU: {:.2f}%".format(100*test_acc))
    else:
        print("Test Accuracy: {:.2f}%".format(100*test_acc))
