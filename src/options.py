#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=100,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=5,
                        help="number of users: K, such as 100")
    parser.add_argument('--frac', type=float, default=0.5,
                        help='the fraction of clients: C, such as 0.1')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='unet', help='model name : mlp, unet')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=3, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")
    # unet arguments
    parser.add_argument('--unet_use_bilinear', action='store_true', default=True)
    parser.add_argument('--resize', type=int, default=512)
    parser.add_argument('--criterion', type=str, default='DiceLoss',
                        help="BCE, NLLLoss, DiceLoss")
    parser.add_argument('--trainloder_BatchSize', type=int, default=10)
    parser.add_argument('--testloder_BatchSize', type=int, default=1)
    parser.add_argument('--Metric_num_classes', type=int, default=2)


    # diceloss arguments
    parser.add_argument('--ignore_index', type=int, default=0)
    parser.add_argument('--diceloss_mode', type=str, default='binary',
                        help="binary, multiclass")

    # tqdm arguments
    parser.add_argument('--ncols', type=int, default=200)

    # other arguments
    parser.add_argument('--dataset', type=str, default='carvana_image', help="name \
                        of dataset : mnist, carvana_image")
    parser.add_argument('--num_classes', type=int, default=1, help="number \
                        of classes : 10, 1")
    parser.add_argument('--gpu', default='cpu', help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU. cpu. cuda:0")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    args = parser.parse_args()
    return args
