#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from sampling import carvana_image_iid, carvana_image_noniid
from DataSet import carvana_image_DataSet


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if args.dataset == 'cifar':
        print(f"data set is cifar")
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif (args.dataset == 'mnist' or args.dataset == 'fmnist'):
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
            print(f"data set is mnist")
        elif args.dataset == 'fmnist':
            data_dir = '../data/fmnist/'
            print(f"data set is fmnist")
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    elif args.dataset == 'carvana_image':
        # print(f"data set is carvana_image")

        data_dir = '../data/carvana_image/train'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             ])
        dataset = carvana_image_DataSet(data_dir, train=True, transform=apply_transform, imgSize=args.resize, debug=args.debug)
        len_dataset = len(dataset)
        indices = np.random.permutation(len_dataset).tolist()
        print(f"args.debug = {args.debug}")
        print(f"len_dataset = {len_dataset}")
        if args.debug:
            valid_idx = indices[0: 20]
            train_idx = indices[20: 100]
        else:
            test_size = len_dataset // 10
            valid_idx = indices[0: 2 * test_size]
            train_idx = indices[2 * test_size:]

        train_dataset = Subset(dataset, train_idx)  # 根据索引获得子数据集：训练集
        test_dataset = Subset(dataset, valid_idx)  # 获得验证集
        print(f"len_train_dataset = {len(train_dataset)}")
        print(f"len_test_dataset = {len(test_dataset)}")

        # train_dataset = carvana_image_DataSet(data_dir, train=True, transform=apply_transform, imgSize=args.resize)
        # test_dataset = carvana_image_DataSet(data_dir, train=False, transform=apply_transform, imgSize=args.resize)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = carvana_image_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = carvana_image_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


if __name__ == '__main__':
    pass
