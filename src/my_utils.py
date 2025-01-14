#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import os

import torch
from torchvision import datasets, transforms
from typing import List, Dict

from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from sampling import carvana_image_iid, carvana_image_noniid
from DataSet import carvana_image_DataSet
import segmentation_models_pytorch as smp


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
        print(f"dataset is carvana_image")

        data_dir_train = '../data/carvana_image/train'
        data_dir_test = '../data/carvana_image/test'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             ])
        train_dataset = carvana_image_DataSet(data_dir_train, train=True, transform=apply_transform, imgSize=args.resize)
        test_dataset = carvana_image_DataSet(data_dir_test, train=False, transform=apply_transform, imgSize=args.resize)

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
                # user_groups = carvana_image_noniid(train_dataset, args.num_users)
                raise NotImplementedError() # carvana_image don't exist noiid

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


def loss_function(args, outputs, labels, device):
    if args.criterion == 'BCE':
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.]).to(device)).to(device)
        return criterion(outputs.float(), labels.unsqueeze(1).float())
    elif args.criterion == 'NLLLoss':
        criterion = torch.nn.NLLLoss().to(device)
        return criterion(outputs, labels)
    elif args.criterion == 'DiceLoss':
        criterion = smp.losses.DiceLoss(mode=args.diceloss_mode, ignore_index=args.ignore_index).to(device)
        # outputs's shape = torch.Size([1, 1, 32, 32])
        # labels's shape = torch.Size([1, 32, 32])
        return criterion(outputs, labels)


def exp_details(args, log):
    print = log.print
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


class Logger:
    def __init__(self, folderPath, log_name):
        self.folder = folderPath
        # os.mkdir(self.folder)  # Create a folder to store this log
        self.log_path = os.path.join(self.folder, log_name)
        self.indices_path = os.path.join(self.folder, log_name[:-4] + "_res.txt")
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
        if os.path.exists(self.indices_path):
            os.remove(self.indices_path)
        self.data = []
        self.df = None

    def print(self, text, show=True):
        if show:
            print(text)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(text + "\r")

    def WriteIndices(self, list, list_name):
        indices_file = open(self.indices_path, "a", encoding='utf-8')
        indices_file.write('\n\n' + list_name + ': \n')
        for line in list:
            indices_file.write(str(line) + '\n')
        indices_file.close()




if __name__ == '__main__':
    x = 'a'
    if x == 'a':
        a = 1
    elif x == 'b':
        a = 2
    elif x == 'c':
        a = 3
    print(a)
