#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, UNet
from my_utils import get_dataset, average_weights, exp_details, Logger


if __name__ == '__main__':
    start_time = time.time()


    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()

    folderPath = '../save/federated/'
    file_name = '{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].txt'. \
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)
    my_logger = Logger(folderPath, file_name)
    print = my_logger.print
    exp_details(args, my_logger)

    if args.gpu != 'cpu':
        torch.cuda.set_device(args.gpu)
    # device = 'cuda' if args.gpu is not None else 'cpu'
    # print(f"device = {device}")
    # torch.cuda.set_device(args.gpu)
    device = args.gpu
    print(device)

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

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

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, val_loss_list, net_list = [], [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0

    # for epoch in tqdm(range(args.epochs)):
    for epoch in range(args.epochs):
        local_weights, local_losses = [], []
        print(f'\n| Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        # print(m)  # m The number of customers to select for training. Not all customers will participate in the training in each round
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # print(idxs_users)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger, print=my_logger.print)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)

        train_accuracy.append(sum(list_acc)/len(list_acc))
        # train_loss.append(sum(list_loss)/len(list_loss))
        val_loss_list.append(sum(list_loss)/len(list_loss))
        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            # print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            print('Train mIOU: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset, args.gpu)

    print(f' \n Results after {args.epochs} global rounds of training:')
    # print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    # print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    print("|---- Avg Train mIOU: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Avg Train loss: {:.4f}".format(train_loss[-1]))
    print("|---- Test mIOU: {:.2f}%".format(100*test_acc))
    print("|---- Test loss: {:.4f}".format(test_loss))

    # Saving the objects train_loss and train_accuracy:
    # file_name = '../save/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)

    # f = open(file_name + '.txt', "w")
    # print(train_loss)
    # print(train_accuracy)
    # res = [str(loss) + '\t' + str(accuracy) + '\n' for loss, accuracy in zip(train_loss, train_accuracy)]
    # f.writelines(res)
    # f.close()
    my_logger.WriteIndices(train_loss, "train_loss")
    my_logger.WriteIndices(train_accuracy, "train_accuracy")


    # Plot loss
    plt.figure()
    plt.plot(range(len(train_loss)), train_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    loss_name = folderPath  + file_name[:-4] + '_loss'
    plt.savefig(loss_name + '.png')


    # Plot mIOU
    plt.figure()
    plt.plot(range(len(train_accuracy)), train_accuracy)
    plt.xlabel('epochs')
    plt.ylabel('Train mIOU')
    mIOU_name = folderPath + file_name[:-4] + '_mIOU'
    plt.savefig(mIOU_name + '.png')







    #
    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))


    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))

