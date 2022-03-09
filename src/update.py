#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings("ignore")

class SegmentationMetric(nn.Module):
    def __init__(self, NUM_CLASSES, use_cuda):
        super().__init__()
        self.numClass = NUM_CLASSES
        self.use_cuda = use_cuda
        self.confusionMatrix = torch.zeros((self.numClass, self.numClass), dtype=torch.int64)
        if self.use_cuda:
            self.confusionMatrix = self.confusionMatrix#.cuda() (cpu的话就注释掉)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    """
        预测1   预测0
    真1  TP     FN
    真0  FP     TN
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2 * Precision * Recall / (Precision + Recall)
    f1 = 2 * TP / (2 * TP + FP + FN + 1e-6)
    """

    def F1(self):
        precision = self.classPixelAccuracy()[1]  # 二分类 0是背景 1是目标
        recall = (torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0))[1]
        # print(f"precision = {precision}")
        # print(f"recall = {recall}")
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        return f1

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = torch.mean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def meanIntersectionOverUnion(self):
        IoU = self.IntersectionOverUnion()
        mIoU = torch.mean(IoU)
        return mIoU

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = torch.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = torch.sum(self.confusionMatrix, dim=1) + torch.sum(self.confusionMatrix, dim=0) - torch.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        # mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        # label = self.numClass * imgLabel[mask] + imgPredict[mask]

        label = self.numClass * imgLabel.flatten() + imgPredict.flatten()
        label = label.int()
        count = torch.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)

        return confusionMatrix

    def getConfusionMatrix(self):  # 同FCN中score.py的fast_hist()函数
        # cfM = self.confusionMatrix / np.sum(self.confusionMatrix, axis=0)
        cfM = self.confusionMatrix
        return cfM

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = torch.sum(self.confusionMatrix, dim=1) / (torch.sum(self.confusionMatrix) + 1e-8)
        iu = torch.diag(self.confusionMatrix) / (
                torch.sum(self.confusionMatrix, dim=1) + torch.sum(self.confusionMatrix, dim=0) -
                torch.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU



    def MeanIOU(self):
        IOU = self.IntersectionOverUnion()
        return IOU, torch.mean(IOU)

    def addBatch(self, imgPredict, imgLabel):
        imgPredict = imgPredict.squeeze(1)  # .cpu().numpy().astype('uint8')
        imgLabel = imgLabel.squeeze(1)  # .cpu().numpy().astype('uint8')
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)  # .float()
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = torch.zeros((self.numClass, self.numClass))
        if self.use_cuda:
            self.confusionMatrix = self.confusionMatrix.cuda()


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu is not None else 'cpu'
        # Default criterion set to NLL loss function
        if args.criterion == 'CE':
            self.criterion = torch.nn.CrossEntropyLoss()
        elif args.criterion == 'NLLLoss':
            self.criterion = torch.nn.NLLLoss().to(self.device)

        # self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]
        # print(f"len(idxs) = {len(idxs)}")
        # print(f"len(idxs_train) = {len(idxs_train)}")
        # print(f"len(idxs_val) = {len(idxs_val)}")
        # print(f"len(idxs_test) = {len(idxs_test)}")
        #
        # print(f"self.args.local_bs = {self.args.local_bs}")
        # print(f"int(len(idxs_val)/10) = {int(len(idxs_val)/10)}")
        # print(f"int(len(idxs_test)/10) = {int(len(idxs_test)/10)}")

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10) if int(len(idxs_val)/10) > 0 else 1, shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10) if int(len(idxs_test)/10) > 0 else 1, shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                if self.args.criterion == 'CE':
                    loss = self.criterion(log_probs.float(), labels.long())
                elif self.args.criterion == 'NLLLoss':
                    loss = self.criterion(log_probs, labels)
                # loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        metric = SegmentationMetric(self.args.num_classes, self.device).to(self.device)
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            if self.args.criterion == 'CE':
                batch_loss = self.criterion(outputs.float(), labels.long())
            elif self.args.criterion == 'NLLLoss':
                batch_loss = self.criterion(outputs, labels)
            # batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            if self.args.dataset == "carvana_image":
                metric.addBatch(outputs.argmax(1), labels)
            else:
                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)
        if self.args.dataset == "carvana_image":
            IOU, meanIOU = metric.MeanIOU()
            IOU = IOU.cpu().detach().numpy()
            meanIOU = meanIOU.cpu().detach().numpy()
            return meanIOU, loss
        else:
            accuracy = correct / total
            return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    if args.criterion == 'CE':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.criterion == 'NLLLoss':
        criterion = torch.nn.NLLLoss().to(device)

    testloader = DataLoader(test_dataset, batch_size=args.testloder_BatchSize,
                            shuffle=False)

    metric = SegmentationMetric(args.num_classes, device).to(device)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        if args.criterion == 'CE':
            batch_loss = criterion(outputs.float(), labels.long())
        elif args.criterion == 'NLLLoss':
            batch_loss = criterion(outputs, labels)

        loss += batch_loss.item()

        if args.dataset == "carvana_image":
            # print(f"outputs.argmax(1).shape = {outputs.argmax(1).shape}")
            # print(f"labels.shape = {labels.shape}")
            metric.addBatch(outputs.argmax(1), labels)
        else:
            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
    if args.dataset == "carvana_image":
        IOU, meanIOU = metric.MeanIOU()
        IOU = IOU.cpu().detach().numpy()
        meanIOU = meanIOU.cpu().detach().numpy()
        return meanIOU, loss
    else:
        accuracy = correct / total
        return accuracy, loss
