#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /home/hky/github/Identification_CR/P01_GNNtrain/gnn_evaluation.py
# Project: /home/hky/github/Identification_CR/P01_GNNtrain
# Created Date: 2022-06-20 13:42:00
# Author: Hu Kongyi
# Email:hukongyi@ihep.ac.cn
# -----
# Last Modified: 2022-06-22 20:46:53
# Modified By: Hu Kongyi
# -----
# HISTORY:
# Date      	By      	Comments
# ----------	--------	----------------------------------------------------
# 2022-06-20	K.Y.Hu		create this file
###
from curses import echo
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, MaxPool2d, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from CRMCDataset import CRMCDataset, pre_filter

particle_list = ['H', 'He', 'C', 'Mg', 'Cl', 'Fe']


# One training epoch for GNN model.
def train(train_loader, model, optimizer, device):
    model.train()
    count = 0

    for data in train_loader:
        count += 1
        print(count)
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(
            output,
            data.y,
            weight=torch.Tensor([0.008, 0.015, 0.073, 0.206, 1, 0.308]),
        )
        loss.backward()
        optimizer.step()


# Get acc. of GNN model.
def test(loader, model, device, epoch):
    model.eval()
    output_logsoftmax = list()
    label = list()
    correct = 0
    ConfusionMatrix = np.zeros([6, 6])
    count = 0
    for data in loader:
        count += 1
        print(count)
        data = data.to(device)
        output = model(data)
        output_logsoftmax.append(output.cpu().detach().numpy())
        label.append(data.y.numpy())
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()

        for i in range(6):
            tmp_T = pred[torch.where(data.y == i)]
            for j in range(6):
                ConfusionMatrix[j, i] += len(tmp_T[tmp_T == j])

    F1_score = getF1_score(ConfusionMatrix)
    output_logsoftmax = np.concatenate(output_logsoftmax)
    output_softmax = np.exp(output_logsoftmax)
    label = np.concatenate(label)
    drawpath = f'/home/hky/github/Identification_CR/P01_GNNtrain/train{epoch}/'
    draw_ROC_classification(drawpath, output_softmax, label)
    return np.mean(F1_score)


def getF1_score(ConfusionMatrix):
    Precission = np.zeros(6)
    Recall = np.zeros(6)
    for i in range(6):
        Precission[i] = ConfusionMatrix[i, i] / np.sum(ConfusionMatrix[i, :])
        Recall[i] = ConfusionMatrix[i, i] / np.sum(ConfusionMatrix[:, i])
    F1_score = 2 * Precission * Recall / (Precission + Recall)
    return F1_score


def draw_ROC_classification(drawpath, output_softmax, label):
    if not os.path.exists(drawpath):
        os.makedirs(drawpath)
    for i, particle_name in enumerate(particle_list):
        plt.figure(figsize=(16, 16))

        Tr = output_softmax[np.where(label == i), i][0]
        Fa = output_softmax[np.where(label != i), i][0]
        plt.hist(Tr, bins=20, range=(0, 1), density=True, color='r', alpha=0.5)
        plt.hist(Fa, bins=20, range=(0, 1), density=True, color='b', alpha=0.5)
        plt.title(f'{particle_name}')
        plt.savefig(drawpath + particle_name + '_classification.png')
        plt.close()

        plt.figure(figsize=(16, 16))
        TPR = list()
        FPR = list()
        T = np.arange(0, 1 + 0.01, 0.01)
        for T_this in T:
            TP = np.sum([output_softmax[np.where(label == i), i][0] > T_this])
            TPR.append(TP / np.sum(label == i))
            FP = np.sum([output_softmax[np.where(label != i), i][0] > T_this])
            FPR.append(FP / np.sum(output_softmax[:, i] < T_this))

        plt.plot(TPR, FPR)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title(f'{particle_name}_ROC')
        plt.savefig(drawpath + particle_name + '_ROC.png')
        plt.close()


# Taken from https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py.
class GIN_CNN(torch.nn.Module):

    def __init__(self, dataset, num_layers, hidden):
        super(GIN_CNN, self).__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(dataset.num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ),
            train_eps=True,
        )
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ),
                    train_eps=True,
                ))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden + 64, 64)
        self.lin3 = Linear(64, dataset.num_classes)

        self.conv1_cnn = Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2_cnn = Conv2d(6, 16, kernel_size=5)
        self.mp = MaxPool2d(2)
        self.relu = ReLU()
        self.fc1 = Linear(16 * 2 * 2, 64)  # 根据图像大小选择

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        batch_number = (torch.max(batch) - torch.min(batch) + 1).item()
        x_MD = data.MD.view(batch_number, 1, 12, 12)
        x_MD = self.relu(self.mp(self.conv1_cnn(x_MD)))
        x_MD = self.relu(self.conv2_cnn(x_MD))
        x_MD = x_MD.view(batch_number, 16 * 2 * 2)
        x_MD = self.relu(self.fc1(x_MD))

        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = torch.cat((x, x_MD), 1)

        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


if __name__ == '__main__':
    dataset = CRMCDataset(
        root="/home/hky/Data/chenxu/pygDataset/",
        calibfile="/home/hky/for_mengy3/src16_1.8/userfile/mccaliblmzhai_20190312.cal",
        kneighbors=8,
        nchmin=16,
        x_range=100,
        y_range=100,
        pre_filter=pre_filter)
    print(len(dataset))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_index, test_index = train_test_split(
        list(range(len(dataset))),
        test_size=0.1,
        random_state=1,
    )
    train_dataset = dataset[train_index]
    test_dataset = dataset[test_index]
    batch_size = 1024
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=10,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=10,
    )
    layers = 3
    hidden = 64

    start_lr = 0.001

    model = GIN_CNN(dataset, layers, hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=0.0000001,
    )

    for epoch in range(1, 10 + 1):
        lr = scheduler.optimizer.param_groups[0]['lr']
        train(train_loader, model, optimizer, device)
        F1_score = test(test_loader, model, device, epoch)
        scheduler.step(F1_score)
        print(f'{epoch} F1_score:{F1_score}')
        torch.save(
            model.state_dict(),
            f'/home/hky/github/Identification_CR/P01_GNNtrain/train{epoch}/train{epoch}.pth',
        )
