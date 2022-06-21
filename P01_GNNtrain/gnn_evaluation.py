#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /home/hky/github/Identification_CR/P01_GNNtrain/gnn_evaluation.py
# Project: /home/hky/github/Identification_CR/P01_GNNtrain
# Created Date: 2022-06-20 13:42:00
# Author: Hu Kongyi
# Email:hukongyi@ihep.ac.cn
# -----
# Last Modified: 2022-06-21 18:56:26
# Modified By: Hu Kongyi
# -----
# HISTORY:
# Date      	By      	Comments
# ----------	--------	----------------------------------------------------
# 2022-06-20	K.Y.Hu		create this file
###
from CRMCDataset import CRMCDataset, pre_filter

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, MaxPool2d, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader


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
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()


# Get acc. of GNN model.
def test(loader, model, device):
    model.eval()

    correct = 0
    TP = np.zeros(3)
    FP = np.zeros(3)
    FN = np.zeros(3)
    TN = np.zeros(3)
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        for i in range(3):
            tmp = pred[torch.where(data.y == i)]
            TP[i] += len(tmp[tmp == i])
            FN[i] += len(tmp[tmp != i])
            tmp = pred[torch.where(data.y != i)]
            FP[i] += len(tmp[tmp == i])
            TN[i] += len(tmp[tmp != i])
    name_list = ['H+He', 'other', 'Fe']
    for i in range(3):
        print(name_list[i])
        print(f'{TP[i]}\t{FP[i]}\n{FN[i]}\t{TN[i]}')
    return correct / len(loader.dataset)


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
        self.fc1 = Linear(16 * 2 * 2, 64)  # 必须为16*5*5

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
    train_index, test_index = train_test_split(list(range(len(dataset))), test_size=0.1)
    train_dataset = dataset[train_index]
    test_dataset = dataset[test_index]
    batch_size = 1024
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    layers = 3
    hidden = 64

    start_lr = 0.001

    model = GIN_CNN(dataset, layers, hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.5,
                                                           patience=5,
                                                           min_lr=0.0000001)

    for epoch in range(1, 1 + 1):
        lr = scheduler.optimizer.param_groups[0]['lr']
        train(train_loader, model, optimizer, device)
        test_acc = test(test_loader, model, device)
        scheduler.step(test_acc)
        print('all_acc:', test_acc)
