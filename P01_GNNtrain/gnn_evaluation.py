#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /home/hky/github/Identification_CR/P01_GNNtrain/gnn_evaluation.py
# Project: /home/hky/github/Identification_CR/P01_GNNtrain
# Created Date: 2022-06-20 13:42:00
# Author: Hu Kongyi
# Email:hukongyi@ihep.ac.cn
# -----
# Last Modified: 2022-06-20 14:40:46
# Modified By: Hu Kongyi
# -----
# HISTORY:
# Date      	By      	Comments
# ----------	--------	----------------------------------------------------
# 2022-06-20	K.Y.Hu		create this file
###
from CRMCDataset import CRMCDataset, pre_filter

import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader


# One training epoch for GNN model.
def train(train_loader, model, optimizer, device):
    model.train()
    count = 0

    for data in train_loader:
        count += 1
        if count % 100 == 0:
            print(count)
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.pri_id)
        loss.backward()
        optimizer.step()


# Get acc. of GNN model.
def test(loader, model, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.pri_id).sum().item()
    return correct / len(loader.dataset)


# Taken from https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py.
class GIN(torch.nn.Module):

    def __init__(self, dataset, num_layers, hidden):
        super(GIN, self).__init__()
        self.conv1 = GINConv(Sequential(
            Linear(dataset.num_features, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            BN(hidden),
        ),
                             train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(Sequential(
                    Linear(hidden, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU(),
                    BN(hidden),
                ),
                        train_eps=True))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, len(dataset[0].pri_id))

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_index, test_index = train_test_split(list(range(len(dataset))), test_size=0.1)
    train_dataset = dataset[train_index.tolist()]
    test_dataset = dataset[test_index.tolist()]
    batch_size = 1024
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    layers = 3
    hidden = 64

    start_lr = 0.001

    model = GIN(dataset, layers, hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.5,
                                                           patience=5,
                                                           min_lr=0.0000001)

    for epoch in range(1, 200 + 1):
        lr = scheduler.optimizer.param_groups[0]['lr']
        train(train_loader, model, optimizer, device)
        test_acc = test(test_loader, model, device)
        scheduler.step(test_acc)
        print(test_acc)
