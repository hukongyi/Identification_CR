#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /home/hky/github/Identification_CR/P00_readMCdata/CRMCDataset.py
# Project: /home/hky/github/Identification_CR/P00_readMCdata
# Created Date: 2022-06-16 14:55:54
# Author: Hu Kongyi
# Email:hukongyi@ihep.ac.cn
# -----
# Last Modified: 2022-06-17 13:31:01
# Modified By: Hu Kongyi
# -----
# HISTORY:
# Date      	By      	Comments
# ----------	--------	----------------------------------------------------
# 2022-06-16	K.Y.Hu		Create this file
###
import os.path as osp
from typing import Callable, Optional

import torch
from torch_geometric.data import InMemoryDataset
import numpy as np


class CRMCDataset(InMemoryDataset):

    def __init__(self,
                 root: str,
                 calibfile: str,
                 kneighbors: int = 4,
                 nchmin: int = 4,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.kneighbors = kneighbors
        self.nchmin = nchmin
        self.calibfile = calibfile
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return '/home/hky/Data/chenxu/'

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f'processed_kneighbors_{self.kneighbors}_nchmin_{self.nchmin}')

    @property
    def raw_file_names(self) -> str:
        return 'MCmodelBany4.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        data_list = list()
        orgin_data = np.load(self.raw_paths[0])

        pri = orgin_data["pri"]
        Tibet = orgin_data["Tibet"]
        MD = orgin_data["MD"]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate([data_list])
        torch.save((data, slices), self.processed_paths[0])


def pre_filter(data, nchmin):
    return len(data.x) >= nchmin

def get_one_trigger_from_numpy():




if __name__ == '__main__':
    CRMCDataset(root="/home/hky/github/Identification_CR/data",
                calibfile="/home/hky/for_mengy3/src16_1.8/userfile/mccaliblmzhai_20190312.cal",
                kneighbors=4,
                nchmin=16,
                pre_filter=pre_filter)
