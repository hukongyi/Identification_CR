#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /home/hky/github/Identification_CR/P00_readMCdata/CRMCDataset.py
# Project: /home/hky/github/Identification_CR/P00_readMCdata
# Created Date: 2022-06-16 14:55:54
# Author: Hu Kongyi
# Email:hukongyi@ihep.ac.cn
# -----
# Last Modified: 2022-06-19 16:50:05
# Modified By: Hu Kongyi
# -----
# HISTORY:
# Date      	By      	Comments
# ----------	--------	----------------------------------------------------
# 2022-06-19	K.Y.Hu		realize should not use InMemoryDataset
# 2022-06-16	K.Y.Hu		Create this file
###
import os.path as osp
from typing import Callable, Optional

import torch
from torch_geometric.data import Data, Dataset
import numpy as np
from tqdm import tqdm


# can not use InMemoryDataset, need Dataset
class CRMCDataset(Dataset):

    def __init__(self,
                 root: str,
                 calibfile: str,
                 kneighbors: int = 4,
                 nchmin: int = 4,
                 x_range: int = 150,
                 y_range: int = 150,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.kneighbors = kneighbors
        self.nchmin = nchmin
        self.calibfile = calibfile
        self.x_range = x_range
        self.y_range = y_range
        self.calibfile = calibfile
        self.len = 0
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return '/home/hky/Data/chenxu/'

    @property
    def processed_dir(self) -> str:
        return osp.join(
            self.root,
            f'processed_kneighbors_{self.kneighbors}_nchmin_{self.nchmin}_{self.x_range}_{self.y_range}'
        )

    @property
    def raw_file_names(self) -> str:
        return 'MCmodelBany4.npz'

    @property
    def processed_file_names(self) -> str:
        return [f'data_{i}.pt' for i in range(self.len)]

    def download(self):
        pass

    def process(self):

        loc_x = list()
        loc_y = list()
        loc_z = list()

        with open(self.calibfile, "r") as f:
            for line in f:
                loc_x.append(float(line.split()[1]))
                loc_y.append(float(line.split()[2]))
                loc_z.append(float(line.split()[3]))
        loc_x = np.array(loc_x)
        loc_y = np.array(loc_y)
        loc_z = np.array(loc_z)

        orgin_data = np.load(self.raw_paths[0])

        # pri_e_num = orgin_data['pri_e_num']
        pri_id = orgin_data['pri_id']
        pri_e = orgin_data['pri_e']
        pri_theta = orgin_data['pri_theta']
        pri_phi = orgin_data['pri_phi']
        pri_ne = orgin_data['pri_ne']
        pri_core_x = orgin_data['pri_core_x']
        pri_core_y = orgin_data['pri_core_y']
        pri_sump = orgin_data['pri_sump']
        pri_n_hit = orgin_data['pri_n_hit']
        Tibetevent = orgin_data['Tibetevent']
        Tibet = orgin_data['Tibet']
        MDevent = orgin_data['MDevent']
        MD = orgin_data['MD']

        particle_id = list(set(pri_id))
        particle_id.sort()

        for j, i in enumerate(particle_id):
            pri_id[pri_id == i] = j

        for i in tqdm(range(len(pri_id))):
            if pri_n_hit[i] >= self.nchmin:
                data = get_one_trigger_from_numpy(
                    indices=i,
                    Tibetevent=Tibetevent,
                    Tibet=Tibet,
                    MDevent=MDevent,
                    MD=MD,
                    kneighbors=self.kneighbors,
                    loc_x=loc_x,
                    loc_y=loc_y,
                    loc_z=loc_z,
                    pri_id=pri_id,
                    pri_theta=pri_theta,
                    pri_phi=pri_phi,
                    pri_core_x=pri_core_x,
                    pri_core_y=pri_core_y,
                    pri_e=pri_e,
                    pri_ne=pri_ne,
                    pri_sump=pri_sump,
                )
                if self.pre_filter(data, self.nchmin, self.x_range, self.y_range):
                    torch.save(data, osp.join(self.processed_dir, f'data_{self.len}.pt'))
                    self.len += 1
        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]

        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]

        # data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])

    def len(self):
        return self.len

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data


def pre_filter(data, nchmin, x_range, y_range):
    return (data.num_nodes >= nchmin) and (abs(data.pri_core_x) < x_range) and (abs(data.pri_core_y)
                                                                                < y_range)


def get_one_trigger_from_numpy(indices: int, Tibetevent: np.ndarray, Tibet: np.ndarray,
                               MDevent: np.ndarray, MD: np.ndarray, kneighbors: int,
                               loc_x: np.ndarray, loc_y: np.ndarray, loc_z: np.ndarray,
                               pri_id: np.ndarray, pri_theta: np.ndarray, pri_phi: np.ndarray,
                               pri_core_x: np.ndarray, pri_core_y: np.ndarray, pri_e: np.ndarray,
                               pri_ne: np.ndarray, pri_sump: np.ndarray) -> Data:
    x = list()
    edge_index = list()
    if indices < len(Tibetevent):
        data_range_Tibet = Tibet[Tibetevent[indices]:Tibetevent[indices + 1], :]
        data_range_MD = MD[MDevent[indices]:MDevent[indices + 1], :]

    else:
        data_range_Tibet = Tibet[Tibetevent[indices]:, :]
        data_range_MD = MD[MDevent[indices], :]
    for i in data_range_Tibet:
        # !only use the \pm 300 ns data
        if (i[3] <= 300) and (i[3] >= -300) and (i[2] > 0.6):
            x.append([loc_x[int(i[1])], loc_y[int(i[1])], loc_z[int(i[1])], i[3], i[2]])

    x = np.array(x)
    if len(x) - 1 <= kneighbors:
        for i in range(len(x)):
            for j in range(len(x)):
                if i != j:
                    edge_index.append([i, j])
    else:
        for i in range(len(x)):
            # + or -
            distance = np.sum(
                (x[i, :3] - x[:, :3]) ** 2, axis=1) + (x[i, 3] - x[:, 3]) ** 2 * 0.3 ** 2
            # print(len(distance), kneighbors)
            min5 = np.argpartition(distance, kneighbors)[:kneighbors]
            for j in min5:
                if i != j:
                    if [i, j] not in edge_index:
                        edge_index.append([i, j])
                    if [j, i] not in edge_index:
                        edge_index.append([j, i])

    MD_tmp = np.zeros([4 * 16 * 4])
    for i in data_range_MD:
        MD_tmp[int(i[1])] = i[2]
    MD_tmp = MD_tmp.reshape(4, 16, 4)
    MD0 = MD_tmp[0, :, 0].reshape(4, 4)
    MD1 = MD_tmp[1, :, 0].reshape(4, 4)
    MD2 = MD_tmp[2, :, 0].reshape(4, 4)
    MD3 = MD_tmp[3, :, 0].reshape(4, 4)

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        MD0=torch.tensor(MD0, dtype=torch.float),
        MD1=torch.tensor(MD1, dtype=torch.float),
        MD2=torch.tensor(MD2, dtype=torch.float),
        MD3=torch.tensor(MD3, dtype=torch.float),
        pri_id=torch.tensor(pri_id[indices], dtype=torch.int),
        pri_theta=torch.tensor(pri_theta[indices], dtype=torch.float),
        pri_phi=torch.tensor(pri_phi[indices], dtype=torch.float),
        pri_core_x=torch.tensor(pri_core_x[indices], dtype=torch.float),
        pri_core_y=torch.tensor(pri_core_y[indices], dtype=torch.float),
        pri_e=torch.tensor(pri_e[indices], dtype=torch.float),
        pri_ne=torch.tensor(pri_ne[indices], dtype=torch.float),
        pri_sump=torch.tensor(pri_sump[indices], dtype=torch.float),
    )

    return data


if __name__ == '__main__':
    dataset = CRMCDataset(
        root="/home/hky/github/Identification_CR/data",
        calibfile="/home/hky/for_mengy3/src16_1.8/userfile/mccaliblmzhai_20190312.cal",
        kneighbors=8,
        nchmin=16,
        x_range=100,
        y_range=100,
        pre_filter=pre_filter)
    print(dataset[0])