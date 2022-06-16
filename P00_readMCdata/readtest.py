#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /home/hky/Identification/test.py
# Project: /home/hky/Identification
# Created Date: 2022-06-15 15:47:18
# Author: Hu Kongyi
# Email:hukongyi@ihep.ac.cn
# -----
# Last Modified: 2022-06-16 14:49:56
# Modified By: Hu Kongyi
# -----
# HISTORY:
# Date      	By      	Comments
# ----------	--------	----------------------------------------------------
# 2022-06-15	K.Y.Hu		Create file
###
import struct
import numpy as np
import time

start = time.time()
output_struct = struct.Struct('3i3di3di1992d256i')


class One_trigger(object):

    def __init__(self, buf):
        output = output_struct.unpack_from(buf)
        self.pri = {
            'e_num': output[0],
            'k': output[1],
            'id': output[2],
            'e': output[3],
            'theta': output[4],
            'phi': output[5],
            'ne': output[6],
            'core_x': output[7],
            'core_y': output[8],
            'sump': output[9],
            'n_hit': output[10],
        }
        self.prtcl = np.array(output[11:1007])
        self.timing = np.array(output[1007:2003])  # 光束到平面的时间
        self.photon = np.array(output[2003:]).reshape([4, 16, 4])
        switch_order = [12, 13, 14, 15, 8, 9, 10, 11]
        for i in range(8):
            self.switch_photon(i, switch_order[i])

    def switch_photon(self, i, j):
        self.photon[:, i, :], self.photon[:, j, :] = self.photon[:, j, :], self.photon[:, i, :]


data_file = open('/home/hky/Data/chenxu/all', 'rb')
len_list = list()
print(output_struct.size)
while (1):
    # for _ in range(5):
    buf = data_file.read(output_struct.size)
    if len(buf) != output_struct.size:
        break
    trigger = One_trigger(buf)
    len_list.append(len(trigger.prtcl[trigger.prtcl > 0]))
len_list = np.array(len_list)
np.save('./nch', len_list)
print(len(len_list))

data_file.close()
print(time.time() - start)  # 10 min
