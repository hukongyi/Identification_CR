#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /home/hky/github/Identification_CR/P00_readMCdata/savetonpz.py
# Project: /home/hky/github/Identification_CR/P00_readMCdata
# Created Date: 2022-06-15 15:47:18
# Author: Hu Kongyi
# Email:hukongyi@ihep.ac.cn
# -----
# Last Modified: 2022-06-17 14:05:42
# Modified By: Hu Kongyi
# -----
# HISTORY:
# Date      	By      	Comments
# ----------	--------	----------------------------------------------------
# 2022-06-16	K.Y.Hu		finsh method
# 2022-06-16	K.Y.Hu		add save to numpy method
# 2022-06-15	K.Y.Hu		Create file
###
import struct
import numpy as np
import time
import datetime
import gzip

from sendwecom.sendwecom import send_to_wecom_after_finish


class One_trigger(object):
    """get data for one event
    """

    def __init__(self, buf: bytes, output_struct: struct.Struct):
        """init

        Args:
            buf (bytes): read data using binary
            output_struct (struct.Struct): data form with struct define by c++
        """
        output = output_struct.unpack_from(buf)
        self.pri_numpy = np.array(output[:11])
        # self.pri = {
        #     'e_num': output[0],
        #     'k': output[1],
        #     'id': output[2],
        #     'e': output[3],
        #     'theta': output[4],
        #     'phi': output[5],
        #     'ne': output[6],
        #     'core_x': output[7],
        #     'core_y': output[8],
        #     'sump': output[9],
        #     'n_hit': output[10],
        # }
        self.prtcl = np.array(output[11:1007])
        self.timing = np.array(output[1007:2003])  # 光束到平面的时间为零点
        self.photon = np.array(output[2003:]).reshape([4, 16, 4])
        switch_order = [12, 13, 14, 15, 8, 9, 10, 11]
        for i in range(8):
            self.switch_photon(i, switch_order[i])

    def switch_photon(self, i: int, j: int):
        """switch i,j because of some mistakes in MD order

        Args:
            i (int): first order for switch
            j (int): second order for switch
        """
        self.photon[:, [i, j], :] = self.photon[:, [j, i], :]


class Data(object):
    """Storing data for all MC event by sparse matrix
    can not using list(out of memory)
    """

    def __init__(self):
        # primary particle information
        self.pri = np.zeros([6225474, 11])
        # Tibet-III event number correspond Tibet
        # with 4 element,
        # number of event,
        # number of fire detector,
        # number of photons, time
        self.Tibet = np.zeros([99869316, 4])
        # with 3 element,
        # number of event,
        # number of fire detector,
        # Number of photons arriving at pmt
        self.MD = np.zeros([76451312, 3])
        self.count = 0
        self.Tibetcount = 0
        self.MDcount = 0

    def addevent(self, trigger: One_trigger):
        """add one event from trigger

        Args:
            trigger (One_trigger): read data for one event
        """
        self.pri[self.count] = trigger.pri_numpy

        order = np.where(trigger.prtcl != 0)[0]
        for i in order:
            self.Tibet[self.Tibetcount] = np.array(
                [self.count, i, trigger.prtcl[i], trigger.timing[i]])
            self.Tibetcount += 1

        photon = trigger.photon.reshape(1, -1)[0]
        order = np.where(photon != 0)[0]
        for i in order:
            self.MD[self.MDcount] = np.array([self.count, i, photon[i]])
            self.MDcount += 1

        self.count += 1

    def save(self, savepath: str):
        """save to savepath with npz

        Args:
            savepath (str): path to save
        """
        # np.savez_compressed(savepath, pri=self.pri, Tibet=self.Tibet, MDevent=self.MD)
        np.savez(savepath, pri=self.pri, Tibet=self.Tibet, MD=self.MD)


# @send_to_wecom_after_finish
def savetonpz(originpath: str, savepath: str):
    """resave data to npz file

    Args:
        originpath (str): data path to read
        savepath (str): path to save

    Returns:
        str: messgae to send to wecom
    """
    start_time = time.time()
    output_struct = struct.Struct('3i3di3di1992d256i')

    data_numpy = Data()
    with gzip.open(originpath, 'rb') as data_file:
        # while (1):
        for i in range(1):
            buf = data_file.read(output_struct.size)
            if len(buf) != output_struct.size:
                break
            trigger = One_trigger(buf, output_struct)
            data_numpy.addevent(trigger)
    # data_numpy.save(savepath)
    print(f'用时：{datetime.timedelta(seconds=time.time() - start_time)}')  # 15-20 min
    # return f'save in {savepath} success!'


if __name__ == '__main__':
    originpath = '/home/hky/Data/chenxu/MCmodelBany4.gz'
    savepath = '/home/hky/Data/chenxu/MCmodelBany4.npz'
    savetonpz(originpath, savepath)
