import numpy as np
import cv2
import argparse
import sys
import os
from PIL import Image

high_freq = 11
low_freq = 3
rwidth = 400
rheight = 400


focalLength = 938.0
centerX = 319.5
centerY = 239.5
scalingFactor = 5000


def unwrap_r(low_f_file, high_f_file, folder):
    file1 = folder + low_f_file
    file2 = folder +  high_f_file
    print("file1:", file1)
    print("file2:", file2)
    wrap1data = np.zeros((rheight, rwidth), dtype=np.float)
    wrap2data = np.zeros((rheight, rwidth), dtype=np.float)
    wrap1data = np.load(file1)  # To be continued
    wrap2data = np.load(file2)
    unwrapdata = np.zeros((rheight, rwidth), dtype=np.float)
    kdata = np.zeros((rheight, rwidth), dtype=np.float)
    # wrap1data = cv2.GaussianBlur(wrap1data, (0, 0), 3, 3)
    # wrap2data = cv2.GaussianBlur(wrap2data, (0, 0), 4, 4)
    for i in range(rheight):
        for j in range(rwidth):
            kdata[i, j] = round(
                (high_freq/low_freq * 1.0 * wrap1data[i, j] - 1.0 * wrap2data[i, j])/2.0)
            unwrapdata[i, j] = (1.0 * wrap2data[i, j] +
                                2.0*kdata[i, j]*np.pi)/2

    print("I'm in unwrap_r")
    # print(unwrapdata[::20, ::20])
    wr_save = folder + 'unwrap.npy'
    np.save(wr_save, unwrapdata, allow_pickle=False)
    print(wr_save)
    # np.save('wrap24.pickle', wrap24data, allow_pickle=True)
    unwrapdata = np.multiply(unwrapdata, 1.0)
    # unwrapdata = np.unwrap(np.transpose(unwrapdata))
    # unwrapdata = cv2.GaussianBlur(unwrapdata,(0,0),3,3)
    unwrapdata = np.multiply(unwrapdata, 1.0)
    cv2.imwrite(folder + 'unwrap.png', unwrapdata)


def abs_unwrap_r(low_f, high_f, output_file,  folder):
    file1 = folder + '/' + low_f
    file2 = folder + '/' + high_f
    wrap1data = np.load(file1)  # To be continued
    wrap2data = np.load(file2)
    unwrapdata = np.zeros((rheight, rwidth), dtype=np.float)
    kdata = np.zeros((rheight, rwidth), dtype=np.float)
    # wrap1data = cv2.GaussianBlur(wrap1data, (0, 0), 3, 3)
    # wrap2data = cv2.GaussianBlur(wrap2data, (0, 0), 4, 4)
    for i in range(rheight):
        for j in range(rwidth):
            kdata[i, j] = round(
                (6 * 1.0 * wrap1data[i, j] - 1.0 * wrap2data[i, j])/2.0)
            unwrapdata[i, j] = 1.0 * wrap2data[i, j] + 2.0*kdata[i, j]*np.pi

    print(kdata[::20, ::20])
    wr_save = folder + 'unwrap.npy'
    np.save(wr_save, unwrapdata, allow_pickle=False)
    print(wr_save)
    # np.save('wrap24.pickle', wrap24data, allow_pickle=True)
    unwrapdata = np.multiply(unwrapdata, 1.0)
    # unwrapdata = np.unwrap(np.transpose(unwrapdata))
    unwrapdata = cv2.GaussianBlur(unwrapdata,(0,0),3,3)
    cv2.imwrite(folder + output_file, unwrapdata)


def deduct_ref(unwrap, reference, folder1, folder2):
    file1 = folder1 + '/' + unwrap
    file2 = folder2 + '/' + reference
    wrap_data = np.load(file1)  # To be continued
    ref_data = np.load(file2)
    net_data = np.subtract(ref_data, wrap_data)
    net_save = folder1 + 'net_wrap.npy'
    np.save(net_save, net_data, allow_pickle=False)
    print(net_save)
    # np.save('wrap24.pickle', wrap24data, allow_pickle=True)
    net_data = np.multiply(net_data, 1.0)
    cv2.imwrite(folder1 + 'abs_unwrap.png', net_data)
    