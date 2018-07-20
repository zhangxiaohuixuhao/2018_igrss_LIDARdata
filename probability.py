# coding=utf-8
import scipy.io as sio
import skimage.io
import numpy as np

model_num = 2
pre_vgg = np.load('data/labelrftwo.npy')
pre_vgg = pre_vgg.reshape((2404*8344, 2))
pre_rf = np.load('data/labelvggtwo.npy')
pre_rf = pre_rf.reshape((2404*8344, 2))
pre_mean = (pre_vgg + pre_rf) / model_num
[row, col] = pre_mean.shape
label = np.zeros((row, 1))
for i in range(row):
    pre_mean_row = pre_mean[i, :]
    pre_mean_row = pre_mean_row.tolist()
    label[i] = pre_mean_row.index(max(pre_mean_row)) + 1
label = label.reshape(2404, 8344)
np.save("result/labeltwo.npy", label)