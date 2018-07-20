# coding=utf-8
import numpy as np
import skimage.io
import scipy.misc
import scipy.signal as signal

im = np.load('result/endlab.npy')
data = []
width, height = im.size
for h in range(height):
    row = []
    for w in range(width):
        value = im.getpixel((w,h))
        row.append(value)
    data.append(row)
# 二维中值滤波
data = np.float32(data)
# 滤波窗口的大小会对结果产生很大影响
data = signal.medfilt2d(data, (3,3))
# 创建并保存结果图像
for h in range(height):
    for w in range(width):
        im.putpixel((w,h), int(data[h][w]))

im[0, 0] = im[0, 1]
im[2403, 0] = im[2403, 1]
im[0,4767]= im[0, 4766]
im[2403, 4767] = im[2403, 4766]
skimage.io.imsave('result/gt.tif', im)

colorbar = [[0,208,0],[136,255,0],[35,108,59],[0,143,0],[0,76,0],[165,82,42],[0,242,242],[255,255,255],[216,191,216],
            [255,0,0],[156,248,139],[81,110,107],[167,0,0],[79,0,0],[238,165,24],[255,255,0],[255,0,255],[155,0,160],
            [0,0,244],[180,199,223]]

image = np.zeros((2404, 8344, 3))
for row in range(2404):
    for col in range(8344):
        label = int(im[row, col])
        image[row, col, 0] = colorbar[label-1][0]
        image[row, col, 1] = colorbar[label-1][1]
        image[row, col, 2] = colorbar[label-1][2]
scipy.misc.imsave('result/gt_rgb.tif', image)