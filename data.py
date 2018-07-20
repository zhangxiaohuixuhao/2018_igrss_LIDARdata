from skimage import io
import numpy as np

def image(img):
    for i in range(int(image.shape[0])):
        for j in range(int(image.shape[1])):
            if img[i,j] >= 10**30:
                img[i,j] = image[i,j - 1]
    return img
c1 = io.imread("data/c1.tif")
c1 = image(c1)
c2 = io.imread("data/c2.tif")
c2 = image(c2)
c3 = io.imread("data/c3.tif")
c3 = image(c3)
DSM = io.imread("data/DSM.tif")
DEM_C123_TLI = io.imread("data/DEM_C123_TLI.tif")
DSM = DSM - DEM_C123_TLI

LIDAR = np.zeros((DSM.shape[0], DSM.shape[1], 4))
LIDAR[:, :, 0] = c1
LIDAR[:, :, 1] = c2
LIDAR[:, :, 2] = c3
LIDAR[:, :, 3] = DSM

for i in range(4):
    max = np.max(LIDAR[:, :, i])
    min = np.min(LIDAR[:, :, i])
    if max - min == 0:
        continue
    else:
        LIDAR[:, :, i] = ((LIDAR[:, :, i] - max)*255)/(max-min)
io.imsave("data/lidar_test_4dsem.tif", LIDAR)
LIDAR = LIDAR[1202:2404,1192:5960,...]
io.imsave("data/lidar_train_4dsem.tif", LIDAR)