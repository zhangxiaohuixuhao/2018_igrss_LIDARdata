import numpy as np
import skimage.io
import random
import scipy.io as sio
train_sets = skimage.io.imread("data/2018_IEEE_GRSS_DFC_GT_TR.tif")
train_lab = np.zeros((1202, 4768))
lab = [8, 9, 20]
labnum = [3, 7, 17]
for i in range(1, 21):
    if i in lab:
        num = 20000
        label_location = np.where(train_sets == i)
        random_x = np.array(range(int(label_location[0].shape[0])))
        random.shuffle(random_x)
        random_x = random_x[0:20000,...]
        random_datax = label_location[0][random_x]
        random_datay = label_location[1][random_x]
    else:
        label_location = np.where(train_sets == i)
        random_x = np.array(range(int(label_location[0].shape[0])))
        random.shuffle(random_x)
        if i in labnum:
            num = int(label_location[0].shape[0])
        else:
            num = 3000
        random_x = random_x[0:num,...]
        random_datax = label_location[0][random_x]
        random_datay = label_location[1][random_x]
    print (str(int(random_datax.shape[0])))
    if i in lab:
        for j in range(num):
            train_lab[random_datax[j], random_datay[j]] = 1
    else:
        for j in range(num):
            train_lab[random_datax[j], random_datay[j]] = 2
np.save("data/train_labtwo.npy", train_lab)
