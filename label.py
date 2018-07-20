import numpy as np
import skimage.io
import random

train_sets = skimage.io.imread("data/2018_IEEE_GRSS_DFC_GT_TR.tif")
train_lab = np.zeros((1202, 4768))
labnum = [3, 7, 17]
def label(lab):
    for i in range(1, 21):
        if i in lab:
            label_location = np.where(train_sets == i)
            random_x = np.array(range(int(label_location[0].shape[0])))
            random.shuffle(random_x)
            if i in labnum:
                num = int(label_location[0].shape[0])
            else:
                num = 20000
            random_x = random_x[0:num,...]
            random_datax = label_location[0][random_x]
            random_datay = label_location[1][random_x]
        else:
            continue
        print (str(int(random_datax.shape[0])))
        for j in range(int(random_datax.shape[0])):
            train_lab[random_datax[j], random_datay[j]] = i
    return train_lab
lab3 = [8, 9, 20]
train_lab3 = label(lab3)
np.save("data/train_lab3.npy", train_lab3)
lab17 = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
train_lab17 = label(lab17)
np.save("data/train_lab17.npy", train_lab17)