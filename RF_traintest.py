# coding=utf-8
import skimage.io
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier

def ReadCity():
    SpectralImage = skimage.io.imread("data/lidar_train_4dsem.tif")
    SpectralImage = SpectralImage * 255
    print (SpectralImage.shape)
    GTImage =  np.load("data/train_labtwo.npy") #  ReadCity(SpectralPath[0], GTFile[0], 0.3)
    train_x = np.array([])   #训练集
    train_x.shape = 0, SpectralImage.shape[2]   #改变数组维数，以便于连接
    train_y = np.array([])
    train_y.shape = 0,
    for k in range(1, 2):
        Rindex, Cindex = np.where(GTImage==k)
        if Rindex.shape[0]==0:
            continue
        index_rand = np.random.permutation(Rindex.shape[0])
        Rindex = Rindex[index_rand]   #随机选择同一类的样本点
        Cindex = Cindex[index_rand]
        # SelectNum = np.int64(Rindex.shape[0])     #训练样本的数量
        Image0 = SpectralImage[Rindex[:], Cindex[:], :]
        train_x = np.concatenate((train_x, Image0), 0)
        Image0 = GTImage[Rindex[:], Cindex[:]]
        train_y = np.concatenate((train_y, Image0), 0)
    index_rand = np.random.permutation(train_x.shape[0])
    train_x = train_x[index_rand]
    train_y =train_y[index_rand]
    return train_x, train_y

train_x, train_y = ReadCity()
SpectralImage1 = skimage.io.imread("data/lidar_test_4dsem.tif")
SpectralImage1 = SpectralImage1 * 255
alldata_x = SpectralImage1.reshape((2404*8344, 4))
print (train_x.shape)
print (alldata_x.shape)
print ("STRAT")
rf = RandomForestClassifier(criterion="entropy", max_features="sqrt",
                            n_estimators=400, min_samples_leaf=2, n_jobs = -1, oob_score=True)
rf.fit(train_x, train_y)
print('OK!')
time_start = time.time()
predicted_y = np.array([])
predicted_y.shape = 0, 2
# time.time()为1970.1.1到当前时间的毫秒数
for i in range(0, 16):
    print (str(i))
    data_x = np.zeros((1253686, 3))
    data_x = alldata_x[i*1253686:(i+1)*1253686, :]
    predicted_y1 = rf.predict_proba(data_x)
    print (predicted_y1.shape)
    predicted_y = np.concatenate((predicted_y, predicted_y1), 0)
predicted_y = predicted_y.reshape((2404, 8344, 2))
np.save("result/labelrftwo.npy", predicted_y)
time_end = time.time()  # time.time()为1970.1.1到当前时间的毫秒数
print (time_end-time_start)
