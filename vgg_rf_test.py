import numpy as np
import keras.backend as K
from keras.models import  Model
from constant import *
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from classify_model import lidar_model
import skimage.io
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

test_size = 1000
batch_size= 1000

lidar_channel = 4

batch_size_LID = 128

patch_size_LID = 16


def data_fix(data, unexpected_value):
    location=np.where(data > unexpected_value)
    for index in range(location[0].size):
        i=1
        while 1:
            patch = data[location[0][index]-i:location[0][index]+i,location[1][index]-i:location[1][index]+i]
            new_location = np.where(patch<unexpected_value)
            if new_location[0].size >0:
                data[location[0][index],location[1][index]] = data[location[0][index]-i+new_location[0][0],location[1][index]-i + new_location[1][0]]
                break
            i=i+1
    return data

train_sets3 = np.load("data/train_lab3.npy")
label_location3 = np.where(train_sets3 != 0)

train_sets17 = np.load("data/train_lab17.npy")
label_location17 = np.where(train_sets17 != 0)
#train data
lidar = skimage.io.imread("data/lidar_train_4dsem.tif")
lidar = lidar * 255
#test data
lidartest = skimage.io.imread("data/lidar_test_4dsem.tif")
lidartest = lidartest * 255
lidar_change = np.zeros((lidartest.shape[0] + 17, lidartest.shape[1] + 17, lidartest.shape[2]), dtype=np.float)
lidar_change[8:2412, 8:8352, :] = lidartest

def get_session(gpu_fraction=0.3):
    """
    This function is to allocate GPU memory a specific fraction
    Assume that you have 6GB of GPU memory and want to allocate ~2GB
    """

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


KTF.set_session(get_session(0.5))  # using 60% of total GPU Memory
os.system("nvidia-smi")  # Execute the command (a string) in a subshell
raw_input("Press Enter to continue...")

def boundary_judge(y,x,patch_size,height,width):
	if y < 0:
		y = 0
	if y > height-patch_size:
		y = height-patch_size
	if x < 0:
		x = 0
	if x > width-patch_size:
		x = width-patch_size
	return y,x

def randomtree_lidar3():
    test_x = np.array([])
    test_x.shape = 0, 512
    test_y = np.array([])
    test_y.shape = 0,
    K.clear_session()
    model = lidar_model()
    model.load_weights('model/lidar_model3.h5')
    print(model.summary())
    model = Model(inputs=model.input, outputs=model.get_layer('dense_2').output)
    print(model.summary())
    for j in range(int(label_location3[0].shape[0]//1000)):
        print str(j)
        lidar_batch_label = np.zeros((batch_size), dtype=int)
        lidar_batch_patch = np.zeros((batch_size, patch_size_LID, patch_size_LID, lidar_channel),
                                   dtype=np.uint8)
        for i in range(1000):
            num = j * 1000 + i
            X = label_location3[0][num]
            Y = label_location3[1][num]
            lab = train_sets3[X, Y]
            lidar_batch_label[i] = lab
            y = label_location3[0][num]
            x = label_location3[1][num]
            bias = int(patch_size_LID / 2)
            upper_y = y - bias
            left_x = x - bias
            upper_y, left_x = boundary_judge(upper_y, left_x, patch_size_LID, lidar.shape[0], lidar.shape[1])
            lidar_batch_patch[i] = lidar[upper_y:upper_y + patch_size_LID, left_x:left_x + patch_size_LID]
        test_y = np.concatenate((test_y, lidar_batch_label), 0)
        pre_labels = model.predict(lidar_batch_patch)
        pre_labels = pre_labels.reshape((test_size, 512))
        test_x = np.concatenate((test_x, pre_labels), 0)
    K.clear_session()
    rf = RandomForestClassifier(criterion="entropy", max_features="sqrt", n_estimators=400, min_samples_leaf=2,
                                n_jobs=-1,
                                oob_score=True)
    rf.fit(test_x, test_y)
    joblib.dump(rf, 'model/rflidar3.model')
    print('OK!')

def randomtree_lidaralldata3():
    alltest_y = np.array([])
    alltest_y.shape = 0
    K.clear_session()
    model = lidar_model()
    model.load_weights('model/lidar_model3.h5')
    print(model.summary())
    model = Model(inputs=model.input, outputs=model.get_layer('dense_2').output)
    print(model.summary())
    rf = joblib.load('model/rflidar3.model')
    for m in range(8, 2412):
        print str(m)
        lidar_batch_patch = np.zeros((8344, patch_size_LID, patch_size_LID, lidar_channel), dtype=np.uint8)
        for n in range(8, 8352):
            bias = int(patch_size_LID / 2)
            upper_y = m - bias
            left_x = n - bias
            lidar_batch_patch[n - 8] = lidar_change[upper_y:upper_y + patch_size_LID, left_x:left_x + patch_size_LID]
        pre_labels = model.predict(lidar_batch_patch)
        pre_labels = pre_labels.reshape((8344, 512))
        predicted_yone = rf.predict(pre_labels)
        alltest_y = np.concatenate((alltest_y, predicted_yone), 0)
    alltest_y = alltest_y.reshape((2404, 8344))
    np.save("Result/alltest_y3.npy", alltest_y)
    K.clear_session()


def randomtree_lidar17():
    test_x = np.array([])
    test_x.shape = 0, 512
    test_y = np.array([])
    test_y.shape = 0,
    K.clear_session()
    model = lidar_model()
    model.load_weights('model/lidar_model17.h5')
    print(model.summary())
    model = Model(inputs=model.input, outputs=model.get_layer('dense_2').output)
    print(model.summary())
    for j in range(int(label_location17[0].shape[0]//1000)):
        print str(j)
        lidar_batch_label = np.zeros((batch_size), dtype=int)
        lidar_batch_patch = np.zeros((batch_size, patch_size_LID, patch_size_LID, lidar_channel),
                                   dtype=np.uint8)
        for i in range(1000):
            num = j * 1000 + i
            X = label_location17[0][num]
            Y = label_location17[1][num]
            lab = train_sets3[X, Y]
            lidar_batch_label[i] = lab
            y = label_location17[0][num]
            x = label_location17[1][num]
            bias = int(patch_size_LID / 2)
            upper_y = y - bias
            left_x = x - bias
            upper_y, left_x = boundary_judge(upper_y, left_x, patch_size_LID, lidar.shape[0], lidar.shape[1])
            lidar_batch_patch[i] = lidar[upper_y:upper_y + patch_size_LID, left_x:left_x + patch_size_LID]
        test_y = np.concatenate((test_y, lidar_batch_label), 0)
        pre_labels = model.predict(lidar_batch_patch)
        pre_labels = pre_labels.reshape((test_size, 512))
        test_x = np.concatenate((test_x, pre_labels), 0)
    K.clear_session()
    rf = RandomForestClassifier(criterion="entropy", max_features="sqrt", n_estimators=400, min_samples_leaf=2,
                                n_jobs=-1,
                                oob_score=True)
    rf.fit(test_x, test_y)
    joblib.dump(rf, 'model/rflidar17.model')
    print('OK!')

def randomtree_lidaralldata17():
    alltest_y = np.array([])
    alltest_y.shape = 0
    K.clear_session()
    model = lidar_model()
    model.load_weights('model/lidar_model17.h5')
    print(model.summary())
    model = Model(inputs=model.input, outputs=model.get_layer('dense_2').output)
    print(model.summary())
    rf = joblib.load('model/rflidar17.model')
    for m in range(8, 2412):
        print str(m)
        lidar_batch_patch = np.zeros((8344, patch_size_LID, patch_size_LID, lidar_channel), dtype=np.uint8)
        for n in range(8, 8352):
            bias = int(patch_size_LID / 2)
            upper_y = m - bias
            left_x = n - bias
            lidar_batch_patch[n - 8] = lidar_change[upper_y:upper_y + patch_size_LID, left_x:left_x + patch_size_LID]
        pre_labels = model.predict(lidar_batch_patch)
        pre_labels = pre_labels.reshape((8344, 512))
        predicted_yone = rf.predict(pre_labels)
        alltest_y = np.concatenate((alltest_y, predicted_yone), 0)
    alltest_y = alltest_y.reshape((2404, 8344))
    np.save("Result/alltest_y17.npy", alltest_y)
    K.clear_session()


if __name__ == '__main__':
    randomtree_lidar3()
    randomtree_lidaralldata3()
    randomtree_lidar17()
    randomtree_lidaralldata17()
