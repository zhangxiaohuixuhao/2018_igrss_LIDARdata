import numpy as np
import keras.backend as K
from constant import *
from classify_model import  lidar_model
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

lidartest = skimage.io.imread("data/lidar_test_4dsem.tif")
print(np.max(lidartest))
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

def lidar_test():
    K.clear_session()
    model = lidar_model()
    model.load_weights('model/lidartwo_model.h5')
    label = np.array([])
    label.shape = 0, 2
    for m in range(8, 2412):
        print str(m)
        lidar_batch_patch = np.zeros((8344, patch_size_LID, patch_size_LID, lidar_channel), dtype=np.uint8)
        for n in range(8, 8352):
            bias = int(patch_size_LID / 2)
            upper_y = m - bias
            left_x = n - bias
            lidar_batch_patch[n - 8, ...] = lidar_change[upper_y:upper_y + patch_size_LID, left_x:left_x + patch_size_LID, :]
        pre_labels = model.predict(lidar_batch_patch)
        label = np.concatenate((label, pre_labels), 0)
    label = label.reshape(2404, 8344, 2)
    np.save('Result/labelvggtwo.npy',label)
    K.clear_session()

if __name__ == '__main__':
    lidar_test()