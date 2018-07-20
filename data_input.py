import numpy as np 
import cv2
import random
from constant import *
import skimage.io



def dense_to_one_hot(labels_dense, num_classes=20):
	num_labels = labels_dense.shape[0]
	index_offset = np.arange(num_labels) * num_classes
	labels_one_hot = np.zeros((num_labels, num_classes))
	labels_one_hot.flat[index_offset + labels_dense.ravel()-1] = 1
	return labels_one_hot

def one_hot_to_dense(labels_one_hot, num_classes=20):
	num_labels = labels_one_hot.shape[0]
	labels_dense = np.zeros(num_labels,dtype=int)
	for i in xrange(num_labels):
		for j in xrange(num_classes):
			if labels_one_hot[i][j] == 1:
				labels_dense[i] = j+1 
	return labels_dense

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

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

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
########ground truth####################################################
train_setstow = np.load("dara/train_labtwo.npy")
train_sets3 = np.load("data/train_lab3.npy")
train_sets17 = np.load("data/train_lab17.npy")
##########lidar dsm data input #############################################
lidar = skimage.io.imread("data/lidar_train_4dsem.tif")
lidar = lidar * 255
lidartest = skimage.io.imread("data/lidar_test_4dsem.tif")
lidartest = lidartest * 255

def lidar_train_data2(batch_size=64, patch_size=16):
	lidar_batch_patch = np.zeros((batch_size,patch_size,patch_size,lidar_channel),dtype=np.uint8)
	lidar_batch_label = np.zeros((batch_size,1),dtype=int)
	for i in range(batch_size):
		label = int(random.randint(1,20))
		label_location = np.where(train_setstow == label)
		index = int(random.randint(0,int(label_location[0].size-1)))
		y = label_location[0][index]
		x = label_location[1][index]
		bias = int(patch_size/2)
		upper_y = y-bias
		left_x = x-bias
		upper_y,left_x = boundary_judge(upper_y,left_x,patch_size,lidar.shape[0],lidar.shape[1])
		lidar_batch_patch[i] = lidar[upper_y:upper_y+patch_size,left_x:left_x+patch_size]
		lidar_batch_label[i] = label
	lidar_batch_label = dense_to_one_hot(lidar_batch_label,num_classes=20)
	return lidar_batch_patch,lidar_batch_label

def lidar_train_data3(batch_size=64, patch_size=16):
	lidar_batch_patch = np.zeros((batch_size,patch_size,patch_size,lidar_channel),dtype=np.uint8)
	lidar_batch_label = np.zeros((batch_size,1),dtype=int)
	for i in range(batch_size):
		label = int(random.randint(1,20))
		label_location = np.where(train_sets3 == label)
		index = int(random.randint(0,int(label_location[0].size-1))) 
		y = label_location[0][index]
		x = label_location[1][index]
		bias = int(patch_size/2)
		upper_y = y-bias
		left_x = x-bias
		upper_y,left_x = boundary_judge(upper_y,left_x,patch_size,lidar.shape[0],lidar.shape[1])
		lidar_batch_patch[i] = lidar[upper_y:upper_y+patch_size,left_x:left_x+patch_size]
		lidar_batch_label[i] = label
	lidar_batch_label = dense_to_one_hot(lidar_batch_label,num_classes=20)
	return lidar_batch_patch,lidar_batch_label

def lidar_train_data17(batch_size=64, patch_size=16):
	lidar_batch_patch = np.zeros((batch_size,patch_size,patch_size,lidar_channel),dtype=np.uint8)
	lidar_batch_label = np.zeros((batch_size,1),dtype=int)
	for i in range(batch_size):
		label = int(random.randint(1,20))
		label_location = np.where(train_sets17 == label)
		index = int(random.randint(0,int(label_location[0].size-1)))
		y = label_location[0][index]
		x = label_location[1][index]
		bias = int(patch_size/2)
		upper_y = y-bias
		left_x = x-bias
		upper_y,left_x = boundary_judge(upper_y,left_x,patch_size,lidar.shape[0],lidar.shape[1])
		lidar_batch_patch[i] = lidar[upper_y:upper_y+patch_size,left_x:left_x+patch_size]
		lidar_batch_label[i] = label
	lidar_batch_label = dense_to_one_hot(lidar_batch_label,num_classes=20)
	return lidar_batch_patch,lidar_batch_label

