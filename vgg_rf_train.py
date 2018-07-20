# coding: utf-8
import keras.backend as K
import time
from data_input import lidar_train_data3, lidar_train_data17
from constant import *
from classify_model import lidar_model
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

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

def lidartrain3():
    total_number = 25000
    save_number = 1000
    K.clear_session()
    model = lidar_model()
    t0 = time.time()
    for i in range(total_number):
        print "i " + str(i)
        lidar, label = lidar_train_data3(batch_size=batch_size_LID, patch_size=patch_size_LID)
        model.fit(lidar, label, epochs=5, batch_size=batch_size_LID)
        if i % save_number == 0:
            print "save model......"
            model.save_weights('model/lidar_model3.h5')
    K.clear_session()
    t1 = time.time()
    print "spend total time:" + str(round(t1 - t0, 2)) + "s"
    K.clear_session()

def lidartrain17():
    total_number = 25000
    save_number = 1000
    K.clear_session()
    model = lidar_model()
    t0 = time.time()
    for i in range(total_number):
        print "i " + str(i)
        lidar, label = lidar_train_data17(batch_size=batch_size_LID, patch_size=patch_size_LID)
        model.fit(lidar, label, epochs=5, batch_size=batch_size_LID)
        if i % save_number == 0:
            print "save model......"
            model.save_weights('model/lidar_model17.h5')
    K.clear_session()
    t1 = time.time()
    print "spend total time:" + str(round(t1 - t0, 2)) + "s"
    K.clear_session()

if __name__ == '__main__':
    lidartrain3()
    lidartrain17()