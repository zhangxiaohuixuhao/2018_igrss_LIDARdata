from keras.models import Sequential, Model
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout,BatchNormalization, merge, MaxPooling2D,AveragePooling2D, Input, Activation,concatenate
from keras.layers.convolutional import Conv2D,SeparableConv2D
from keras.optimizers import Adam,SGD
from constant import *

def lidar_model():
    input_shape=(patch_size_LID, patch_size_LID, lidar_channel)
    input_img = Input(shape=input_shape)
     
    conv1 = Conv2D(64, (5, 5), padding="same", strides=(1, 1), activation="relu", name="conv1")(input_img)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, (5, 5), padding="same", strides=(1, 1), activation="relu", name="conv2")(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(conv1)
    
    conv2 = Conv2D(128, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv3")(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv4")(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv5")(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(conv2)
    
    conv3 = Conv2D(256, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv6")(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv7")(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv8")(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(conv3)
    
    x = Flatten()(conv3)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(drop_probability)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(drop_probability)(x)
    out = Dense(20, activation='softmax')(x)

    model = Model(inputs=input_img, outputs=[out])
    model.compile(optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True), loss='categorical_crossentropy',metrics=['accuracy'])
    return model