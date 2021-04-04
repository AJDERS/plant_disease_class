import os
import configparser
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Dropout, MaxPooling2D, BatchNormalization


class ConvNet:
    def __init__(self):
        pass

    def _make_model(self):
        # Initializing the CNN
        model = Sequential()

        # Convolution Step 1 + Max Pooling
        model.add(Convolution2D(96, 11, strides = (4, 4), padding = 'valid', input_shape=(256, 256, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
        model.add(BatchNormalization())

        # Convolution Step 2 + Max Pooling
        model.add(Convolution2D(256, 11, strides = (1, 1), padding='valid', activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding='valid'))
        model.add(BatchNormalization())

        # Convolution Step 3
        model.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
        model.add(BatchNormalization())

        # Convolution Step 4
        model.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
        model.add(BatchNormalization())

        # Convolution Step 5
        model.add(Convolution2D(256, 3, strides=(1,1), padding='valid', activation = 'relu'))

        # Max Pooling
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
        model.add(BatchNormalization())

        # Flattening Step
        model.add(Flatten())

        # Full Connection Step
        model.add(Dense(units = 4096, activation = 'relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())
        model.add(Dense(units = 4096, activation = 'relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())
        model.add(Dense(units = 1000, activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(units = 38, activation = 'softmax'))
        return model
    