import os
import configparser
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Dropout


class ConvNet(Model):
    def __init__(self, num_classes, config_path):
        super().__init__()
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.num_classes = num_classes
        self.drop_rate = self.config['TRAINING'].getfloat('DropRate')


    def _make_model(self):
        model = tf.keras.Sequential([
            Conv2D(32, 3, activation='relu', input_shape=(256, 256, 3)),
            Conv2D(64, 5, activation='relu'),
            Conv2D(128, 3, activation='relu'),
            Dropout(self.drop_rate),
            Conv2DTranspose(128, 3, activation='relu'),
            Conv2DTranspose(64, 3, activation='relu'),
            Conv2DTranspose(32, 3, activation='relu'),
            Dense(3, activation='relu'),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        return model
    