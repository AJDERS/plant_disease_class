import os
import configparser
import numpy as np
import tensorflow as tf
import importlib
from util import make_tfrecords
#from tensorflow.keras.optimizers import Adam

class Compiler():

    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.config_path = config_path
        self.storage_dir = self.config['DATA'].get('StorageDirectory')
        self.type = self.config['DATA'].get('Type')
        self.shuffle_buffer = self.config['TRAINING'].getint('ShuffleBuffer')
        self.batch_size = self.config['TRAINING'].getint('BatchSize')
        self.model_name = self.config['TRAINING'].get('ModelName')
        self.epochs = self.config['TRAINING'].getint('Epochs')
        self._load_data()
        self._build_model()

    def _load_data(self):
        if not make_tfrecords.make_records(self.storage_dir, self.type):
            print('Making records...')
        for mode in ['train', 'valid', 'eval']:
            if mode == 'train':
                d = np.load(f'{os.path.join(self.storage_dir, self.type, mode)}_data.npy')
                l = np.load(f'{os.path.join(self.storage_dir, self.type, mode)}_label.npy')
                self.train_data = tf.data.Dataset.from_tensor_slices((d,l))
                self.train_data = self.train_data.shuffle(self.shuffle_buffer).batch(self.batch_size)
            elif mode == 'valid':
                d = np.load(f'{os.path.join(self.storage_dir, self.type, mode)}_data.npy')
                l = np.load(f'{os.path.join(self.storage_dir, self.type, mode)}_label.npy')
                self.valid_data = tf.data.Dataset.from_tensor_slices((d,l))
                self.valid_data = self.valid_data.shuffle(self.shuffle_buffer).batch(self.batch_size)
            else:
                d = np.load(f'{os.path.join(self.storage_dir, self.type, mode)}_data.npy')
                l = np.load(f'{os.path.join(self.storage_dir, self.type, mode)}_label.npy')
                self.eval_data = tf.data.Dataset.from_tensor_slices((d,l))
                self.eval_data = self.eval_data.shuffle(self.shuffle_buffer).batch(self.batch_size)
        num_classes = len([x for x in os.listdir(os.path.join(self.storage_dir, 'train')) if self.type in x])
        self.num_classes = num_classes

    def _build_model(self):
        tmp = self.model_name.split('.')
        module = '.'.join(tmp[:2])
        class_name = tmp[-1]
        model = importlib.import_module(
                module,
                package=None
        )
        class_ = getattr(model, class_name)
        self.model = class_(self.num_classes, self.config_path)._make_model()

    def compile(self):
        self.model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    def fit(self):
        self.model.fit(self.train_data, epochs=self.epochs, validation_data=self.valid_data)

    def evaluate(self):
        self.model.evaluate(self.eval_data)
