import os
import configparser
import numpy as np
import tensorflow as tf
import importlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.callbacks import ModelCheckpoint
from util.save_to_csv import save_result_to_csv


#from tensorflow.keras.optimizers import Adam

class Compiler():

    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.config_path = config_path
        self.storage_dir = self.config['DATA'].get('TrainingInputDirectory')
        self.output_dir = self.config['DATA'].get('TrainingOutputDirectory')
        self.prediction_output_dir = self.config['PREDICTION'].get('PredictionOutputDirectory')
        self.batch_size = self.config['TRAINING'].getint('BatchSize')
        self.model_name = self.config['TRAINING'].get('ModelName')
        self.epochs = self.config['TRAINING'].getint('Epochs')
        self.patience = self.config['TRAINING'].getint('Patience')
        self.we_are_training = self.config['TRAINING'].get('Training')
        self.load_model = self.config['PREDICTION'].get('LoadModel')
        self.saved_model_path = self.config['PREDICTION'].get('ModelWeightPath')
        self.loaded_weights = False
        self._load_data()
        self.load_or_build_model()

    def _load_data(self):
        if self.we_are_training == 'Y':
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                width_shift_range=0.2,
                height_shift_range=0.2,
                fill_mode='nearest'
            )
            valid_datagen = ImageDataGenerator(rescale=1./255)

            self.training_set = train_datagen.flow_from_directory(
                self.storage_dir+'/train',
                target_size=(256, 256),
                batch_size=self.batch_size,
                class_mode='categorical'
            )
            self.valid_set = valid_datagen.flow_from_directory(
                self.storage_dir+'/valid',
                target_size=(256, 256),
                batch_size=self.batch_size,
                class_mode='categorical'
            )
            self.train_num = self.training_set.samples
            self.valid_num = self.valid_set.samples
            self.class_dict = self.training_set.class_indices
        else:
            self._load_class_dict()

    def _load_class_dict(self):
        class_dict_df = pd.read_csv(f'{self.output_dir}/class_dict.csv')
        class_dict = class_dict_df.to_dict(orient='list')
        self.class_dict = {k: v[0] for (k, v) in class_dict.items()}

    def _build_model(self):
        tmp = self.model_name.split('.')
        module = '.'.join(tmp[:2])
        class_name = tmp[-1]
        model = importlib.import_module(
                module,
                package=None
        )
        class_ = getattr(model, class_name)
        self.model = class_()._make_model()

    def compile(self):
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def fit(self):
        weightpath = f"{self.output_dir }/best_checkpoint_weight.hdf5"
        checkpoint = ModelCheckpoint(weightpath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
        patience = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.patience)
        callbacks_list = [checkpoint, patience]
        self.history = self.model.fit(self.training_set,
                         steps_per_epoch=self.train_num//self.batch_size,
                         validation_data=self.valid_set,
                         epochs=self.epochs,
                         validation_steps=self.valid_num//self.batch_size,
                         callbacks=callbacks_list
        )
        filepath=f"{self.output_dir }/model_path.hdf5"
        self.model.save(filepath)
        return self.history

    def plot_history(self):
        #plotting training values
        sns.set()

        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(1, len(loss) + 1)

        #accuracy plot
        plt.plot(epochs, acc, color='green', label='Training Accuracy')
        plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()

        plt.figure()
        #loss plot
        plt.plot(epochs, loss, color='pink', label='Training Loss')
        plt.plot(epochs, val_loss, color='red', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.savefig(f'{self.output_dir}/history.png')

    def load_or_build_model(self):
        self._build_model()
        if self.load_model == 'Y':
            self.model.load_weights(self.saved_model_path)
            self.loaded_weights = True
            

    def predict_one_shot(self, image_path):
        new_img = load_img(image_path, target_size=(256, 256))
        img = img_to_array(new_img)
        img = np.expand_dims(img, axis=0)
        img = img/255
        prediction = self.model.predict(img)
        d = prediction.flatten()
        j = d.max()
        li = list(self.class_dict.keys())
        for index,item in enumerate(d):
            if item == j:
                class_name = li[index]
        return class_name, j

    def predict(self, image_path):
        class_name, confidence = self.predict_one_shot(image_path)
        record = {}
        record['filename'] = image_path
        record['time-stamp'] = os.path.split(image_path)[1].split('.')[0]
        record['confidence'] = confidence
        record['classification'] = class_name
        save_result_to_csv(self.prediction_output_dir, record)