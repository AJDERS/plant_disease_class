import os
import configparser
import numpy as np
import tensorflow as tf
import importlib
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns

#from tensorflow.keras.optimizers import Adam

class Compiler():

    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.config_path = config_path
        self.storage_dir = self.config['DATA'].get('StorageDirectory')
        self.batch_size = self.config['TRAINING'].getint('BatchSize')
        self.model_name = self.config['TRAINING'].get('ModelName')
        self.epochs = self.config['TRAINING'].getint('Epochs')
        self.patience = self.config['TRAINING'].getint('Patience')
        self.loaded_weights = False
        self._load_data()
        self._build_model()

    def _load_data(self):
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
        weightpath = f"output/best_checkpoint_weight.hdf5"
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
        filepath=f"output/model_path.hdf5"
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

        plt.savefig('output/history.png')

    def plot_prediction(self):
        image_path = f"{self.storage_dir}/valid/Tomato___Septoria_leaf_spot/0a5edec2-e297-4a25-86fc-78f03772c100___JR_Sept.L.S 8468.JPG"
        prediction = self.predict(image_path)

        #ploting image with predicted class name        
        plt.figure(figsize = (4,4))
        plt.imshow(new_img)
        plt.axis('off')
        plt.title(prediction)
        plt.savefig('output/prediction.png')

    def load_weights(self, path):
        self.model.load_weights(path)
        self.loaded_weights = True

    def predict(self, image_path):
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
        return class_name