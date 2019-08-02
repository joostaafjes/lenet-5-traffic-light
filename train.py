# Load the data

import gzip
import re
import os

import numpy as np
import pandas as pd
from time import time
from PIL import Image, ImageOps

from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import keras.layers as layers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.backend import tf as ktf

from pathlib2 import Path  # python 2 backport


from os import listdir
from os.path import isfile, join

traffic_light_colors = ['red', 'yellow', 'green', 'unknown']
traffic_light_categories = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]

RESIZED_DIR = 'images_resized/'

def read_images(images_path):
    labels = []
    features = []
    for root, dirs, files in os.walk(images_path, topdown=False):
        path = os.path.dirname(RESIZED_DIR + root + '/')
        Path(path).mkdir(exist_ok=True)
        for filename in files:
            added = False
            for index, color in enumerate(traffic_light_colors):
                if filename.find(color) != -1:
                    labels.append(traffic_light_categories[index])
                    img = load_img(root + '/' + filename)  # this is a PIL image
                    img = crop_image(img)
                    img.save('{}{}{}.jpeg'.format(RESIZED_DIR, root + '/', remove_ext(filename)), 'JPEG')
                    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
                    features.append(x)

                    added = True
                    break
            if not added:
                print('Error invalid filename:', filename)

    return np.array(features), np.array(labels)

def remove_ext(filename):
    return re.sub(r'\.png|\.jpeg|\.jpg', '', filename, re.IGNORECASE)

def crop_image(img):
    img.thumbnail((32, 32), Image.ANTIALIAS)
    width, height = img.size
    delta_w = 32 - width
    delta_h = 32 - height
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    img = ImageOps.expand(img, padding, fill=0)  # fill with black dots
    return img


features, labels = read_images('images')

train = {}
test = {}

train['features'], test['features'], train['labels'], test['labels'] = train_test_split(features, labels, test_size=0.05)

# explore the data

print('# of training images:', train['features'].shape[0])
print('# of test images:', test['features'].shape[0])

# plot training data
train_labels_count = np.unique(train['labels'], return_counts=True)
dataframe_train_labels = pd.DataFrame({'Label':train_labels_count[0], 'Count':train_labels_count[1]})
print(dataframe_train_labels)

# Split training data into training and validation
validation = {}
train['features'], validation['features'], train['labels'], validation['labels'] = train_test_split(train['features'], train['labels'], test_size=0.2, random_state=0)

print('# of training images:', train['features'].shape[0])
print('# of validation images:', validation['features'].shape[0])

print(features.shape)
print(labels.shape)

print("Updated Image Shape: {}".format(train['features'][0].shape))

model = Sequential()

model.add(layers.Lambda(lambda x: x/255.0 - 0.5, input_shape=(32,32,3))) # added

model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,3)))
model.add(layers.AveragePooling2D())

model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(layers.AveragePooling2D())

model.add(layers.Flatten())

model.add(layers.Dense(units=120, activation='relu'))

model.add(layers.Dense(units=84, activation='relu'))

model.add(layers.Dense(units=4, activation = 'softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit(train['features'], train['labels'], epochs=52, validation_split=0.3, shuffle=True, callbacks=[tensorboard])

# EPOCHS = 10
# BATCH_SIZE = 128

# X_train, y_train = train['features'], train['labels']
# X_validation, y_validation = validation['features'], validation['labels']

# train_generator = ImageDataGenerator().flow(X_train, y_train, batch_size=BATCH_SIZE)
# validation_generator = ImageDataGenerator().flow(X_validation, y_validation, batch_size=BATCH_SIZE)

# print('# of training images:', train['features'].shape[0])
# print('# of validation images:', validation['features'].shape[0])

# steps_per_epoch = X_train.shape[0]//BATCH_SIZE
# validation_steps = X_validation.shape[0]//BATCH_SIZE

# tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
# model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
#                     validation_data=validation_generator, validation_steps=validation_steps,
#                     shuffle=True, callbacks=[tensorboard])

score = model.evaluate(test['features'], test['labels'])
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('model/model.h5')
