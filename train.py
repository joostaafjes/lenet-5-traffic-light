# Load the data

import gzip
import numpy as np
import pandas as pd
from time import time
from PIL import Image, ImageOps
import cv2

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

from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

traffic_light_colors = ['red', 'yellow', 'green']
traffic_light_categories = [[1,0,0], [0,1,0], [0,0,1]]

def read_images(images_path: str):
    files = [f for f in listdir(images_path) if isfile(join(images_path, f))]

    labels = []
    features = []
    for filename in files:
        added = False
        for index, color in enumerate(traffic_light_colors):
            if filename.find(color) != -1:
                labels.append(traffic_light_categories[index])
                # img = Image.open(images_path + '/' + filename)
                # img = cv2.imread(images_path + '/' + filename)
                # features.append(img)

                img = load_img(images_path + '/' + filename)  # this is a PIL image
                img.thumbnail((32, 32), Image.ANTIALIAS)
                delta_w = 32 - img.width
                delta_h = 32 - img.height
                padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
                img = ImageOps.expand(img, padding, fill=0)
                # img.show()
                # img.save('testimg.jpeg', 'JPEG')
                x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
                # x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
                # x = np.expand_dims(x, axis=0)
                features.append(x)

                added = True
                break
        if not added:
            print('Error invalid filename:', filename)
            exit()

    return np.array(features), np.array(labels)

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
# validation = {}
# train['features'], validation['features'], train['labels'], validation['labels'] = train_test_split(train['features'], train['labels'], test_size=0.2, random_state=0)

print('# of training images:', train['features'].shape[0])
# print('# of validation images:', validation['features'].shape[0])

print(features.shape)
print(labels.shape)

# prepare our input features
# Pad images with 0s
# train['features'] = np.pad(train['features'], ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
# validation['features'] = np.pad(validation['features'], ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
# test['features'] = np.pad(test['features'], ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

print("Updated Image Shape: {}".format(train['features'][0].shape))

model = keras.Sequential()

# model.add(layers.Input(None, None, 3))

# model.add(layers.Lambda(lambda image: ktf.image.resize_images(image, (32, 32))))

model.add(layers.Lambda(lambda x: x/255.0 - 0.5, input_shape=(32,32,3))) # added

model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,3)))
model.add(layers.AveragePooling2D())

model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(layers.AveragePooling2D())

model.add(layers.Flatten())

model.add(layers.Dense(units=120, activation='relu'))

model.add(layers.Dense(units=84, activation='relu'))

model.add(layers.Dense(units=3, activation = 'softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

model.fit(train['features'], train['labels'], epochs=52, validation_split=0.3, shuffle=True )

# EPOCHS = 10
# BATCH_SIZE = 128

# X_train, y_train = train['features'], to_categorical(train['labels'])
# X_validation, y_validation = validation['features'], to_categorical(validation['labels'])

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
