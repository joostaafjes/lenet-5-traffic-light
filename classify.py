import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import time
import os

class_model = load_model('model'+ '/model.h5')

traffic_light_colors = ['red', 'yellow', 'green']

def predict(image_name):
    img = load_img(image_name, False, target_size=(32, 32))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = class_model.predict_classes(x)
    prob = class_model.predict_proba(x)

    return preds[0], prob[0]


for root, dirs, files in os.walk("images_resized", topdown=False):
    for filename in files:
        start = time.time()
        pred, prob = predict(root + '/' + filename)
        elapsed = time.time() - start
        print(filename, ':', traffic_light_colors[pred], '- prodb:', prob, '-elapsed time:', elapsed, ' s')

