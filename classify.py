import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import time
import os

class_model = load_model('model'+ '/model.h5')

traffic_light_colors = ['red', 'yellow', 'green', 'unknown']

def predict(image_name):
    img = load_img(image_name, False, target_size=(32, 32))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = class_model.predict_classes(x)
    prob = class_model.predict_proba(x)

    return preds[0], prob[0]

total = 0
errors = 0
for root, dirs, files in os.walk("images_resized/images/green", topdown=False):
    for filename in files:
        start = time.time()
        pred, prob = predict(root + '/' + filename)
        elapsed = time.time() - start
        print(filename, ':', traffic_light_colors[pred], '- prodb:', prob, '-elapsed time:', elapsed, ' s')
        total += 1
        if filename.find(traffic_light_colors[pred]) == -1:
            print('not correct')
            errors += 1
        else:
            print('correct')

print('errors: {} ({} %) out of {}'.format(errors, 100 * errors / total, total))

