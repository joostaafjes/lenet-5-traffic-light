import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import time

class_model = load_model('model'+ '/model.h5')


def predict(image_name):
    img = load_img(image_name, False, target_size=(32, 32), color_mode='rgb')
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = class_model.predict_classes(x)
    prob = class_model.predict_proba(x)
    print(preds, prob)


start = time.time()
predict('images-resized/086-0-yellow.png.jpeg')
elapsed = time.time() - start
print('elapsed time:', elapsed, ' ms')

predict('images-resized/067-2-green.png.jpeg')
predict('images-resized/019-0-red.png.jpeg')
