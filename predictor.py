from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
import os
import sys
import pathlib

# Предобработка изображения: месштабирование и нормализация.
def preprocess_image(image):
  image = tf.image.decode_image(image, channels=3)
  image = tf.image.resize(image, [64, 64])
  image /= 255.0
  return image

# Загрузка и предобработка изображения.
def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)

# Предсказание от многоклассовой модели.
def make_multi_prediction(path):
  image = load_and_preprocess_image(path)
  image = np.expand_dims(image, axis=0)
  return np.argmax(model.predict(image))

model_path = sys.argv[1]
data_path = sys.argv[2]
output_path = sys.argv[3]

model = tf.keras.models.load_model(model_path)

paths = [f for f in os.listdir(data_path) if (os.path.isfile(os.path.join(data_path, f)) and f.endswith('.jpg'))]

f = open(output_path, "w+")

for path in paths:
    full_path = '%s/%s' % (data_path, path)
    try:
        f.write('%s\t%d\n' % (path, make_multi_prediction(full_path)))
    except:
        f.write('%s\t-1\n' % path)

f.close()
