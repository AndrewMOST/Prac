import numpy as np
import os
import sys
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow import keras

import pathlib

path_to_data = sys.argv[1]
batch_size = int(sys.argv[2])
num_of_epochs = int(sys.argv[3])
use_simple_model = int(sys.argv[4])
path_to_save = sys.argv[5]

data_dir = pathlib.Path(path_to_data)
image_count = len(list(data_dir.glob('*/*.jpg')))
print('%d images were found in %s' %(image_count, path_to_data))

img_height = 64
img_width = 64

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path_to_data,
    validation_split=0.2,
    subset="training",
    seed=123,
    batch_size = batch_size,
    image_size=(img_height, img_width))

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path_to_data,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

def process(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label

train_ds_normal = train_ds.map(process)
val_ds_normal = val_ds.map(process)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds_normal = train_ds_normal.cache().prefetch(buffer_size=AUTOTUNE)
val_ds_normal = val_ds_normal.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

if use_simple_model:
    final_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
        ])
else:
    from tensorflow.keras.applications.resnet50 import ResNet50
    from tensorflow.keras.models import Model
    resnet = ResNet50(include_top=False, weights= 'imagenet', input_shape=(64,64,3))

    x = resnet.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    predictions = tf.keras.layers.Dense(num_classes, activation= 'softmax')(x)

    for layer in resnet.layers:
        layer.trainable = False

    final_model = Model(inputs = resnet.input, outputs = predictions)

    final_model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy'])

    final_model.fit(
        train_ds_normal,
        epochs = 10
        )

    for layer in resnet.layers:
        layer.trainable = True

final_model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['sparse_categorical_accuracy'])

final_model.fit(
  train_ds_normal,
  validation_data=val_ds_normal,
  epochs = num_of_epochs
)

final_model.evaluate(val_ds_normal, verbose=2)

final_model.save(path_to_save)
