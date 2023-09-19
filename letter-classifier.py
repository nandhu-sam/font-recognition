#!/usr/bin/python3

import pathlib as path
import random
import string

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import keras.models as models
import keras.layers as layers
import keras.preprocessing as preprocessing

all_glyphs_classes = [c + '-U+' + hex(ord(c))[2:] for c in string.ascii_letters + string.digits]
IMG_SIZE = (32, 32)

train_ds, validation_ds = preprocessing.image_dataset_from_directory(
    path.Path('dataset') / 'train',
    label_mode='categorical',
    class_names=all_glyphs_classes,
    validation_split=(0.1 / 0.7),
    subset='both',
    shuffle=True,
    seed=random.SystemRandom().randint(0, 2 ** 32 - 1),
    color_mode='grayscale',
    image_size=IMG_SIZE,
)

test_ds = preprocessing.image_dataset_from_directory(
    path.Path('dataset') / 'test',
    label_mode='categorical',
    class_names=all_glyphs_classes,
    shuffle=True,
    seed=random.SystemRandom().randint(0, 2 ** 32 - 1),
    color_mode='grayscale',
    image_size=IMG_SIZE
)

rescale = models.Sequential([layers.Input((32, 32, 1)), layers.Rescaling(1.0 / 255)])

model = models.Sequential(
    [
        layers.Input(IMG_SIZE+(1,)),
        layers.Rescaling(1.0 / 255),

        layers.RandomZoom(0.2),
        layers.RandomFlip(mode='horizontal'),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(units=128, activation='relu'),
        layers.Dense(units=len(all_glyphs_classes), activation='softmax')
    ],
    name='letter-train-pipe'
)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

epochs = 150

history = model.fit(train_ds,
                    epochs=epochs,
                    validation_data=validation_ds)


measures = np.c_[(history.history[k] for k in history.history.keys())]
epoch_count = range(len(measures[0]))

model_save_path = path.Path('letter-clf-model')
model.save(model_save_path)

plt.plot(epoch_count, measures)
plt.savefig(str(model_save_path / 'result.png'))

