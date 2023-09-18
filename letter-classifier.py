#!/usr/bin/python3

import pathlib as path
import random
import string

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import math as tfmath


import keras.models as models
import keras.layers as layers
import keras.preprocessing as preprocessing

train_ds, validation_ds = preprocessing.image_dataset_from_directory(
    path.Path('dataset') / 'train',
    label_mode='categorical',
    validation_split=(0.1 / 0.7),
    subset='both',
    shuffle=True,
    seed=random.SystemRandom().randint(0, 2 ** 32 - 1),
    color_mode='grayscale',
    image_size=(32, 32)
)

test_ds = preprocessing.image_dataset_from_directory(
    path.Path('dataset') / 'test',
    label_mode='categorical',
    shuffle=True,
    seed=random.SystemRandom().randint(0, 2 ** 32 - 1),
    color_mode='grayscale',
    image_size=(32, 32)
)

rescale = models.Sequential([layers.Input((32, 32, 1)), layers.Rescaling(1.0 / 255)])

model = models.Sequential(
    [
        layers.Input((32, 32, 1)),
        layers.Rescaling(1.0 / 255),

        layers.RandomZoom(0.2),
        layers.RandomFlip(mode='horizontal'),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(units=128, activation='relu'),
        layers.Dense(units=62, activation='softmax')
    ],
    name='train-pipe'
)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(train_ds,
          epochs=int(input("epochs: ")),
          validation_data=validation_ds
          )


def labelFromVec(v):
    ls = list(sorted(string.ascii_letters + string.digits))
    return ls[list(v).index(1.0)]


def toMaxVec(v):
    l = np.array(list(v))
    return (l>=l.max()).astype(np.float64)


# print(np.concatenate([[y, model.predict(x)] for x, y in test_ds]).shape)
# plt.imshow(tfmath.confusion_matrix(np.concatenate([y for _, y in test_ds]),np.concatenate([model.predict(x) for x, _ in test_ds])))

plt.imshow(tfmath.confusion_matrix(tfmath.argmax(np.concatenate([model.predict(x) for x, y in test_ds]).T),
                        tfmath.argmax(np.concatenate([y for x, y in test_ds]).T)), origin='lower', cmap='Greys_r')
plt.show()




plt.show()
