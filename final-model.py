import string
import pathlib as path
import joblib

import functools as fns

import numpy as np
import matplotlib.pyplot as plt
import seaborn
import tensorflow as tf

root_dir = path.Path('.')
proj_dir = path.Path('.')


def model_loader(letter_class):
    print("loading model", letter_class)
    return tf.keras.models.load_model(root_dir / 'font-clf-models' / letter_class)


letters = string.ascii_letters + string.digits
classes = list(map(lambda c: c + '-U+' + hex(ord(c))[2:], letters))

letter_model = tf.keras.models.load_model(root_dir / 'letter-clf-model')
# font_models = joblib.Parallel(n_jobs=6)(
#    (joblib.delayed(model_loader)(c) for c in classes))

font_models = tuple(map(model_loader, classes))

ds = tf.keras.utils.image_dataset_from_directory(root_dir / 'dataset' / 'train',
                                                 label_mode='categorical',
                                                 class_names=classes,
                                                 color_mode='grayscale',
                                                 image_size=(64, 64))

ds_font = fns.reduce(tf.data.Dataset.concatenate,
                     [tf.keras.utils.image_dataset_from_directory(root_dir / 'dataset' / 'train' / c,
                                                                  label_mode='categorical',
                                                                  class_names=[str(n).zfill(2) for n in range(10)],
                                                                  color_mode='grayscale',
                                                                  image_size=(64, 64)).unbatch() for c in classes])

imgs, labels = zip(*(ds.unbatch()))
labels = tf.argmax(tf.stack(labels), axis=1)
letter_predictions = tf.argmax(letter_model.predict(tf.stack(imgs)), axis=1)
font_names = tuple(sorted((root_dir / 'font-resources').glob('*.ttf'), key=lambda p: p.name[:2]))

conf_mat = np.zeros((10, 10))

for img, font in ds_font:
    actual_font = tf.argmax(font)
    expanded_img = tf.expand_dims(img, axis=0)
    pred = tf.argmax(letter_model.predict(expanded_img), axis=1)[0]
    s = tf.argmax(font_models[pred].predict(expanded_img), axis=1)[0]
    conf_mat[actual_font, s] += 1

seaborn.heatmap(conf_mat)
plt.show()
exit()
