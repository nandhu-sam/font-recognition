#!/usr/bin/python3

import pathlib as path
import random
import string

# import tensorflow as tf

import keras.layers as layers
import keras.models as models
import keras.saving as saving
import keras.preprocessing as preprocessing

all_glyphs_classes = [c + '-U+' + hex(ord(c))[2:] for c in string.ascii_letters + string.digits]
all_font_classes = [str(n).zfill(2) for n in range(10)]

IMG_SIZE = (32, 32)


def fontClassifier(glyph: str, ds_root: path.Path):
    EPOCHS = 150

    train_ds = preprocessing.image_dataset_from_directory(
        ds_root / 'train' / glyph,
        label_mode='categorical',
        class_names=all_font_classes,
        shuffle=True,
        seed=random.SystemRandom().randint(0, 2 ** 32 - 1),
        color_mode='grayscale',
        image_size=IMG_SIZE,
    )

    test_ds = preprocessing.image_dataset_from_directory(
        ds_root / 'test' / glyph,
        label_mode='categorical',
        class_names=all_font_classes,
        shuffle=True,
        seed=random.SystemRandom().randint(0, 2 ** 32 - 1),
        color_mode='grayscale',
        image_size=IMG_SIZE
    )

    font_clf_model = models.Sequential(
        [
            layers.Input(IMG_SIZE + (1,)),
            layers.Rescaling(1.0 / 255),

            layers.RandomZoom(0.2),
            layers.RandomFlip(mode='horizontal'),

            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),

            layers.Dense(units=128, activation='relu'),
            layers.Dense(units=10, activation='softmax')
        ],
        name='font-train-pipe-' + glyph.replace('+', '_')
    )
    font_clf_model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

    history = font_clf_model.fit(train_ds, epochs=EPOCHS, validation_data=test_ds)

    return history, font_clf_model


if __name__ == '__main__':
    ds_path = path.Path('dataset')
    save_dir = path.Path('font-clf-models')
    save_dir.mkdir(parents=True, exist_ok=True)

    for g in all_glyphs_classes:
        hist, model = fontClassifier(g, ds_path)
        saving.save_model(model, str(save_dir / g), save_format='tf')
