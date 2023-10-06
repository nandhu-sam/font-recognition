#!/usr/bin/env python3

import pathlib as path
import random
import string

import numpy as np
import matplotlib.pyplot as plt
import seaborn

import tensorflow as tf
import keras.layers as layers
import keras.models as models
import keras.saving as saving
import keras.preprocessing as preprocessing


def fontClassifierSaveHistory(history, save_dir, glyph, test_loss, test_accuracy):
    plt.title("Font Classifier (" + glyph + ")")
    plt.plot(history.epoch, history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history.epoch, history.history['accuracy'], label='Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(title='Test Accuracy: ' + str(test_accuracy))
    plt.savefig(save_dir / glyph / f'result-accuracy-{glyph}.svg')
    plt.clf()

    plt.title(f"Font Classifier ({glyph})")
    plt.plot(history.epoch, history.history['val_loss'], label='Validation Loss')
    plt.plot(history.epoch, history.history['loss'], label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(title='Test Loss: ' + str(test_loss))
    plt.savefig(save_dir / glyph / f'result-loss-{glyph}.svg')
    plt.clf()

    with open(save_dir / glyph / 'test-set-results.txt', 'w') as outfile:
        print("Test loss:", test_loss, file=outfile)
        print("Test accuracy:", test_accuracy, file=outfile)

    with open(save_dir / glyph / f'test-set-loss-{glyph}.txt', 'w') as outfile:
        print(test_loss, file=outfile)

    with open(save_dir / glyph / f'test-set-accuracy-{glyph}.txt', 'w') as outfile:
        print(test_accuracy, file=outfile)


def fontClassifier(glyph: str, img_size, train_ds, validation_ds):
    EPOCHS = 150

    font_clf_model = models.Sequential(
        [
            layers.Input(img_size + (1,)),
            layers.Rescaling(1.0 / 255, dtype=tf.dtypes.float64),

            layers.RandomZoom(0.2),
            layers.RandomTranslation(0.05, 0.05),

            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),

            layers.Dense(units=128, activation='relu'),
            layers.Dense(units=10, activation='softmax')
        ],
        name='font-train-pipe-' + glyph.replace('+', '_')
    )
    font_clf_model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    font_clf_model.summary()

    history = font_clf_model.fit(train_ds, epochs=EPOCHS, validation_data=validation_ds)

    return history, font_clf_model


def fontClassifierConfusionMatrix(model, glyph, test_ds, test_accuracy, save_dir):
    classes = [str(n).zfill(2) for n in range(10)]
    xs, ys = zip(*test_ds.unbatch().as_numpy_iterator())
    xs = np.array(xs)
    ys = np.argmax(np.array(ys), axis=-1)

    preds = np.argmax(model.predict(xs), axis=-1)
    conf_mat = tf.math.confusion_matrix(ys, preds, len(classes)).numpy()
    plt.figure(figsize=(12, 9))
    seaborn.heatmap(conf_mat, annot=True,
                    xticklabels=classes, yticklabels=classes,
                    square=True, cmap='rocket', label='')

    plt.title(f"Font Classifier ({glyph}) Confusion Matrix", fontsize=20, y=1.1)
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.xticks(rotation=0)
    plt.legend(title=f'Test Accuracy: {test_accuracy}', labels=[''], loc=(0.0, 1.0))
    plt.savefig(save_dir / f'confusionmatrix-{glyph}.svg')
    plt.clf()


def trainFontClassifierModel(glyph, ds_path, glyph_classes, font_classes, img_shape, save_dir):
    train_ds, validation_ds = preprocessing.image_dataset_from_directory(
        ds_path / 'train' / glyph,
        label_mode='categorical',
        class_names=font_classes,
        validation_split=(0.1 / 0.7),
        subset='both',
        shuffle=True,
        seed=random.SystemRandom().randint(0, 2 ** 32 - 1),
        color_mode='grayscale',
        image_size=img_shape,
    )

    test_ds = preprocessing.image_dataset_from_directory(
        ds_path / 'test' / glyph,
        label_mode='categorical',
        class_names=font_classes,
        shuffle=True,
        seed=random.SystemRandom().randint(0, 2 ** 32 - 1),
        color_mode='grayscale',
        image_size=img_shape
    )
    hist, model = fontClassifier(glyph, img_shape, train_ds, validation_ds)
    saving.save_model(model, str(save_dir / glyph), save_format='tf')

    loss, accuracy = model.evaluate(test_ds)
    fontClassifierSaveHistory(hist, save_dir, glyph, loss, accuracy)
    fontClassifierConfusionMatrix(model, glyph, test_ds, accuracy, save_dir / glyph)


def main(img_shape=(64, 64)):
    all_glyphs_classes = [c + '-U+' + hex(ord(c))[2:] for c in string.ascii_letters + string.digits]
    all_font_classes = [str(n).zfill(2) for n in range(10)]

    ds_path = path.Path('dataset')
    save_dir = path.Path('font-clf-models')
    save_dir.mkdir(parents=True, exist_ok=True)

    for g in all_glyphs_classes:
        trainFontClassifierModel(g, ds_path, all_glyphs_classes, all_font_classes, img_shape, save_dir)


if __name__ == '__main__':
    main()
