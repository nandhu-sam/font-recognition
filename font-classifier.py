#!/usr/bin/python3

import pathlib as path
import random
import string

# import tensorflow as tf
import matplotlib.pyplot as plt

import keras.layers as layers
import keras.models as models
import keras.saving as saving
import keras.preprocessing as preprocessing


def fontClassifier(glyph: str, img_size, train_ds, validation_ds):
    EPOCHS = 150

    font_clf_model = models.Sequential(
        [
            layers.Input(img_size + (1,)),
            layers.Rescaling(1.0 / 255),

            layers.RandomZoom(0.2),
            layers.RandomTranslation(0.2, 0.2),
            layers.RandomFlip(mode='horizontal'),

            layers.Conv2D(32, (3, 3), activation='relu'),  # Added for (64, 64) size
            layers.MaxPooling2D((2, 2)),

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

    history = font_clf_model.fit(train_ds, epochs=EPOCHS, validation_data=validation_ds)

    return history, font_clf_model


def main(img_shape=(64, 64)):
    all_glyphs_classes = [c + '-U+' + hex(ord(c))[2:] for c in string.ascii_letters + string.digits]
    all_font_classes = [str(n).zfill(2) for n in range(10)]

    ds_path = path.Path('dataset')
    save_dir = path.Path('font-clf-models')
    save_dir.mkdir(parents=True, exist_ok=True)

    for g in all_glyphs_classes:
        train_ds, validation_ds = preprocessing.image_dataset_from_directory(
            ds_path / 'train' / g,
            label_mode='categorical',
            class_names=all_font_classes,
            validation_split=(0.1 / 0.7),
            subset='both',
            shuffle=True,
            seed=random.SystemRandom().randint(0, 2 ** 32 - 1),
            color_mode='grayscale',
            image_size=img_shape,
        )

        test_ds = preprocessing.image_dataset_from_directory(
            ds_path / 'test' / g,
            label_mode='categorical',
            class_names=all_font_classes,
            shuffle=True,
            seed=random.SystemRandom().randint(0, 2 ** 32 - 1),
            color_mode='grayscale',
            image_size=img_shape
        )
        hist, model = fontClassifier(g, img_shape, train_ds, validation_ds)
        saving.save_model(model, str(save_dir / g), save_format='tf')

        loss, accuracy = model.evaluate(test_ds)

        plt.title("Font Classifier (" + g + ")")
        plt.plot(hist.epoch, hist.history['accuracy'], label='Train Accuracy')
        plt.plot(hist.epoch, hist.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(title='Test Accuracy: '+str(accuracy))
        plt.savefig(save_dir / g / ('result-accuracy-'+g+'.svg'))
        plt.clf()

        plt.title("Font Classifier (" + g + ")")
        plt.plot(hist.epoch, hist.history['loss'], label='Train Loss')
        plt.plot(hist.epoch, hist.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(title='Test Loss: '+str(loss))
        plt.savefig(save_dir / g / ('result-loss'+g+'.svg'))
        plt.clf()

        with open(save_dir / g / 'test-set-results.txt', 'w') as outfile:
            print("Test loss:", loss, file=outfile)
            print("Test accuracy:", accuracy, file=outfile)


if __name__ == '__main__':
    main()
