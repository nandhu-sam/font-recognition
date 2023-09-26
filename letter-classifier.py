#!/usr/bin/python3

import pathlib as path
import random
import string

import keras.layers as layers
import keras.models as models
import keras.saving as saving
import keras.preprocessing as preprocessing
import matplotlib.pyplot as plt


def main(img_shape=(64, 64)):
    all_glyphs_classes = [c + '-U+' + hex(ord(c))[2:] for c in string.ascii_letters + string.digits]

    train_set_dir = path.Path('dataset') / 'train'
    test_set_dir = path.Path('dataset') / 'test'

    if not train_set_dir.exists():
        raise FileNotFoundError(train_set_dir)

    if not test_set_dir.exists():
        raise FileNotFoundError(test_set_dir)

    train_ds, validation_ds = preprocessing.image_dataset_from_directory(
        train_set_dir,
        label_mode='categorical',
        class_names=all_glyphs_classes,
        validation_split=(0.1 / 0.7),
        subset='both',
        shuffle=True,
        seed=random.SystemRandom().randint(0, 2 ** 32 - 1),
        color_mode='grayscale',
        image_size=img_shape,
    )

    test_ds = preprocessing.image_dataset_from_directory(
        test_set_dir,
        label_mode='categorical',
        class_names=all_glyphs_classes,
        shuffle=True,
        seed=random.SystemRandom().randint(0, 2 ** 32 - 1),
        color_mode='grayscale',
        image_size=img_shape
    )

    letter_classifier_model = models.Sequential(
        [
            layers.Input(img_shape + (1,)),
            layers.Rescaling(1.0 / 255),

            layers.RandomZoom(0.2),
            layers.RandomTranslation(0.2, 0.2),
            layers.RandomFlip(mode='horizontal'),

            layers.Conv2D(32, (3, 3), activation='relu', strides=2),  # Added for (64, 64) size
            layers.MaxPooling2D((2, 2)),

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

    letter_classifier_model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    letter_classifier_model.summary()

    epochs = 150

    history = letter_classifier_model.fit(train_ds,
                                          epochs=epochs,
                                          validation_data=validation_ds)

    model_save_path = path.Path('letter-clf-model')
    saving.save_model(letter_classifier_model, str(model_save_path), save_format='tf')

    loss, accuracy = letter_classifier_model.evaluate(test_ds)

    plt.title("Letter Classifier")
    plt.plot(history.epoch, history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.epoch, history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(title='Test Accuracy: '+str(accuracy))
    plt.savefig(model_save_path/'result-accuracy.svg')
    plt.clf()

    plt.title("Letter Classifier")
    plt.plot(history.epoch, history.history['loss'], label='Train Loss')
    plt.plot(history.epoch, history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(title='Test Loss: '+str(loss))
    plt.savefig(model_save_path/'result-loss.svg')
    plt.clf()

    with open(model_save_path/'test-set-results.txt', 'w') as outfile:
        print("Test loss:", loss, file=outfile)
        print("Test accuracy:", accuracy, file=outfile)


if __name__ == '__main__':
    main()
