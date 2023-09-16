
import keras.models as models
import keras.layers as layers

clf = models.Sequential(
    layers.Input((32, 32)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128),
    layers.Dense(62))


