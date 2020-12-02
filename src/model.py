#!/usr/bin/python

from tensorflow import keras

def create(learning_rate=1e-3):
    model = keras.Sequential([
        keras.layers.Conv2D(
            filters=32,
            kernel_size=5,
            input_shape=(120, 300, 3),
            activation='elu',
            strides=(2, 2),
            padding='same',
        ),
        keras.layers.Conv2D(
            filters=64,
            kernel_size=5,
            activation='elu',
            strides=(2, 2),
            padding='same',
        ),
        keras.layers.Conv2D(
            filters=128,
            kernel_size=3,
            activation='elu',
            strides=(1, 1),
            padding='same',
        ),
        keras.layers.Conv2D(
            filters=128,
            kernel_size=3,
            activation='relu',
            strides=(1, 1),
            padding='same',
        ),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='elu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation='elu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1)
    ])


    adam = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam, loss='mse')

    print('created model...')
    print(model.summary())

    return model

