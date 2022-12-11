from typing import Tuple, Type
from numpy.typing import NDArray

import numpy as np
from tensorflow import keras
from keras import layers, Sequential

from config import *


def get_data() -> Tuple[Tuple[NDArray[np.int_], NDArray[np.int_]], Tuple[NDArray[np.int_], NDArray[np.int_]]]:
    train_data: NDArray[np.int_]
    train_labels: NDArray[np.int_]
    test_data: NDArray[np.int_]
    test_labels: NDArray[np.int_]

    # 60 000 train images, 10 000 test images
    (train_data, train_labels), (test_data, test_labels) = keras.datasets.mnist.load_data()

    train_data = np.expand_dims(train_data, -1)
    test_data = np.expand_dims(test_data, -1)

    train_labels = keras.utils.to_categorical(train_labels, OUTPUT_SIZE)
    test_labels = keras.utils.to_categorical(test_labels, OUTPUT_SIZE)

    return (train_data, train_labels), (test_data, test_labels)


def create_convolutional_network() -> Sequential:
    convolutional_network: Sequential = keras.Sequential(
        [
            keras.Input(shape=(INPUT_DIMENSION, INPUT_DIMENSION, 1)),
            layers.Conv2D(FILTER_COUNT, kernel_size=(KERNEL_DIMENSION, KERNEL_DIMENSION), activation="relu", strides=(1, 1)),
            # layers.BatchNormalization(),
            # layers.Activation(activation=keras.activations.relu),
            layers.MaxPool2D(pool_size=(POOl_DIMENSION, POOl_DIMENSION), strides=(2, 2)),
            layers.Flatten(),
            layers.Dense(PERCEPTRON_NEURON_COUNT, activation="tanh"),
            layers.Dense(OUTPUT_SIZE, activation="softmax"),
        ]
    )

    return convolutional_network


def create_mlp_network() -> Sequential:
    return keras.Sequential(
        [
            layers.Flatten(),
            keras.Input(shape=(INPUT_DIMENSION * INPUT_DIMENSION, 1)),
            layers.Dense(40, activation="tanh"),
            layers.Dense(60, activation="tanh"),
            layers.Dense(80, activation="tanh"),
            layers.Dense(OUTPUT_SIZE, activation="softmax"),
        ]
    )


def main():
    file_id = '14'
    conv_path = 'data/' + file_id + '-conv.csv'
    mlp_path = 'data/' + file_id + '-mlp.csv'

    ((train_data, train_labels), (test_data, test_labels)) = get_data()

    conv_logger = keras.callbacks.CSVLogger(conv_path)
    convolutional_network: Sequential = create_convolutional_network()
    convolutional_network.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    convolutional_network.fit(train_data, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1,
                              verbose=1, callbacks=conv_logger)
    # convolutional_score = convolutional_network.evaluate(test_data, test_labels, verbose=0, callbacks=conv_logger)

    # mlp_logger = keras.callbacks.CSVLogger(mlp_path)
    # mlp_network: Sequential = create_mlp_network()
    # mlp_network.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    # mlp_network.fit(train_data, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1,
    #                           verbose=1, callbacks=mlp_logger)


if __name__ == '__main__':
    main()
