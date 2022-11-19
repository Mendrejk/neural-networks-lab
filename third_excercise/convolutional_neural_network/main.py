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


def create_network() -> Sequential:
    convolutional_network: Sequential = keras.Sequential(
        [
            keras.Input(shape=(INPUT_DIMENSION, INPUT_DIMENSION, 1)),
            layers.Conv2D(FILTER_COUNT, kernel_size=(KERNEL_DIMENSION, KERNEL_DIMENSION), activation="relu"),
            layers.MaxPool2D(pool_size=(POOl_DIMENSION, POOl_DIMENSION)),
            layers.Flatten(),
            layers.Dense(PERCEPTRON_NEURON_COUNT, activation="tanh"),
            layers.Dense(OUTPUT_SIZE, activation="softmax"),
        ]
    )

    print(convolutional_network.summary())
    return convolutional_network


def main():
    ((train_data, train_labels), (test_data, test_labels)) = get_data()
    network: Sequential = create_network()

    network.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    network.fit(train_data, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)

    score = network.evaluate(test_data, test_labels, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


if __name__ == '__main__':
    main()
