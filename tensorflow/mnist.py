# This was created following along at https://keras.io/examples/vision/mnist_convnet/

# disable tensorflow messages about library loading
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Model/training parameters
num_classes = 10          # 10 digits total
input_shape = (28, 28, 1) # mnist consists of 28x28 grayscale images
batch_size = 128          # number of inputs per training batch
epochs = 15               # number of epochs to train
validation_split = 0.1    # fraction of training data to use as validation

def load_data():
    # keras is kind enough to provide the mnist dataset for us
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # scale the grayscale pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # convert images' shape from (28, 28) -> (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # convert numerical values to binary class vectors
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)

def build_model():
    m = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPool2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPool2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    m.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return m

def fit_with_gpu(model, train):
    try:
        with tf.device("/device:GPU:0"):
            # convert the training data into tensors
            (x_train, y_train) = train
            x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
            y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    except RuntimeError as e:
        print(e)

def main():
    train, (x_test, y_test) = load_data()
    model = build_model()
    fit_with_gpu(model, train)

    # evaluate model performance on the test data (unseen data)
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1] * 100, "%")

if __name__ == "__main__":
    main()