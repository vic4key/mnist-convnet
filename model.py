# https://keras.io/examples/vision/mnist_convnet/
# https://www.tensorflow.org/tutorials/keras/save_and_load

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from config import *

def create_model():
	model = keras.Sequential(
	    [
	        keras.Input(shape=input_shape),
	        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
	        layers.MaxPooling2D(pool_size=(2, 2)),
	        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
	        layers.MaxPooling2D(pool_size=(2, 2)),
	        layers.Flatten(),
	        layers.Dropout(0.5),
	        layers.Dense(num_classes, activation="softmax"),
	    ]
	)

	model.compile(
		loss="categorical_crossentropy",
		optimizer="adam",
		metrics=["accuracy"])

	return model