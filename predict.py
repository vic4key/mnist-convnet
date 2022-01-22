import numpy as np
from tensorflow import keras, train
from tensorflow.keras import layers

from data import *

# Model loading

def load_hdf5_model():
	model = keras.models.load_model(hdf5_model)
	model.summary()
	return model

def load_ckpt_model():
	from model import create_model
	model = create_model()
	model.summary()
	ckpt_model_latest = train.latest_checkpoint(ckpt_model.split("/")[0])
	model.load_weights(ckpt_model_latest).expect_partial()
	# train.Checkpoint(model).restore(ckpt_model_latest).expect_partial()
	return model

# Model testing

model = load_hdf5_model()
# model = load_ckpt_model()

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

result = model.predict(x_test).shape
print(result)