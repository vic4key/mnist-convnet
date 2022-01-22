import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from data import *
from model import *
from config import *

model = create_model()
model.summary()

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=ckpt_model,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model.fit(
	x_train,
	y_train,
	batch_size=batch_size,
	epochs=epochs,
	validation_split=0.1,
	callbacks=[model_checkpoint_callback])

model.save(hdf5_model)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
