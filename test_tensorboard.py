
import numpy as np
import tensorflow as tf
from datetime import datetime

x = np.random.uniform(-1, 0, (200000, 4))
y = np.random.randint(2, size=200000)

init = tf.keras.initializers.RandomUniform(minval=-1, maxval=0, seed=None)

inp = tf.keras.layers.Input(shape=(4,))

dense_1 = tf.keras.layers.Dense(64, activation="selu",
                                kernel_initializer="lecun_normal",
                                bias_initializer="lecun_normal")(inp)

"""
dense_2 = tf.keras.layers.Dense(64, activation="relu",
                                kernel_initializer="RandomNormal",
                                bias_initializer="zeros")(dense_1)
dense_3 = tf.keras.layers.Dense(64, activation="relu",
                                kernel_initializer="RandomNormal",
                                bias_initializer="zeros")(dense_2)
"""

alp_drop = tf.keras.layers.AlphaDropout(0.25)(dense_1)

out = tf.keras.layers.Dense(1, activation="sigmoid")(alp_drop)

model = tf.keras.models.Model(inp, out)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1,
                                                      write_grads=True)

history = model.fit(x, y, epochs=5, verbose=True,
                    validation_split=0.2, callbacks=[tensorboard_callback])