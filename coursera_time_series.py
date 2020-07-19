import numpy as np
import tensorflow as tf
import os, sys

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
	dataset = tf.data.Dataset.from_tensor_slices(series)
	dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
	dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
	dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
	dataset = dataset.batch(batch_size).prefetch(1)
	return dataset

def windowed_dataset2(series, window_size, batch_size, shuffle_buffer):
	series = tf.expand_dims(series, axis=-1)
	dataset = tf.data.Dataset.from_tensor_slices(series)
	dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
	dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
	dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
	dataset = dataset.batch(batch_size).prefetch(1)
	return dataset

def model_forcast(model, series, window_size):
	ds = tf.data.Dataset.from_tensor_slices(series)
	ds = ds.window(window_size, shift=1, drop_remainder=True)
	ds = ds.flat_map(lambda window: window.batch(window_size))
	ds = ds.batch(32).prefetch(1)
	forecast = model.predict(ds)
	return forecast


window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

sys.exit(0)

series = np.array(list(range(50)))
dataset = windowed_dataset(series, window_size, batch_size, shuffle_buffer_size)
l0 = tf.keras.layers.Dense(1, input_shape=[window_size])
model = tf.keras.models.Sequential([l0])

model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
model.fit(dataset, epochs=100, verbose=0)
print("Layer weights {}".format(l0.get_weights()))

print(series[1:21])
print(model.predict(series[1:21][np.newaxis]))

forecast = []
for time in range(len(series) - window_size):
	forecast.append(model.predict(series[time:time + window_size][np.newaxis]))


tf.keras.backend.clear_sessioN()
train_set = windowed_dataset(x_train, window_size, batch_size=128,
                             shuffle_buffer=shuffle_buffer_size)
model = tf.keras.models.Sequential([
	tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
	# tf.keras.layers.SimpleRNN(40, return_sequences=True),
	# tf.keras.layers.SimpleRNN(40),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32), return_sequences=True),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32), return_sequences=True),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
	tf.keras.layers.Dense(1),
	tf.keras.layers.Lambda(lambda x: x + 100.0)
])

lr_schedule = tf.keras.callbacks.LearingRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])


model = tf.keras.models.Sequential([
	tf.keras.layers.Conv1D(filters=32, kernel_size=5,
	                       strides=1, padding="causal",
	                       activation="relu",
	                       input_shape=[None, 1]),
	tf.keras.layers.LSTM(32, return_sequences=True),
	tf.keras.layers.LSTM(32, return_sequences=True),
	tf.keras.layers.Dense(1),
	tf.keras.layers.Lambda(lambda x: x * 200)
])

optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
model.fit(dataset, epochs=500)


train_set = windowed_dataset(x_train, window_size=60, batch_size=2250, shuffle_buffer_size)
model = tf.keras.models.Sequential([
	tf.keras.layers.Conv1D(filters=60, kernel_size=5, strides=1, padding="causal",
	                       activation="relu", input_shape=[None, 1]),
	tf.keras.layers.LSTM(60, return_sequences=True),
	tf.keras.layers.LSTM(60, return_sequences=True),
	tf.keras.layers.Dense(30, activation="relu"),
	tf.keras.layers.Dense(10, activation="relu"),
	tf.keras.layers.Dense(1),
	tf.keras.layers.Lambda(lambda x: x * 400)
])

optmizer = tf.keras.layers.optimizers.SGD(lr=1e-5, momentum=0.9)
model.compile(loss=tf.keras.layers.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set, epochs=500)
