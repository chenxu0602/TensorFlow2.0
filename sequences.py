
import tensorflow as tf
import numpy as np

def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)
    return series[..., np.newaxis].astype(np.float32)


n_steps = 50
series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

y_pred = X_valid[:, -1]
print("Naive forecasting:")
print(np.mean(tf.keras.losses.mean_squared_error(y_valid, y_pred)))

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[50, 1]),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer="adam", loss="mse")
#history = model.fit(X_train, y_train, 
#    validation_data=((X_valid, y_valid)), batch_size=32, epochs=20)

model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(1, input_shape=[None, 1])
])
model.compile(optimizer="adam", loss="mse")
#history = model.fit(X_train, y_train, 
#    validation_data=((X_valid, y_valid)), batch_size=32, epochs=5)

model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.SimpleRNN(20, return_sequences=True),
    tf.keras.layers.SimpleRNN(1)
])
model.compile(optimizer="adam", loss="mse")
#history = model.fit(X_train, y_train, 
#    validation_data=((X_valid, y_valid)), batch_size=32, epochs=20)

model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.SimpleRNN(20),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer="adam", loss="mse")
#history = model.fit(X_train, y_train, 
#    validation_data=((X_valid, y_valid)), batch_size=32, epochs=5)

"""
series = generate_time_series(1, n_steps + 10)
X_new, y_new = series[:, :n_steps], series[:, n_steps:]
X = X_new
for step_ahead in range(10):
    y_pred_one = model.predict(X[:, step_ahead:])[:, np.newaxis, :]
    X = np.concatenate([X, y_pred_one], axis=1)
y_pred = X[:, n_steps:]
"""

series = generate_time_series(10000, n_steps + 10)
X_train, y_train = series[:7000, :n_steps], series[:7000, -10:, 0]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -10:, 0]
X_test, y_test = series[9000:, :n_steps], series[9000:, -10:, 0]

model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.SimpleRNN(20),
    tf.keras.layers.Dense(10)
])
model.compile(optimizer="adam", loss="mse")
#history = model.fit(X_train, y_train, 
#    validation_data=((X_valid, y_valid)), batch_size=32, epochs=20)

y = np.empty((10000, n_steps, 10))
for step_ahead in range(1, 10 + 1):
    y[:, :, step_ahead - 1] = series[:, step_ahead:step_ahead + n_steps, 0]

y_train = y[:7000]
y_valid = y[7000:9000]
y_test = y[9000:]

model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.SimpleRNN(20, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10))
])

def last_time_step_mse(y_true, y_pred):
    return tf.keras.metrics.mean_squared_error(y_true[:, -1], y_pred[:, -1])

optimizer = tf.keras.optimizers.Adam(lr=0.01)
model.compile(optimizer=optimizer, loss="mse", metrics=[last_time_step_mse])
#history = model.fit(X_train, y_train, 
#    validation_data=((X_valid, y_valid)), batch_size=32, epochs=20)


class LNSimpleRNNCell(tf.keras.layers.Layer):
    def __init__(self, units, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.state_size = units
        self.output_szie = units
        self.simple_rnn_cell = tf.keras.layers.SimpleRNNCell(units, activation=None)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.activation = tf.keras.activations.get(activation)

    def call(self, inputs, states):
        outputs, new_states = self.simple_rnn_cell(inputs, states)
        norm_outputs = self.activation(self.layer_norm(outputs))
        return norm_outputs, [norm_outputs]


model = tf.keras.models.Sequential([
    tf.keras.layers.RNN(LNSimpleRNNCell(20), return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.RNN(LNSimpleRNNCell(20), return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10))
])
optimizer = tf.keras.optimizers.Adam(lr=0.01)
model.compile(optimizer=optimizer, loss="mse", metrics=[last_time_step_mse])
#history = model.fit(X_train, y_train, 
#    validation_data=((X_valid, y_valid)), batch_size=32, epochs=20)

"""
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.LSTM(20, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10))
])
"""

"""
model = tf.keras.models.Sequential([
    tf.keras.layers.RNN(tf.keras.layers.LSTMCell(20), return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.RNN(tf.keras.layers.LSTMCell(20), return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10))
])
"""

"""
model = tf.keras.models.Sequential([
    tf.keras.layers.RNN(tf.keras.experimental.PeepholeLSTMCell(20), return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.RNN(tf.keras.experimental.PeepholeLSTMCell(20), return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10))
])
"""

"""
model = tf.keras.models.Sequential([
    tf.keras.layers.RNN(tf.keras.layers.GRUCell(20), return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.RNN(tf.keras.layers.GRUCell(20), return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10))
])
"""

"""
model = tf.keras.models.Sequential([
    tf.keras.layers.GRU(20, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.GRU(20, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10))
])
"""

"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding='VALID', input_shape=[None, 1]),
    tf.keras.layers.GRU(20, return_sequences=True),
    tf.keras.layers.GRU(20, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10))
])
"""

"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding='VALID', input_shape=[None, 1]),
    tf.keras.layers.LSTM(20, return_sequences=True),
    tf.keras.layers.LSTM(20, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10))
])

model.compile(optimizer=optimizer, loss="mse", metrics=[last_time_step_mse])
history = model.fit(X_train, y_train[:, 3::2], 
    validation_data=((X_valid, y_valid[:, 3::2])), batch_size=32, epochs=20)
"""

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=[None, 1]))
for rate in (1, 2, 4, 8) * 2:
    model.add(tf.keras.layers.Conv1D(filters=20, kernel_size=2, padding="causal", 
                activation="relu", dilation_rate=rate))
model.add(tf.keras.layers.Conv1D(filters=10, kernel_size=1))
model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))