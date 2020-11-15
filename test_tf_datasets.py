
import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf

datasets = tfds.load(name="mnist")
mnist_train, mnist_test = datasets["train"], datasets["test"]
mnist_train = mnist_train.repeat(5).batch(32)
mnist_train = mnist_train.map(lambda items: (items["image"], items["label"]))
mnist_train = mnist_train.prefetch(1)

for images, labels in mnist_train.take(1):
	print(images.shape)
	print(labels.numpy())

tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

datasets = tfds.load(name="mnist", batch_size=32, as_supervised=True)
mnist_train = datasets["train"].repeat().prefetch(1)

model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(input_shape=[28, 28, 1]),
	tf.keras.layers.Lambda(lambda images: tf.cast(images, tf.float32)),
	tf.keras.layers.Dense(10, activation="softmax")])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.SGD(lr=1e-3),
              metrics=["accuracy"])

model.fit(mnist_train, steps_per_epoch=60000 // 32, epochs=5)

import tensorflow_hub as hub

hub_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1",
                           output_shape=[50], input_shape=[], dtype=tf.string)

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

sentences = tf.constant(["It was a great movie", "The actors were amazing"])
embeddings = hub_layer(sentences)