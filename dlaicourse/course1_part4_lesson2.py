
import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()


import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128, activation="relu"),
	tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		if logs.get("loss") < 0.4:
			print("\nRecord 60% accuarcy so cancelling training!")
			self.model.stop_training = True

callbacks = myCallback()

model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])

model.evaluate(test_images, test_labels)

classification = model.predict(test_images)
print(classification[0])
print(test_labels[0])