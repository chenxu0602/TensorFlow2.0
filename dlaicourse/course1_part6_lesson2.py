from pathlib import Path
import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.

test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
	model = tf.keras.models.Sequential([
		tf.keras.layers.Conv2D(64, (3,3), activation="relu", input_shape=(28, 28, 1)),
		tf.keras.layers.MaxPooling2D(2, 2),
		tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
		tf.keras.layers.MaxPooling2D(2, 2),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(128, activation="relu"),
		tf.keras.layers.Dense(10, activation="softmax")
	])

	model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
	print(model.summary())

checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = str(Path(checkpoint_dir) / "ckpt_{epoch}")

def decay(epoch):
	if epoch < 3:
		return 1e-3
	elif epoch >= 3 and epoch < 7:
		return 1e-4
	else:
		return 1e-5

class PrintLR(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		print("\nLearning rate for epoch {} is {}".format(epoch + 1,
														  model.optimizer.lr.numpy()))

callbacks = [
	tf.keras.callbacks.TensorBoard(log_dir="./logs"),
	tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
									   save_weights_only=True),
	tf.keras.callbacks.LearningRateScheduler(decay),
	PrintLR()
]

model.fit(training_images, training_labels, epochs=5, callbacks=callbacks)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
test_loss = model.evaluate(test_images, test_labels)

import matplotlib.pyplot as plt
f, axarr = plt.subplots(3, 4)
FIRST_IMAGE=0
SECOND_IMAGE=7
THIRD_IMAGE=26
CONVOLUTION_NUMBER=1

layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

for i in range(4):
	f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[i]
	axarr[0, i].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap="inferno")
	axarr[0, i].grid(False)
	f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[i]
	axarr[1, i].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap="inferno")
	axarr[1, i].grid(False)
	f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[i]
	axarr[2, i].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap="inferno")
	axarr[2, i].grid(False)



