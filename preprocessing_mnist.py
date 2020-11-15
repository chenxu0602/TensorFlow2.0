
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os, sys

(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train))
valid_set = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test))

def create_example(image, label):
	image_data = tf.io.serialize_tensor(image)
	return tf.train.Example(
		features=tf.train.Features(
			feature={
				"image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data.numpy()])),
				"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
			}
		)
	)

for image, label in valid_set.take(1):
	print(create_example(image, label))

from contextlib import ExitStack

def write_tfrecords(name, dataset, n_shards=10):
	paths = ["{}.tfrecords-{:05d}-of-{:05d}".format(name, index, n_shards)
	         for index in range(n_shards)]
	with ExitStack() as stack:
		writers = [stack.enter_context(tf.io.TFRecordWriter(path))
		           for path in paths]
		for index, (image, label) in dataset.enumerate():
			shard = index % n_shards
			example = create_example(image, label)
			writers[shard].write(example.SerializeToString())
	return paths

train_filepaths = write_tfrecords("my_fashion_mnist.train", train_set)
valid_filepaths = write_tfrecords("my_fashion_mnist.valid", valid_set)
test_filepaths = write_tfrecords("my_fashion_mnist.test", test_set)

def preprocess(tfrecord):
	feature_description = {
		"image": tf.io.FixedLenFeature([], tf.string, default_value=""),
		"label": tf.io.FixedLenFeature([], tf.int64, default_value=-1)
	}
	example = tf.io.parse_single_example(tfrecord, feature_description)
	image = tf.io.parse_tensor(example["image"], out_type=tf.uint8)
	image = tf.reshape(image, shape=[28, 28])
	return image, example["label"]

def mnist_dataset(filepaths, n_read_threads=5, shuffle_buffer_size=None,
                  n_parse_threads=5, batch_size=32, cache=True):
	dataset = tf.data.TFRecordDataset(filepaths,
	                                  num_parallel_reads=n_read_threads)
	if cache:
		dataset = dataset.cache()
	if shuffle_buffer_size:
		dataset = dataset.shuffle(shuffle_buffer_size)
	dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
	dataset = dataset.batch(batch_size)
	return dataset.prefetch(1)

train_set = mnist_dataset(train_filepaths, shuffle_buffer_size=60000)
valid_set = mnist_dataset(train_filepaths)
test_set = mnist_dataset(train_filepaths)

for X, y in train_set.take(1):
	for i in range(5):
		plt.subplot(1, 5, i + 1)
		plt.imshow(X[i].numpy(), cmap="binary")
		plt.axis("off")
		plt.title(str(y[i].numpy()))

tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

class Standardization(tf.keras.layers.Layer):
	def adapt(self, data_sample):
		self.means_ = np.mean(data_sample, axis=0, keepdims=True)
		self.stds_ = np.mean(data_sample, axis=0, keepdims=True)
	def call(self, inputs):
		return (inputs - self.means_) / (self.stds_ + tf.keras.backend.epsilon())

standardization = Standardization(input_shape=[28, 28])

sample_image_batches = train_set.take(100).map(lambda image, label: image)
sample_images = np.concatenate(list(sample_image_batches.as_numpy_iterator()),
                               axis=0).astype(np.float32)
standardization.adapt(sample_images)

model = tf.keras.models.Sequential([
	standardization,
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(100, activation="relu"),
	tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="nadam", metrics=["accuracy"])

from datetime import datetime
logs = os.path.join(os.curdir, "my_logs",
                    "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))

tensorboard_cb = tf.keras.callbacks.TensorBoard(
	log_dir=logs, histogram_freq=1, profile_batch=10)

model.fit(train_set, epochs=5, validation_data=valid_set,
          callbacks=[tensorboard_cb])
