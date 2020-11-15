
import numpy as tf
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import os, sys

DOWNLOAD_ROOT = "http://ai.stanford.edu/~amaas/data/sentiment/"
FILENAME = "aclImdb_v1.tar.gz"

filepath = tf.keras.utils.get_file(FILENAME, DOWNLOAD_ROOT + FILENAME, extract=True)
path = Path(filepath).parent / "aclImdb"

for name, subdirs, files in os.walk(path):
	indent = len(Path(name).parts) - len(path.parts)
	print("    " * indent + Path(name).parts[-1] + os.sep)
	for index, filename in enumerate(sorted(files)):
		if index == 3:
			print("    " * (indent + 1) + "...")
			break
		print("    " * (indent + 1) + filename)

def review_paths(dirpath):
    return [str(path) for path in dirpath.glob("*.txt")]

train_pos = review_paths(path / "train" / "pos")
train_neg = review_paths(path / "train" / "neg")
test_valid_pos = review_paths(path / "test" / "pos")
test_valid_neg = review_paths(path / "test" / "neg")

np.random.shuffle(test_valid_pos)

test_pos = test_valid_pos[:5000]
test_neg = test_valid_neg[:5000]
valid_pos = test_valid_pos[5000:]
valid_neg = test_valid_neg[5000:]

"""
def imdb_dataset(filepaths_positive, filepaths_negative):
	reviews, labels = [], []
	for filepaths, label in ((filepaths_negative, 0), (filepaths_negative, 1)):
		for filepath in filepaths:
			with open(filepath) as review_file:
				reviews.append(review_file.read())
			labels.append(label)
	return tf.data.Dataset.from_tensor_slices(
		(tf.constant(reviews), tf.constant(labels)))

for X, y in imdb_dataset(train_pos, train_neg).take(3):
	print(X)
	print(y)
	print()
"""

def imdb_dataset(filepaths_positive, filepaths_negative, n_read_threads=5):
	dataset_neg = tf.data.TextLineDataset(filepaths_negative,
	                                      num_parallel_reads=n_read_threads)
	dataset_neg = dataset_neg.map(lambda review: (review, 0))
	dataset_pos = tf.data.TextLineDataset(filepaths_positive,
	                                      num_parallel_reads=n_read_threads)
	dataset_pos = dataset_pos.map(lambda review: (review, 1))
	return tf.data.Dataset.concatenate(dataset_pos, dataset_neg)

batch_size = 32

train_set = imdb_dataset(train_pos, train_neg).shuffle(25000).batch(batch_size).prefetch(1)
valid_set = imdb_dataset(valid_pos, valid_neg).batch(batch_size).prefetch(1)
test_set = imdb_dataset(test_pos, test_neg).batch(batch_size).prefetch(1)

def preprocess(X_batch, n_words=50):
	shape = tf.shape(X_batch) * tf.constant([1, 0]) + tf.constant([0, n_words])
	Z = tf.strings.substr(X_batch, 0, 300)
	Z = tf.strings.lower(Z)
	Z = tf.strings.regex_replace(Z, b"<br\\s*/?>", b" ")
	Z = tf.strings.split(Z)
	return Z.to_tensor(shape=shape, default_value=b"<pad>")

X_example = tf.constant(["It's a great, great movie! I loved it.", "It was terrible, run away!!!"])
preprocess(X_example)

from collections import Counter

def get_vocabulary(data_sample, max_size=1000):
	preprocessed_reviews = preprocess(data_sample).numpy()
	counter = Counter()
	for words in preprocessed_reviews:
		for word in words:
			if word != b"<pad>":
				counter[word] += 1
	return [b"<pad>"] + [word for word, count in counter.most_common(max_size)]

get_vocabulary(X_example)

class TextVectorization(tf.keras.layers.Layer):
	def __init__(self, max_vocabulary_size=1000, n_oov_buckets=100, dtype=tf.string, **kwargs):
		super().__init__(dtype=dtype, **kwargs)
		self.max_vocabulary_size = max_vocabulary_size
		self.n_oov_buckets = n_oov_buckets

	def adapt(self, data_sample):
		self.vocab = get_vocabulary(data_sample, self.max_vocabulary_size)
		words = tf.constant(self.vocab)
		word_ids = tf.range(len(self.vocab), dtype=tf.int64)
		vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
		self.table = tf.lookup.StaticVocabularyTable(vocab_init, self.n_oov_buckets)

	def call(self, inputs):
		preprocessed_inputs = preprocess(inputs)
		return self.table.lookup(preprocessed_inputs)

text_vectorization = TextVectorization()

text_vectorization.adapt(X_example)
text_vectorization(X_example)

max_vocabulary_size = 1000
n_oov_buckets = 100

sample_review_batches = train_set.map(lambda review, label: review)
sample_reviews = np.concatenate(list(sample_review_batches.as_numpy_iterator()), axis=0)

text_vectorization = TextVectorization(max_vocabulary_size, n_oov_buckets,
                                       input_shape=[])
text_vectorization.adapt(sample_reviews)
text_vectorization(X_example)

simple_example = tf.constant([[1, 3, 1, 0, 0], [2, 2, 0, 0, 0]])
print(tf.reduce_sum(tf.one_hot(simple_example, 4), axis=1))


class BagOfWords(tf.keras.layers.Layer):
	def __init__(self, n_tokens, dtype=tf.int32, **kwargs):
		super().__init__(dtype=tf.int32, **kwargs)
		self.n_tokens = n_tokens
	def call(self, inputs):
		one_hot = tf.one_hot(inputs, self.n_tokens)
		return tf.reduce_sum(one_hot, axis=1)[:, 1:]

bag_of_words = BagOfWords(n_tokens=4)
print(bag_of_words(simple_example))

n_tokens = max_vocabulary_size + n_oov_buckets
bag_of_words = BagOfWords(n_tokens)

model = tf.keras.models.Sequential([
	text_vectorization,
	bag_of_words,
	tf.keras.layers.Dense(100, activation="relu"),
	tf.keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
model.fit(train_set, epochs=5, validation_data=valid_set)

def compute_mean_embedding(inputs):
	not_pad = tf.math.count_nonzero(inputs, axis=-1)
	n_words = tf.math.count_nonzero(not_pad, axis=-1, keepdims=True)
	sqrt_n_words = tf.math.sqrt(tf.cast(n_words, tf.float32))
	return tf.reduce_mean(inputs, axis=1) * sqrt_n_words

another_example = tf.constant([[[1., 2., 3.], [4., 5., 0.], [0., 0., 0.]],
                               [[6., 0., 0.], [0., 0., 0.], [0., 0., 0.]]])
print(compute_mean_embedding(another_example))

embedding_size = 20

model = tf.keras.models.Sequential([
	text_vectorization,
	tf.keras.layers.Embedding(input_dim=n_tokens,
	                          output_dim=embedding_size,
	                          mask_zero=True),
	tf.keras.layers.Lambda(compute_mean_embedding),
	tf.keras.layers.Dense(100, activation="relu"),
	tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy",
              optimizer="nadam",
              metrics=["accuracy"])
model.fit(train_set, epochs=5, validation_data=valid_set)
