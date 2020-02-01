
import numpy as np
import tensorflow as tf

class Standardization(tf.keras.layers.Layer):
    def adapt(self, data_sample):
        self.means_ = np.mean(data_sample, axis=0, keepdims=True)
        self.stds_ = np.std(data_sample, axis=0, keepdims=True)

    def call(self, inputs):
        return (inputs - self.means_) / (self.stds_ + tf.keras.backend.epsilon())

std_layer = Standardization()
#std_layer.adapt(data_sample)

vocab = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
indices = tf.range(len(vocab), dtype=tf.int64)
table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)

num_oov_buckets = 2
table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)

categories = tf.constant(["NEAR BAY", "DESERT", "INLAND", "INLAND"])
cat_indices = table.lookup(categories)
print(cat_indices)
cat_one_hot = tf.one_hot(cat_indices, depth=len(vocab) + num_oov_buckets)
print(cat_one_hot)

embedding_dim = 2
embed_init = tf.random.uniform([len(vocab) + num_oov_buckets, embedding_dim])
embedding_matrix = tf.Variable(embed_init)
print(embedding_matrix)

print(tf.nn.embedding_lookup(embedding_matrix, cat_indices))

embedding = tf.keras.layers.Embedding(input_dim=len(vocab) + num_oov_buckets,
                                      output_dim=embedding_dim)
print(embedding(cat_indices))

regular_inputs = tf.keras.layers.Input(shape=[8])
categories = tf.keras.layers.Input(shape=[], dtype=tf.string)
cat_indices = tf.keras.layers.Lambda(lambda cats: table.lookup(cats))(categories)
cat_embed = tf.keras.layers.Embedding(input_dim=6, output_dim=2)(cat_indices)
encoded_inputs = tf.keras.layers.concatenate([regular_inputs, cat_embed])
outputs = tf.keras.layers.Dense(1)(encoded_inputs)
model = tf.keras.models.Model(inputs=[regular_inputs, categories], outputs=[outputs])


import tensorflow_transform as tft

def preprocess(inputs):
    median_age = inputs["housing_median_age"]
    ocean_proximity = inputs["ocean_proximity"]
    standardized_age = tft.scale_to_z_score(median_age)
    ocean_proximity_id = tft.compute_and_apply_vocabulary(ocean_proximity)
    return {
        "standardized_median_age": standardized_age,
        "ocean_proximity_id": ocean_proximity_id
    }

import tensorflow_datasets as tfds

dataset = tfds.load(name="mnist")
mnist_train, mnist_test = dataset["train"], dataset["test"]
mnist_train = mnist_train.shuffle(10000).batch(32).prefetch(1)
mnist_train = mnist_train.map(lambda items: (items["image"], items["label"]))
mnist_train.prefetch(1)
