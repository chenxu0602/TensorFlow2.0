import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.model_selection import train_test_split

from pathlib import Path

dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
csv_file = 'datasets/petfinder-mini/petfinder-mini.csv'

tf.keras.utils.get_file("petfinder_mini.zip",
                        dataset_url,
                        extract=True,
                        cache_dir='.')

dataframe = pd.read_csv(csv_file)

# In the original dataset "4" indicates the pet was not adopted.
dataframe["target"] = np.where(dataframe["AdoptionSpeed"] == 4, 0, 1)

# Drop un-used columns.
dataframe = dataframe.drop(columns=["AdoptionSpeed", "Description"])

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(dataframe, test_size=0.2)
print(len(train), "train examples")
print(len(val), "val examples")
print(len(test), "test examples")

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

batch_size = 5 # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(1):
    print("Every feature: ", list(feature_batch.keys()))
    print("A batch of ages: ", feature_batch["Age"])
    print("A batch of targets: ", label_batch)


# We will use this batch to demonstrate several types of feature columns
example_batch = next(iter(train_ds))[0]

# A utility method to create a feature column
# and to transform a batch of data
def demo(feature_column):
    feature_layer = tf.keras.layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())

photo_count = tf.feature_column.numeric_column("PhotoAmt")
demo(photo_count)

age = tf.feature_column.numeric_column("Age")
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[1, 3, 5])
demo(age_buckets)

animal_type = tf.feature_column.categorical_column_with_vocabulary_list(
    "Type", ["Cat", "Dog"])
animal_type_one_hot = tf.feature_column.indicator_column(animal_type)
demo(animal_type_one_hot)


# Notice the input to the embedding column is the categorical column
# we previously created
breed1 = tf.feature_column.categorical_column_with_vocabulary_list(
    "Breed1", dataframe.Breed1.unique())
breed1_embedding = tf.feature_column.embedding_column(breed1, dimension=8)
demo(breed1_embedding)

breed1_hashed = tf.feature_column.categorical_column_with_hash_bucket(
    "Breed1", hash_bucket_size=10)
demo(tf.feature_column.indicator_column(breed1_hashed))

crossed_feature = tf.feature_column.crossed_column([age_buckets, animal_type], hash_bucket_size=10)
demo(tf.feature_column.indicator_column(crossed_feature))


feature_columns = []

# numeric cols
for header in ["PhotoAmt", "Fee", "Age"]:
    feature_columns.append(tf.feature_column.numeric_column(header))

# bucketized cols
age = tf.feature_column.numeric_column("Age")
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[1,2,3,4,5])
feature_columns.append(age_buckets)

# indicator_columns
indicator_column_names = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize',
                          'FurLength', 'Vaccinated', 'Sterilized', 'Health']
for col_name in indicator_column_names:
    categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
        col_name, dataframe[col_name].unique())
    indicator_column = tf.feature_column.indicator_column(categorical_column)
    feature_columns.append(indicator_column)

# embedding columns
breed1 = tf.feature_column.categorical_column_with_vocabulary_list(
    "Breed1", dataframe.Breed1.unique())
breed1_embedding = tf.feature_column.embedding_column(breed1, dimension=8)
feature_columns.append(breed1_embedding)

# crossed columns
age_type_feature = tf.feature_column.crossed_column([age_buckets, animal_type], hash_bucket_size=10)
feature_columns.append(tf.feature_column.indicator_column(age_type_feature))

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

model = tf.keras.Sequential([
    feature_layer,
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(.1),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam",
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=["accuracy"])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=10)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)


