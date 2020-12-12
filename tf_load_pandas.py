import pandas as pd
import tensorflow as tf

csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv')

df = pd.read_csv(csv_file)
df["thal"] = pd.Categorical(df["thal"])
df["thal"] = df.thal.cat.codes

target = df.pop("target")
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

for feat, targ in dataset.take(5):
    print(f"Features: {feat}, Target: {targ}")

train_dataset = dataset.shuffle(len(df)).batch(1)

def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer="adam",
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    return model

model = get_compiled_model()
model.fit(train_dataset, epochs=15)


inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df.keys()}
x = tf.stack(list(inputs.values()), axis=-1)

x = tf.keras.layers.Dense(10, activation="relu")(x)
output = tf.keras.layers.Dense(1)(x)

model_func = tf.keras.Model(inputs=inputs, outputs=output)

model_func.compile(optimizer="adam",
                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                   metrics=["accuracy"])

dict_slices = tf.data.Dataset.from_tensor_slices((df.to_dict("list"), target.values)).batch(16)

for dict_slice in dict_slices.take(1):
    print(dict_slice)

model_func.fit(dict_slices, epochs=15)


