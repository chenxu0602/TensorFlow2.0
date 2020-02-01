import io
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

def get_batch_data(buffer_size, batch_size):
    (train_data, test_data), info = tfds.load("imdb_reviews/subwords8k",
                                    split=(tfds.Split.TRAIN, tfds.Split.TEST),
                                    with_info=True, as_supervised=True)
    encoder = info.features["text"].encoder
    padded_shapes= ([None], ())
    train_batches = train_data.shuffle(buffer_size).padded_batch(batch_size, 
                                          padded_shapes=padded_shapes)
    test_batches = test_data.padded_batch(batch_size, 
                                          padded_shapes=padded_shapes)

    return train_batches, test_batches, encoder

def get_model(encoder, embedding_dim=16):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(encoder.vocab_size, embedding_dim),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                    metrics=["accuracy"])
    return model

def plot_history(history):
    history_dict = history.history
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    epochs = list(range(1, len(acc) + 1))

    plt.figure(figsize=(12, 9))
    plt.plot(epochs, acc, "bo", label="Training Acc")
    plt.plot(epochs, val_acc, "b", label="Validation Acc")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.ylim((0.5, 1))
    plt.show()

def retrieve_embeddings(model, encoder):
    out_vectors = io.open("vecs.tsv", "w", encoding="utf-8")
    out_metadata = io.open("meta.tsv", "w", encoding="utf-8")
    weights = model.layers[0].get_weights()[0]

    for num, word in enumerate(encoder.subwords):
        vec = weights[num+1]
        out_metadata.write(word + '\n')
        out_vectors.write('\t'.join([str(x) for x in vec]) + '\n')

    out_vectors.close()
    out_metadata.close()

"""
train_batches, test_batches, encoder = get_batch_data()
model = get_model(encoder)
history = model.fit(train_batches, epochs=10, validation_data=test_batches,
                    validation_steps=20)

plot_history(history)
retrieve_embeddings(model, encoder)
"""


BUFFER_SIZE = 10000
BATCH_SIZE = 64
train_batches, test_batches, encoder = get_batch_data(BUFFER_SIZE, BATCH_SIZE)

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")
])


model.compile(loss="binary_crossentropy", 
              optimizer=tf.keras.optimizers.Adam(lr=1e-4),
              metrics=["accuracy"])

history = model.fit(train_batches, epochs=5, validation_data=test_batches,
                    validation_steps=30)

def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec

def sample_predict(sentence, pad):
    encoded_sample_pred_text = encoder.encode(sentence)
    if pad:
        encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
    predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))
    return predictions

sample_text = ("This movie was awesome. The acting was incredible. Highly recommend.")
predictions = sample_predict(sample_text, pad=True) * 100
print(f"Probabililty this is a poistive review {predictions}%.")

sample_text = ("This movie was so so. The acting was mediocre. Kinf of disappointing.")
predictions = sample_predict(sample_text, pad=True) * 100
print(f"Probabililty this is a poistive review {predictions}%.")

