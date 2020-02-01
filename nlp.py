
import tensorflow as tf
import numpy as np
import datetime


shakespeare_url = "https://homl.info/shakespeare"
filepath = tf.keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_text = f.read()

tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts([shakespeare_text])

max_id = len(tokenizer.word_index)

dataset_size = tokenizer.document_count

[encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1


train_size = int(len(encoded) * dataset_size * 90 / 100)
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

n_steps = 100
window_length = n_steps + 1
dataset = dataset.window(window_length, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(window_length))

batch_size = 32
dataset = dataset.shuffle(10000).batch(batch_size)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))

dataset = dataset.map(lambda X_batch, y_batch: (tf.one_hot(X_batch, depth=max_id), y_batch))
dataset = dataset.prefetch(1)

model = tf.keras.models.Sequential([
    tf.keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id],
                        dropout=0.2, recurrent_dropout=0.2),
    tf.keras.layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(max_id, activation="softmax"))
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
#history = model.fit(dataset, epochs=5)

def preprocess(texts):
    X = np.array(tokenizer.texts_to_sequences(texts)) - 1
    return tf.one_hot(X, max_id)

X_new = preprocess(["How are yo"])
y_pred = model.predict_classes(X_new)
print(tokenizer.sequences_to_texts(y_pred+1)[0][-1])

def next_char(text, temperature=1):
    X_new = preprocess([text])
    y_proba = model.predict(X_new)[0, -1:, :]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]

def complete_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text

#print(complete_text("t", temperature=0.2))
#print(complete_text("w", temperature=1))
#print(complete_text("w", temperature=2))


dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
dataset = dataset.window(window_length, shift=n_steps, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(window_length))
dataset = dataset.batch(1)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
dataset = dataset.map(lambda X_batch, y_batch: (tf.one_hot(X_batch, depth=max_id), y_batch))
dataset = dataset.prefetch(1)

model = tf.keras.models.Sequential([
    tf.keras.layers.GRU(128, return_sequences=True, stateful=True,
                        dropout=0.2, recurrent_dropout=0.2, 
                        batch_input_shape=[batch_size, None, max_id]),
    tf.keras.layers.GRU(128, return_sequences=True, stateful=True,
                        dropout=0.2, recurrent_dropout=0.2),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(max_id, activation="softmax"))
])

class ResetStatesCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
#model.fit(dataset, epochs=50, callbacks=[ResetStatesCallback()])


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data()

word_index = tf.keras.datasets.imdb.get_word_index()
id_to_word = {id_ + 3: word for word, id_ in word_index.items()}

for id_, token in enumerate(("<pad>", "<sos>", "<unk>")):
    id_to_word[id_] = token

print(" ".join([id_to_word[id_] for id_ in X_train[0][:10]]))

import tensorflow_datasets as tfds

datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
train_size = info.splits["train"].num_examples

def preprocess(X_batch, y_batch):
    X_batch = tf.strings.substr(X_batch, 0, 300)
    X_batch = tf.strings.regex_replace(X_batch, b"<br\\s*/?>", b" ")
    X_batch = tf.strings.regex_replace(X_batch, b"<^a-zA-Z'>", b" ")
    X_batch = tf.strings.split(X_batch)
    return X_batch.to_tensor(default_value=b"<pad>"), y_batch

from collections import Counter
vocabulary = Counter()
for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
    for review in X_batch:
        vocabulary.update(list(review.numpy()))


vocab_size = 10000
truncated_vocabulary = [
    word for word, count in vocabulary.most_common()[:vocab_size]]

words = tf.constant(truncated_vocabulary)
word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
num_oov_buckets = 1000
table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)
print(table.lookup(tf.constant([b"This movie was faaaaaaatastic".split()])))

def encode_words(X_batch, y_batch):
    return table.lookup(X_batch), y_batch

train_set = datasets["train"].batch(32).map(preprocess)
train_set = train_set.map(encode_words).prefetch(1)

embed_size = 128
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size, input_shape=[None]),
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.GRU(128),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

K = tf.keras.backend
inputs = tf.keras.layers.Input(shape=[None])
mask = tf.keras.layers.Lambda(lambda inputs: K.not_equal(inputs, 0))(inputs)
z = tf.keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size)(inputs)
z = tf.keras.layers.GRU(128, return_sequences=True)(z, mask=mask)
z = tf.keras.layers.GRU(128)(z, mask=mask)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(z)
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
#history = model.fit(train_set, epochs=5, callbacks=[tensorboard_callback])

import tensorflow_hub as hub

model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1", 
        dtype=tf.string, input_shape=[], output_shape=[50]),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
train_size = info.splits["train"].num_examples
batch_size = 32
train_set = datasets["train"].batch(batch_size).prefetch(1)
#history = model.fit(train_set, epochs=5)

import tensorflow_addons as tfa

encoder_inputs = tf.keras.layers.Input(shape=[None], dtype=np.int32)
decoder_inputs = tf.keras.layers.Input(shape=[None], dtype=np.int32)
sequence_lengths = tf.keras.layers.Input(shape=[], dtype=np.int32)

embeddings = tf.keras.layers.Embedding(vocab_size, embed_size)
encoder_embeddings = embeddings(encoder_inputs)
decoder_embeddings = embeddings(decoder_inputs)
encoder = tf.keras.layers.LSTM(512, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_embeddings)
encoder_state = [state_h, state_c]

sampler = tfa.seq2seq.sampler.TrainingSampler()

decoder_cell = tf.keras.layers.LSTMCell(512)
output_layer = tf.keras.layers.Dense(vocab_size)
decoder = tfa.seq2seq.basic_decoder.BasicDecoder(decoder_cell, sampler, output_layer=output_layer)

final_outputs, final_state, final_sequence_lengths = decoder(
    decoder_embeddings, initial_state=encoder_state, sequence_length=sequence_lengths)
y_proba = tf.nn.softmax(final_outputs.rnn_output)

model = tf.keras.Model(inputs=[encoder_inputs, decoder_inputs, sequence_lengths], outputs=[y_proba])


beam_width = 10
decoder = tfa.seq2seq.beam_search_decoder.BeamSearchDecoder(
    cell=decoder_cell, beam_width=beam_width, output_layer=output_layer)
decoder_initial_state = tfa.seq2seq.beam_search_decoder.tile_batch(
    encoder_state, multiplier=beam_width)
outputs, _, _ = decoder(decoder_embeddings, start_tokens=start_tokens,
    end_token=end_token, initial_state=decoder_initial_state)



