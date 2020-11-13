import io, os, sys, re, tqdm, re, string
import tensorflow as tf
import numpy as np
import itertools

SEED = 42
AUTOTUNE = tf.data.experimental.AUTOTUNE

sentence = "The wide road shimmered in the hot sun"
tokens = list(sentence.lower().split())
print(len(tokens))

vocab, index = {}, 1
vocab['<pad>'] = 0
for token in tokens:
    if token not in vocab:
        vocab[token] = index
        index += 1

vocab_size = len(vocab)
print(vocab)

inverse_vocab = {index: token for token, index in vocab.items()}
print(inverse_vocab)

example_sequence = [vocab[word] for word in tokens]
print(example_sequence)

window_size = 2
positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
    example_sequence,
    vocabulary_size=vocab_size,
    window_size=window_size,
    negative_samples=0)
print(len(positive_skip_grams))

for target, context in positive_skip_grams[:5]:
    print(f"({target}, {context}): ({inverse_vocab[target]}, {inverse_vocab[context]})")


target_word, context_word = positive_skip_grams[0]
num_ns = 4

context_class = tf.reshape(tf.constant(context_word, dtype="int64"), (1, 1))

negative_smapling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
    true_classes=context_class,
    num_true=1,
    num_sampled=num_ns,
    unique=True,
    range_max=vocab_size,
    seed=SEED,
    name="negative_sampling"
)
print(negative_smapling_candidates)
print([inverse_vocab[index.numpy()] for index in negative_smapling_candidates])

negative_smapling_candidates = tf.expand_dims(negative_smapling_candidates, 1)


context = tf.concat([context_class, negative_smapling_candidates], 0)

label = tf.constant([1] + [0] * num_ns, dtype="int64")

target = tf.squeeze(target_word)
context = tf.squeeze(context)
label = tf.squeeze(label)

print(f"target_index              : {target}")
print(f"target_word               : {inverse_vocab[target_word]}")
print(f"context_indices           : {context}")
print(f"context_words             : {[inverse_vocab[c.numpy()] for c in context]}")
print(f"label                     : {label}")

print(f"target    :", target)
print(f"context   :", context)
print(f"label     :", label)


sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(size=10)


def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    targets, contexts, labels = [], [], []
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    for sequence in tqdm.tqdm(sequences):
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0)

    for target_word, context_word in positive_skip_grams:
        context_class = tf.expand_dims(
            tf.constant([context_word], dtype="int64"), 1)
        negative_smapling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
            true_classes=context_class,
            num_true=1,
            num_sampled=num_ns,
            unique=True,
            range_max=vocab_size,
            seed=SEED,
            name="negative_sampling")

        negative_smapling_candidates = tf.expand_dims(
            negative_smapling_candidates, 1)


        context = tf.concat([context_class, negative_smapling_candidates], 0)
        label = tf.constant([1] + [0] * num_ns, dtype="int64")

        targets.append(target_word)
        contexts.append(context)
        labels.append(label)

    return targets, contexts, labels


path_to_file = tf.keras.utils.get_file("shakespeare.txt", "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")

with open(path_to_file) as f:
    lines = f.read().splitlines()

for line in lines[:20]:
    print(line)

text_ds = tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x), bool))

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase,
                                    "[%s]" % re.escape(string.punctuation), "")

vocab_size = 4096
sequence_length = 10

vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length)

vectorize_layer.adapt(text_ds.batch(1024))
inverse_vocab = vectorize_layer.get_vocabulary()
print(len(inverse_vocab))
print(inverse_vocab[:20])

def vectorize_text(text):
    text = tf.expand_dims(text, -1)
    return tf.squeeze(vectorize_layer(text))

text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()

sequences = list(text_vector_ds.as_numpy_iterator())
print(len(sequences))

for seq in sequences[:5]:
    print(f"{seq} => {[inverse_vocab[i] for i in seq]}")

targets, contexts, labels = generate_training_data(
    sequences=sequences,
    window_size=2,
    num_ns=4,
    vocab_size=vocab_size,
    seed=SEED)

print(len(targets), len(contexts), labels)


BATCH_SIZE = 1024
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(dataset)

dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
print(dataset)
