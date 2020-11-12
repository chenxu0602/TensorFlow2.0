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
