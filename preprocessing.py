
import tensorflow as tf 

X = tf.range(10)
dataset = tf.data.Dataset.from_tensor_slices(X)

for item in dataset:
    print(item)

dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)

dataset = dataset.map(lambda x: x * 2)

dataset = dataset.apply(tf.data.experimental.unbatch())

dataset = dataset.filter(lambda x: x < 10)

for item in dataset.take(3):
    print(item)

dataset = tf.data.Dataset.range(10).repeat(3)
dataset = dataset.shuffle(buffer_size=5, seed=42).batch(7)
for item in dataset:
    print(item)


n_inputs = 8
X_mean, X_std = [0.] * n_inputs, [0.] * n_inputs

def preprocess(line):
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return (x - X_mean) / X_std, y


def csv_reader_dataset(filepaths, repeat=1, n_readers=5, n_read_threads=None,
                       shuffle_buffer_size=10000, n_parse_threads=5, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths)
    dataset = dataset.interleave(lambda filepath:
                                 tf.data.TextLineDataset(filepath).skip(1),
                                 cycle_length=n_readers,
                                 num_parallel_calls=n_read_threads)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    dataset = dataset.shuffle(shuffle_buffer_size).repeat(repeat)
    return dataset.batch(batch_size).prefetch(1)

@tf.function
def train(model, optimizer, loss_fn, n_epochs, [...]):
    train_set = csv_reader_dataset(train_filepaths, repeat=n_epochs, [...])
    for X_batch, y_batch in train_set:
        with tf.GradientTape() as tape:
            y_pred = model(X_batch)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))