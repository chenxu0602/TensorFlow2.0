
import tensorflow as tf

encoder = tf.keras.models.Sequential([tf.keras.layers.Dense(2, input_shape=[3])])
decoder = tf.keras.models.Sequential([tf.keras.layers.Dense(2, input_shape=[2])])
autoencoder = tf.keras.models.Sequential([encoder, decoder])

autoencoder.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=0.1))

#history = autoencoder.fit(X_train, X_train, epochs=20)
#codings = encoder.predict(X_train)


stacked_encoder = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(100, activation="selu"),
    tf.keras.layers.Dense(30, activation="selu"),
])

stacked_decoder = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, activation="selu", input_shape=[30]),
    tf.keras.layers.Dense(28 * 28, activation="sigmoid"),
    tf.keras.layers.Reshape([28, 28])
])

stacked_ae = tf.keras.models.Sequential([stacked_encoder, stacked_decoder])
stacked_ae.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.SGD(lr=1.5))
#history = stacked_ae.fit(X_train, X_train, epochs=10, validation_data=[X_valid, X_valid])


class DenseTranspose(tf.keras.layers.Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = tf.keras.activations.get(activation)
        super().__init__(**kwargs)

    def build(self, batch_input_shape):
        self.biases = self.add_weight(name="bias", initializer="zeros",
            shape=[self.dense.input_shape[-1]])
        super().build(batch_input_shape)

    def call(self, inputs):
        z =  tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)


dense_1 = tf.keras.layers.Dense(100, activation="selu")
dense_2 = tf.keras.layers.Dense(30, activation="selu")

tied_encoder = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    dense_1,
    dense_2
])

tied_decoder = tf.keras.models.Sequential([
    DenseTranspose(dense_2, activation="selu"),
    DenseTranspose(dense_1, activation="sigmoid"),
    tf.keras.layers.Reshape([28, 28])
])

tied_ae = tf.keras.models.Sequential([tied_encoder, tied_decoder])


conv_encoder = tf.keras.models.Sequential([
    tf.keras.layers.Reshape([28, 28, 1], input_shape=[28, 28]),
    tf.keras.layers.Conv2D(16, kernel_size=3, padding="same", activation="selu"),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="selu"),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="selu"),
    tf.keras.layers.MaxPool2D(pool_size=2)
])

conv_decoder = tf.keras.models.Sequential([
    tf.keras.layers.Conv2DTranspose(32, kernel_size=3, 
        strides=2, padding="valid", activation="selu", input_shape=[3, 3, 64]),
    tf.keras.layers.Conv2DTranspose(16, kernel_size=3, 
        strides=2, padding="same", activation="selu"),
    tf.keras.layers.Conv2DTranspose(1, kernel_size=3,
        strides=2, padding="same", activation="sigmoid"),
    tf.keras.layers.Reshape([28, 28])
])

conv_ae = tf.keras.models.Sequential([conv_encoder, conv_decoder])


recurrent_encoder = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(100, return_sequences=True, input_shape=[None, 28]),
    tf.keras.layers.LSTM(30)
])

recurrent_decoder = tf.keras.models.Sequential([
    tf.keras.layers.RepeatVector(28, input_shape=[30]),
    tf.keras.layers.LSTM(100, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(28, activation="sigmoid"))
])

recurrent_ae = tf.keras.models.Sequential([recurrent_encoder, recurrent_decoder])


dropout_encoder = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(100, activation="selu"),
    tf.keras.layers.Dense(30, activation="selu")
])

dropout_decoder = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, activation="selu", input_shape=[30]),
    tf.keras.layers.Dense(28 * 28, activation="sigmoid"),
    tf.keras.layers.Reshape([28, 28])
])

dropout_ae = tf.keras.models.Sequential([dropout_encoder, dropout_decoder])


sparse_l1_encoder = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(100, activation="selu"),
    tf.keras.layers.Dense(300, activation="sigmoid"),
    tf.keras.layers.ActivityRegularization(l1=1e-3)
])

sparse_l1_decoder = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, activation="selu", input_shape=[300]),
    tf.keras.layers.Dense(28 * 28, activation="sigmoid"),
    tf.keras.layers.Reshape([28, 28])
])

sparse_l1_ae = tf.keras.models.Sequential([sparse_l1_encoder, sparse_l1_decoder])

kl_divergence = tf.keras.losses.kullback_leibler_divergence

class KLDivergenceRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, weight, target=0.1):
        self.weight = weight
        self.target = target
    def __call__(self, inputs):
        mean_activities = tf.keras.backend.mean(inputs, axis=0)
        return self.weight * (
            kl_divergence(self.target, mean_activities) + 
            kl_divergence(1. - self.target, 1. - mean_activities))


kld_reg = KLDivergenceRegularizer(weight=0.05, target=0.1)
sparse_kl_encoder = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(100, activation="selu"),
    tf.keras.layers.Dense(300, activation="sigmoid", activity_regularizer=kld_reg)
])
sparse_kl_decoder = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, activation="selu", input_shape=[300]),
    tf.keras.layers.Dense(28 * 28, activation="sigmoid"),
    tf.keras.layers.Reshape([28, 28])
])
sparse_kl_ae = tf.keras.models.Sequential([sparse_kl_encoder, sparse_kl_decoder])

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return tf.keras.backend.random_normal(tf.shape(log_var)) * tf.keras.backend.exp(log_var / 2) + mean


codings_size = 10
inputs = tf.keras.layers.Input(shape=[28, 28])
z = tf.keras.layers.Flatten()(inputs)
z = tf.keras.layers.Dense(150, activation="selu")(z)
z = tf.keras.layers.Dense(100, activation="selu")(z)
codings_mean = tf.keras.layers.Dense(codings_size)(z)
codings_log_var = tf.keras.layers.Dense(codings_size)(z)
codings = Sampling()([codings_mean, codings_log_var])
variational_encoder = tf.keras.Model(
    inputs=[inputs], outputs=[codings_mean, codings_log_var, codings]
)

decoder_inputs = tf.keras.layers.Input(shape=[codings_size])
x = tf.keras.layers.Dense(100, activation="selu")(decoder_inputs)
x = tf.keras.layers.Dense(150, activation="selu")(x)
x = tf.keras.layers.Dense(28 * 28, activation="sigmoid")(x)
outputs = tf.keras.layers.Reshape([28, 28])(x)
variational_decoder = tf.keras.Model(inputs=[decoder_inputs], outputs=[outputs])

_, _, codings = variational_encoder(inputs)
reconstructions = variational_decoder(codings)
variational_ae = tf.keras.Model(inputs=[inputs], outputs=[reconstructions])

latent_loss = -0.5 * tf.keras.backend.sum(
    1 + codings_log_var - tf.keras.backend.exp(codings_log_var) -
    tf.keras.backend.square(codings_mean), axis=-1)
variational_ae.add_loss(tf.keras.backend.mean(latent_loss) / 784.)
variational_ae.compile(loss="binary_crossentropy", optimizer="rmsprop")

(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000] / 255.0, y_train_full[5000:] / 255.0

#history = variational_ae.fit(X_train, X_train, epochs=50, batch_size=128,
#    validation_data=(X_valid, X_valid))


codings = tf.random.normal(shape=[12, codings_size])
images = variational_decoder(codings).numpy()

codings_grid = tf.reshape(codings, [1, 3, 4, codings_size])
larger_grid = tf.image.resize(codings_grid, size=[5, 7])
interpolated_codings = tf.reshape(larger_grid, [-1, codings_size])
images = variational_decoder(interpolated_codings).numpy()

codings_size = 30

generator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, activation="selu", input_shape=[codings_size]),
    tf.keras.layers.Dense(150, activation="selu"),
    tf.keras.layers.Dense(28 * 28, activation="sigmoid"),
    tf.keras.layers.Reshape([28, 28])
])
discriminator = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(150, activation="selu"),
    tf.keras.layers.Dense(100, activation="selu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
gan = tf.keras.models.Sequential([generator, discriminator])

discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
discriminator.trainable = False
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")

batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(tf.cast(X_train, dtype=tf.float32)).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

def train_gan(gan, dataset, batch_size, codings_size, n_epochs=50):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        for X_batch in dataset:
            noise = tf.random.normal(shape=[batch_size, codings_size])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)

            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)

#train_gan(gan, dataset, batch_size, codings_size)


codings_sizei = 100

generator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(7 * 7 * 128, input_shape=[codings_size]),
    tf.keras.layers.Reshape([7, 7, 128]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, 
                                    padding="same", activation="selu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, 
                                    padding="same", activation="tanh")
])
distriminator = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="same",
                            activation=tf.keras.layers.LeakyReLU(0.2),
                            input_shape=[28, 28, 1]),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="same",
                            activation=tf.keras.layers.LeakyReLU(0.2)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
gan = tf.keras.models.Sequential([generator, discriminator])

X_train = X_train.reshape(-1, 28, 28, 1) * 2. - 1.