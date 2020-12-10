import tensorflow as tf
import kerastuner
import IPython


(img_train, label_train), (img_test, label_test) = tf.keras.datasets.fashion_mnist.load_data()

img_train = img_train.astype("float32") / 255.0
img_test = img_test.astype("float32") / 255.0

def model_builder(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int("units", min_value=32, max_value=512, step=32)
    model.add(tf.keras.layers.Dense(units=hp_units, activation="relu"))
    model.add(tf.keras.layers.Dense(10))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    return model


tuner = kerastuner.Hyperband(model_builder,
                            objective="val_accuracy",
                            max_epochs=10,
                            factor=3,
                            directory="my_dir",
                            project_name="intro_to_kt")

class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)


tuner.search(img_train, label_train, epochs=10, validation_data=(img_test, label_test), callbacks=[ClearTrainingOutput()])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)
model.fit(img_train, label_train, epochs=10, validation_data=(img_test, label_test))
