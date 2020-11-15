
import os, sys
import zipfile
import tensorflow as tf

# local_zip = "horse-or-human.zip"
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall("horse-or-human")
#
# local_zip = "validation-horse-or-human.zip"
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall("validation-horse-or-human")
# zip_ref.close()


train_horse_dir = os.path.join("horse-or-human/horses")
train_human_dir = os.path.join("horse-or-human/humans")

validation_horse_dir = os.path.join("validation-horse-or-human/horses")
validation_human_dir = os.path.join("validation-horse-or-human/humans")

train_horse_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)

validation_horse_names = os.listdir(validation_horse_dir)
validation_human_names = os.listdir(validation_human_dir)

model = tf.keras.models.Sequential([
	tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(150, 150, 3)),
	tf.keras.layers.MaxPooling2D(2, 2),
	tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
	tf.keras.layers.MaxPooling2D(2, 2),
	tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
	tf.keras.layers.MaxPooling2D(2, 2),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(512, activation="relu"),
	tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
              metrics=["accuracy"])


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255.)
validation_datagen = ImageDataGenerator(rescale=1/255.)

train_generator = train_datagen.flow_from_directory(
	"horse-or-human",
	target_size=(150, 150),
	batch_size=128,
	class_mode="binary")

validation_generator = validation_datagen.flow_from_directory(
	"validation-horse-or-human",
	target_size=(150, 150),
	batch_size=128,
	class_mode="binary")

history = model.fit(
	train_generator,
	steps_per_epoch=8,
	epochs=15,
	verbose=1,
	validation_data=validation_generator,
	validation_steps=8)

import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

successive_outputs = [layer.output for layer in model.layers[1:]]

visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)

horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
img_path = random.choice(horse_img_files + human_img_files)

img = load_img(img_path, target_size=(150, 150))
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

x /= 255.

successive_feature_maps = visualization_model.predict(x)
layer_names = [layer.name for layer in model.layers]

for layer_name, feature_map in zip(layer_names, successive_feature_maps):
	if len(feature_map.shape) == 4:
		n_features = feature_map.shape[-1]
		size = feature_map.shape[1]
		display_grid = np.zeros((size, size * n_features))
		for i in range(n_features):
			x = feature_map[0, :, :, i]
			x -= x.mean()
			x /= x.std()
			x *= 64
			x += 128
			x = np.clip(x, 0, 255).astype("uint8")
			display_grid[:, i * size : (i + 1) * size] = x

		scale = 20. / n_features
		plt.figure(figsize=(scale * n_features, scale))
		plt.title(layer_name)
		plt.grid(False)
		plt.imshow(display_grid, aspect="auto", cmap="viridis")





