
import os, zipfile

local_zip = "horse-or-human.zip"
zip_ref = zipfile.ZipFile(local_zip)
zip_ref.extractall("horse-or-human")

local_zip = "validation-horse-or-human.zip"
zip_ref = zipfile.ZipFile(local_zip)
zip_ref.extractall("validation-horse-or-human")
zip_ref.close()

train_horse_dir = os.path.join("horse-or-human/horses")
train_human_dir = os.path.join("horse-or-human/humans")
validation_horse_dir = os.path.join("validation-horse-or-human/horses")
validation_human_dir = os.path.join("validation-horse-or-human/humans")

train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])
train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

validation_horse_names = os.listdir(validation_horse_dir)
print(validation_horse_names[:10])
validation_human_names = os.listdir(validation_human_dir)
print(validation_human_names[:10])

print("total training horse images:", len(os.listdir(train_horse_dir)))
print("total training human images:", len(os.listdir(train_human_dir)))
print("total validation horse images:", len(os.listdir(validation_horse_dir)))
print("total validation human images:", len(os.listdir(validation_human_dir)))


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

nrows, ncols = 4, 4
pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname)
                  for fname in train_horse_names[pic_index-8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname)
                  for fname in train_human_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_horse_pix + next_human_pix):
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis("Off")
    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation="relu", input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
	tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

print(model.summary())

model.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
              metrics=["accuracy"])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1/255.)
validation_datagen = ImageDataGenerator(rescale=1/255.)

train_generator = train_datagen.flow_from_directory(
    "horse-or-human",
    target_size=(300, 300),
    batch_size=128,
    class_mode="binary")

validation_generator = validation_datagen.flow_from_directory(
    "validation-horse-or-human",
    target_size=(300, 300),
    batch_size=32,
    class_mode="binary")

history = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=8)
