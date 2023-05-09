import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam



# Constants
image_folder = 'E:\Documents\School\Spring 2023\CS 4200 Artificial Intelligence\Project\Images\/train'  # Replace with your images folder
train_ratio = 0.8
image_size = 224
batch_size = 32
epochs = 50

# Read image filenames
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
random.shuffle(image_files)

# Split the data into training and test sets
train_size = int(train_ratio * len(image_files))
train_files = image_files[:train_size]
test_files = image_files[train_size:]

# Create ImageDataGenerators for training and test data
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1.0/255)

# Define a generator function to yield (image, label) pairs


def generator(file_list, datagen):
    while True:
        random.shuffle(file_list)
        for i in range(0, len(file_list), batch_size):
            batch_files = file_list[i:i+batch_size]
            images = []
            labels = []
            for file in batch_files:
                img = tf.keras.preprocessing.image.load_img(os.path.join(
                    image_folder, file), target_size=(image_size, image_size))
                img = tf.keras.preprocessing.image.img_to_array(img)
                img = datagen.random_transform(img)
                label = [1, 0, 0] if file.startswith('cat') else (
                    [0, 1, 0] if file.startswith('dog') else [0, 0, 1])
                images.append(img)
                labels.append(label)
            images = np.array(images)
            labels = np.array(labels)
            yield images, labels


train_generator = generator(train_files, train_datagen)
test_generator = generator(test_files, test_datagen)

# Load the pre-trained MobileNetV2 model (without the top layers)
base_model = MobileNetV2(
    weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers (we don't want to retrain them)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=epochs, steps_per_epoch=len(train_files)//batch_size, validation_data=test_generator, validation_steps=len(test_files)//batch_size)

# Save the model
model.save('cat_dog_other_model.h5')

# Function to predict the class of an image
def predict_image(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(image_size, image_size))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255

    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    
    if class_index == 0:
        return "cat"
    elif class_index == 1:
        return "dog"
    else:
        return "not a cat or dog"

# Test the function with an example image
image_path = "E:\Documents\School\Spring 2023\CS 4200 Artificial Intelligence\Project\Images\/train/cat.144.jpg"  # Replace with the path to your test image
print(predict_image(image_path, model))

