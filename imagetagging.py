# Import required libraries
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Load the image from a local file
file_path = 'F:\Downloads\catforproject.jpg'
img = Image.open(file_path)

# Preprocess the image
img = img.resize((224, 224))  # Resize the image to 224x224 pixels
img_array = np.array(img)  # Convert the image to a NumPy array
img_array = img_array / 255.0  # Normalize the pixel values to the range [0, 1]
# Add an extra dimension for the batch
img_array = np.expand_dims(img_array, axis=0)

# Perform image recognition
predictions = model.predict(img_array)

# Process predictions
top_k = tf.keras.applications.mobilenet_v2.decode_predictions(
    predictions, top=5)
for _, label, confidence in top_k[0]:
    print(f'{label}: {confidence * 100:.2f}%')
