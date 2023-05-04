import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from transformers import pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Load the sentiment analysis model
nlp_model = pipeline('sentiment-analysis')

# Prepare the dataset for linear regression


def generate_dataset():
    images = [np.random.rand(224, 224, 3) for _ in range(1000)]
    confidences = []

    for img in images:
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        top_k = tf.keras.applications.mobilenet_v2.decode_predictions(
            predictions, top=1)
        confidences.append(top_k[0][0][2])

    avg_pixel_values = [np.mean(img) for img in images]
    return np.array(avg_pixel_values).reshape(-1, 1), np.array(confidences)


X, y = generate_dataset()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train the linear regression model
reg = LinearRegression().fit(X_train, y_train)

# Test the linear regression model
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Function to predict confidence based on the average pixel value


def predict_confidence(image):
    avg_pixel_value = np.mean(image)
    confidence = reg.predict([[avg_pixel_value]])
    return confidence[0]


def recognize_image(file_path):
    img = Image.open(file_path)
    img = img.resize((224, 224))  # Resize the image to 224x224 pixels
    img_array = np.array(img)  # Convert the image to a NumPy array
    # Normalize the pixel values to the range [0, 1]
    img_array = img_array / 255.0
    # Add an extra dimension for the batch
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    top_k = tf.keras.applications.mobilenet_v2.decode_predictions(
        predictions, top=5)

    return top_k[0]


def analyze_sentiment(text):
    return nlp_model(text)


def browse_image():
    file_path = filedialog.askopenfilename(title='Select an image')
    if file_path:
        result_text.delete('1.0', tk.END)
        img = Image.open(file_path)
        img = img.resize((224, 224))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

        result = recognize_image(file_path)
        for _, label, confidence in result:
            sentiment = analyze_sentiment(label)
            sentiment_str = sentiment[0]['label'].lower()
            result_text.insert(
                tk.END, f'{label}: {confidence * 100:.2f}% ({sentiment_str})\n')

        img_array = np.array(Image.open(file_path).resize((224, 224))) / 255.0
        predicted_confidence = predict_confidence(img_array)
        result_text.insert(
            tk.END, f'\nPredicted confidence for top label: {predicted_confidence * 100:.2f}%')
        result_text.insert(
            tk.END, f'\nLinear regression model performance:\nMSE: {mse:.5f}\nR2 score: {r2:.5f}')


root = tk.Tk()
root.title('Image Recognition with NLP and Supervised Learning')

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

image_label = tk.Label(frame)
image_label.pack()

browse_button = tk.Button(frame, text='Browse Image', command=browse_image)
browse_button.pack(pady=5)

result_text = tk.Text(frame, wrap='word', height=12, width=60)
result_text.pack(pady=5)

root.mainloop()
