import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

# Load the pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet')


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
            result_text.insert(tk.END, f'{label}: {confidence * 100:.2f}%\n')


root = tk.Tk()
root.title('Image Recognition')

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

image_label = tk.Label(frame)
image_label.pack()

browse_button = tk.Button(frame, text='Browse Image', command=browse_image)
browse_button.pack(pady=5)

result_text = tk.Text(frame, wrap='word', height=6, width=40)
result_text.pack(pady=5)

root.mainloop()
