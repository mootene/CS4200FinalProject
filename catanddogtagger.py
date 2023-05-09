import numpy as np
import tensorflow as tf
from tkinter import filedialog, messagebox, Tk, Label, Button, Entry
from PIL import Image, ImageTk
from transformers import AutoTokenizer

# Load the saved model
model = tf.keras.models.load_model('cat_dog_other_model.h5')

def preprocess_image(file_path):
    img = Image.open(file_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_text(text):
    tokens = tokenizer(text, return_tensors="tf", padding=True, truncation=True)
    return {k: tf.squeeze(v, axis=0) for k, v in tokens.items()}

def predict_image(file_path):
    img_array = preprocess_image(file_path)
    prediction = model.predict(img_array)
    return np.argmax(prediction, axis=1)[0]


def browse_image():
    file_path = filedialog.askopenfilename(title='Select an image')
    if file_path:
        img = Image.open(file_path)
        img = img.resize((224, 224))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

        prediction = predict_image(file_path)
        if prediction == 0:
            description_entry.delete(0, 'end')
            description_entry.insert(0, "The image is a cat.")
        elif prediction == 1:
            description_entry.delete(0, 'end')
            description_entry.insert(0, "The image is a dog.")
        else:
            description_entry.delete(0, 'end')
            description_entry.insert(0, "Not a cat or a dog.")


root = Tk()
root.title('Cat and Dog Image Classifier')

frame = Label(root)
frame.pack(padx=10, pady=10)

image_label = Label(frame)
image_label.pack()

browse_button = Button(frame, text='Browse Image', command=browse_image)
browse_button.pack(pady=5)

description_label = Label(frame, text="Image description:")
description_label.pack(pady=5)

description_entry = Entry(frame, width=50)
description_entry.pack(pady=5)

root.mainloop()
