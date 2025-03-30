import sys
import numpy as np
import cv2
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

# Constants
IMG_WIDTH = 30
IMG_HEIGHT = 30

class TrafficSignRecognizer:
    def __init__(self, root, model_file):
        self.root = root
        self.root.title("Traffic Sign Recognition")
        self.model = tf.keras.models.load_model(model_file)
        
        self.label = Label(root, text="Select an image to predict", font=("Arial", 14))
        self.label.pack(pady=10)
        
        self.image_label = Label(root)
        self.image_label.pack()
        
        self.predict_button = Button(root, text="Select Image", command=self.load_image)
        self.predict_button.pack(pady=10)
        
        self.result_label = Label(root, text="", font=("Arial", 12))
        self.result_label.pack()
    
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.ppm;*.png;*.jpg;*.jpeg")])
        if not file_path:
            return
        
        image = cv2.imread(file_path)
        if image is None:
            self.result_label.config(text="Error: Could not read the image file.")
            return
        
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = image.astype("float32") / 255.0  # Normalize
        self.predict(image, file_path)
    
    def predict(self, image, file_path):
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        prediction = self.model.predict(image)
        predicted_category = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        img = Image.open(file_path)
        img = img.resize((200, 200), Image.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
        img = ImageTk.PhotoImage(img)
        
        self.image_label.config(image=img)
        self.image_label.image = img
        
        self.result_label.config(text=f"Predicted Category: {predicted_category}\nConfidence: {confidence:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python recognition.py model.h5")
    
    model_file = sys.argv[1]
    root = tk.Tk()
    app = TrafficSignRecognizer(root, model_file)
    root.mainloop()
