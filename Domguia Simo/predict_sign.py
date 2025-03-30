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

# Accurate GTSRB Labels (Complete List)
CATEGORY_LABELS = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "Entry prohibited",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signal",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}

class TrafficSignRecognizer:
    def __init__(self, root, model_file):
        self.root = root
        self.root.title("Traffic Sign Recognition")
        
        try:
            self.model = tf.keras.models.load_model(model_file)
            print("Model loaded successfully!")
        except Exception as e:
            sys.exit(f"Error loading model: {str(e)}")
        
        self.label = Label(root, text="Select an image to predict", font=("Arial", 14))
        self.label.pack(pady=10)
        
        self.image_label = Label(root)
        self.image_label.pack()
        
        self.predict_button = Button(root, text="Select Image", command=self.load_image)
        self.predict_button.pack(pady=10)
        
        self.result_label = Label(root, text="", font=("Arial", 12))
        self.result_label.pack()
    
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.ppm;*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.tiff")])
        if not file_path:
            return
        
        try:
            img = Image.open(file_path).convert('RGB')  # Ensure image is in RGB format
            image = np.array(img)
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            image = image.astype("float32") / 255.0  # Normalize
            image = np.expand_dims(image, axis=0)  # Ensure proper input shape
            
            self.predict(image, file_path)
        except Exception as e:
            self.result_label.config(text=f"Error: {str(e)}")
    
    def predict(self, image, file_path):
        try:
            prediction = self.model.predict(image)
            predicted_category = int(np.argmax(prediction))
            confidence = float(np.max(prediction) * 100)
            
            category_label = CATEGORY_LABELS.get(predicted_category, "Unknown Sign")
            
            print("Raw Prediction Values:", prediction)
            print(f"Predicted Category: {predicted_category}, Confidence: {confidence:.2f}%")
            
            img = Image.open(file_path)
            img = img.resize((200, 200), Image.LANCZOS)  # Resize for display
            img = ImageTk.PhotoImage(img)
            
            self.image_label.config(image=img)
            self.image_label.image = img
            
            self.result_label.config(text=f"Predicted Category: {category_label} ({predicted_category})\nConfidence: {confidence:.2f}%")
        except Exception as e:
            self.result_label.config(text=f"Prediction Error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python recognition.py model.h5")
    
    model_file = sys.argv[1]
    root = tk.Tk()
    app = TrafficSignRecognizer(root, model_file)
    root.mainloop()
