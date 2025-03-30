import sys
import numpy as np
import cv2
import tensorflow as tf

# Constants
IMG_WIDTH = 30
IMG_HEIGHT = 30


def main():
    # Check command-line arguments
    if len(sys.argv) != 3:
        sys.exit("Usage: python recognition.py model.h5 image_file")
    
    model_file = sys.argv[1]
    image_file = sys.argv[2]
    
    # Load the trained model
    model = tf.keras.models.load_model(model_file)
    print("Model loaded successfully.")
    
    # Load and preprocess the image
    image = load_image(image_file)
    if image is None:
        sys.exit("Error: Could not read the image file.")
    
    # Predict category
    prediction = model.predict(np.array([image]))
    predicted_category = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    print(f"Predicted Category: {predicted_category} (Confidence: {confidence:.2f}%)")

def load_image(image_path):
    """
    Load and preprocess an image for prediction.
    Returns the image as a numpy array.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    return image


if __name__ == "__main__":
    main()
