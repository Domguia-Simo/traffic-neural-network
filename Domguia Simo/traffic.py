import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Constants
EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    data_dir = sys.argv[1]

    # Load data
    images, labels = load_data(data_dir)

    if not images or not labels:
        sys.exit("Error: No data found. Please check your dataset path and structure.")

    # Convert labels to categorical format
    labels = tf.keras.utils.to_categorical(labels, num_classes=NUM_CATEGORIES)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get compiled model
    model = get_model()

    # Train the model
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate model performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model if filename is provided
    if len(sys.argv) == 3:
        model.save(sys.argv[2])
        print(f"Model saved to {sys.argv[2]}.")

def load_data(data_dir):
    """
    Load images and labels from `data_dir`, specifically handling .ppm images.

    Returns:
        images: List of images as numpy arrays.
        labels: List of corresponding category labels.
    """
    images, labels = [], []

    # Check if directory exists
    if not os.path.exists(data_dir):
        sys.exit(f"Error: Directory '{data_dir}' does not exist.")

    # Loop through category directories (0 to NUM_CATEGORIES-1)
    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))

        # Skip missing categories
        if not os.path.exists(category_path):
            print(f"Warning: Category {category} directory missing, skipping.")
            continue

        # Process only .ppm image files
        for filename in os.listdir(category_path):
            if not filename.endswith(".ppm"):  # Ensure only .ppm files are processed
                continue

            image_path = os.path.join(category_path, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Warning: Could not read {image_path}, skipping.")
                continue

            # Resize and store the image
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            images.append(image)
            labels.append(category)

    return images, labels

def get_model():
    """
    Returns a compiled CNN model for traffic sign classification.
    """
    model = tf.keras.models.Sequential([
        # First convolutional layer
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Second convolutional layer
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Third convolutional layer
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

        # Flatten layer
        tf.keras.layers.Flatten(),

        # Dense layers
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        # Output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    main()
