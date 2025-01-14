import tensorflow as tf
import os
import pandas as pd
from keras.utils import load_img, img_to_array
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Constants
IMAGE_SIZE = 224
testDirectory = "/Users/shagunbhatia30/Downloads/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/test"

# Dataset and Model Paths
DatasetPath = "/Users/shagunbhatia30/Downloads/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/new"
trainPath = DatasetPath + "/train"
best_model_file = "/Users/shagunbhatia30/PycharmProjects/vlg/best_model.keras"

# Load Model
CLASSES = sorted(os.listdir(trainPath))  # Sorting ensures consistent ordering of class indices
try:
    model = tf.keras.models.load_model(best_model_file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

print(model.summary())


# Image Preparation Function
def prepareImage(pathForImage):
    image = load_img(pathForImage, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    imgResult = img_to_array(image)
    imgResult = np.expand_dims(imgResult, axis=0)
    imgResult = preprocess_input(imgResult)  # MobileNetV2 specific preprocessing
    return imgResult


# Process All Test Images
results = []
batch_size = 32  # You can adjust this based on your memory capacity
batch_images = []
batch_files = []

for file_name in os.listdir(testDirectory):
    file_path = os.path.join(testDirectory, file_name)
    if os.path.isfile(file_path) and file_name.lower().endswith(('.jpg', '.png')):
        print(f"Processing {file_name}")
        imgForModel = prepareImage(file_path)
        batch_images.append(imgForModel)
        batch_files.append(file_name)

        # Process in batches
        if len(batch_images) == batch_size:
            batch_images_np = np.vstack(batch_images)
            predictions = model.predict(batch_images_np, verbose=0)
            for idx, pred in enumerate(predictions):
                predicted_index = np.argmax(pred, axis=0)
                predicted_class = CLASSES[predicted_index]
                results.append({'image_id': batch_files[idx], 'class': predicted_class})

            # Reset batch
            batch_images = []
            batch_files = []

# Handle any remaining images in the batch
if batch_images:
    batch_images_np = np.vstack(batch_images)
    predictions = model.predict(batch_images_np, verbose=0)
    for idx, pred in enumerate(predictions):
        predicted_index = np.argmax(pred, axis=0)
        predicted_class = CLASSES[predicted_index]
        results.append({'image_id': batch_files[idx], 'class': predicted_class})

# Check if results list has any predictions
print(f"Total results: {len(results)}")

# Save Results to CSV
if results:
    output_csv_path = os.path.join(os.getcwd(), "predictions.csv")  # Saving to current working directory
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")
else:
    print("No predictions to save.")
