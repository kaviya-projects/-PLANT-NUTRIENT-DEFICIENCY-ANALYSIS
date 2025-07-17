import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained model
model = tf.keras.models.load_model('/content/drive/MyDrive/plant_nutrient_model.h5')

# Define image size (should match the input size used during model training)
img_height, img_width = 150, 150

# Class labels (adjust these based on your dataset)
class_labels = ['Nitrogen_Deficiency', 'Phosphorus_Deficiency', 'Potassium_Deficiency', 'Healthy']

# Function to load and preprocess the input image
def preprocess_image(image_path):
    # Load the image with target size
    img = load_img(image_path, target_size=(img_height, img_width))

    # Convert image to array
    img_array = img_to_array(img)

    # Rescale pixel values (just like in training)
    img_array = img_array / 255.0

    # Expand dimensions to match the input shape for the model
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# Function to predict nutrient deficiency
def predict_nutrient_deficiency(image_path):
    # Preprocess the input image
    img_array = preprocess_image(image_path)

    # Make prediction
    predictions = model.predict(img_array)

    # Get the predicted class (highest probability)
    predicted_class_index = np.argmax(predictions[0])

    # Get the predicted class label
    predicted_label = class_labels[predicted_class_index]

    # Print prediction confidence
    confidence = np.max(predictions[0]) * 100
    print(f"Predicted Class: {predicted_label} ({confidence:.2f}% confidence)")

    # Show the input image
    img = load_img(image_path)
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_label} ({confidence:.2f}% confidence)")
    plt.axis('off')
    plt.show()

# Example Usage
image_path = '/001.jpeg'
predict_nutrient_deficiency(image_path)
