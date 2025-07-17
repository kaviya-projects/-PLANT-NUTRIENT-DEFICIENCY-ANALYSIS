import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('/content/drive/MyDrive/plant_nutrient_model.h5')
img_height, img_width = 150, 150
class_labels = ['Nitrogen_Deficiency', 'Phosphorus_Deficiency', 'Potassium_Deficiency', 'Healthy']

# Function to load and preprocess the input image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict nutrient deficiency
def predict_nutrient_deficiency(image_path):
    img_array = preprocess_image(image_path)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
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
image_path = '/N_Def_439.jpg'
predict_nutrient_deficiency(image_path)
