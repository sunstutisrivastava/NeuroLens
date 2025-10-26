import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load your model
model = tf.keras.models.load_model("checkpoints/best_image_model.h5")

# Path to the test image
img_path = "test.jpg"  # replace with your actual test image path

# Preprocess image
img = image.load_img(img_path, target_size=(48, 48))  # assuming 48x48 grayscale or color
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Predict emotion
predictions = model.predict(img_array)
class_labels = ["angry", "disgust", "fear", "happy"]

predicted_class = class_labels[np.argmax(predictions)]
print(f"Predicted Emotion: {predicted_class}")
