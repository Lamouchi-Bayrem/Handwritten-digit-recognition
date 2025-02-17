import tensorflow as tf
from flask import Flask, render_template, request, jsonify
import numpy as np
import base64
import cv2
from PIL import Image
import io
import os

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Load the trained model
model_path = os.path.join("model", "tf-cnn-model.h5")
model = tf.keras.models.load_model(model_path)

def preprocess_image(image_data):
    """
    Convert base64 image data to a preprocessed image array.
    """
    # Decode base64 image
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Convert to grayscale

    # Resize to 28x28 (MNIST size)
    image = image.resize((28, 28))

    # Convert to NumPy array
    img_array = np.array(image)

    # Invert colors (black background, white digit)
    img_array = cv2.bitwise_not(img_array)

    # Normalize pixel values
    img_array = img_array / 255.0

    # Reshape to match model input
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    img_data = data['image']

    # Preprocess image
    processed_img = preprocess_image(img_data)

    # Predict using the model
    prediction = model.predict(processed_img)
    predicted_digit = int(np.argmax(prediction))  # Get the predicted class

    return jsonify({'prediction': predicted_digit})

if __name__ == '__main__':
    app.run(debug=True)
