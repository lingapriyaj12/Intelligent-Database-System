import warnings
warnings.filterwarnings('ignore')
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model #noqa
from tensorflow.keras.preprocessing import image #noqa
import numpy as np
import os
from PIL import Image
import cv2

# Initialize Flask app
app = Flask(__name__)

# Load your trained model (ensure you save your model after training)
model = load_model('pneumonia_model.h5')

# Define allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Save the uploaded image temporarily
        img_path = os.path.join('uploads', file.filename)
        file.save(img_path)

        # Process the image for prediction
        img = Image.open(img_path)
        img = img.resize((150, 150))  # Resize image to the size expected by the model
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Prediction using the model
        prediction = model.predict(img_array)
        prediction_class = 'Pneumonia' if prediction[0][0] > 0.5 else 'Normal'

        return jsonify({'prediction': prediction_class})

    return jsonify({'error': 'Invalid file type'}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
