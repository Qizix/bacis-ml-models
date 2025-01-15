import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image, ImageOps
import io
import base64

app = Flask(__name__)

# Load the saved model
try:
    model = tf.keras.models.load_model('mnist_cnn_model.keras')
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Get the image data from the request
    image_data = request.json['image'].split(',')[1]
    
    # Decode the base64 image
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # Preprocess the image
    image = image.convert('L')  # Convert to grayscale
    
    # Auto-scale the image
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
    
    # Add padding to make it square
    size = max(image.size)
    new_image = Image.new('L', (size, size), 255)
    new_image.paste(image, ((size - image.size[0]) // 2, (size - image.size[1]) // 2))
    image = new_image
    
    image = image.resize((28, 28), Image.LANCZOS)  # Resize to 28x28
    image = ImageOps.invert(image)  # Invert colors
    image = ImageOps.autocontrast(image, cutoff=10)  # Enhance contrast
    
    # Convert to numpy array and normalize
    image_array = np.array(image).reshape(1, 28, 28, 1).astype('float32') / 255.0

    # Make prediction
    prediction = model.predict(image_array)
    
    # Convert prediction to list and round to 4 decimal places
    probabilities = [round(float(p), 4) for p in prediction[0]]

    return jsonify({'probabilities': probabilities})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)