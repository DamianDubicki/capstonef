from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from PIL import Image
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Load the label encoder
label_encoder = joblib.load("label_encoder.pkl")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            app.logger.error("No file part")
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            app.logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400

        if not allowed_file(file.filename):
            app.logger.error("Unsupported file type")
            return jsonify({'error': 'Unsupported file type'}), 400

        # Process the image
        pil_image = Image.open(file.stream).convert('RGB')
        pil_image = pil_image.resize((224, 224))
        image_array = np.expand_dims(np.array(pil_image), axis=0)
        image_array = image_array.astype(np.float32) / 127.5 - 1  # Equivalent to preprocess_input

        # Prepare input for TensorFlow Lite model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()

        # Get predictions
        predictions = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_index = np.argmax(predictions)

        # Get predicted artwork name
        predicted_artwork = label_encoder.inverse_transform([predicted_class_index])[0]

        return jsonify({'artwork': predicted_artwork})
    
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
