from flask import Flask, request, jsonify
from classifier import BiodegradableClassifier
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your ML model
MODEL_PATH = 'model/biodegradable_classifier.keras'
classifier = BiodegradableClassifier(MODEL_PATH)

@app.route('/')
def home():
    return "Smart Dustbin Flask API is running!"

@app.route('/upload', methods=['POST'])
def upload_image():
    """Endpoint to upload an image and get classification result."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform prediction using your classifier
        try:
            result = classifier.predict_image(file_path)
            return jsonify({
                'class_label': result['class_label'],
                'probability': result['probability'],
                'confidence': result['confidence']
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'File upload failed'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
