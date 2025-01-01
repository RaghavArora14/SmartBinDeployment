import os
import sqlite3
import matplotlib.pyplot as plt
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path
import io
import base64
from PIL import Image
import numpy as np
import cv2
from collections import Counter
import datetime

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DATABASE = 'classifications.db'

# Initialize TensorFlow model
class BiodegradableClassifier:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path, compile=False)

    @tf.function
    def preprocess_image(self, image):
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.expand_dims(image, 0)
        return image

    def predict_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3)
        processed_img = self.preprocess_image(img)
        prediction = self.model(processed_img, training=False)
        probability = float(prediction[0][0])
        return 'Biodegradable' if not probability > 0.5 else 'Non-biodegradable'

model = BiodegradableClassifier('model/biodegradable_classifier.keras')

def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    classification TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )''')
    conn.commit()
    conn.close()

def get_classification_data():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT timestamp, classification FROM results ORDER BY timestamp DESC LIMIT 6")
    recent_data = c.fetchall()
    c.execute("SELECT classification FROM results")
    all_data = [row[0] for row in c.fetchall()]
    conn.close()
    return recent_data, all_data

def get_db():
    conn = sqlite3.connect('classifications.db')
    conn.row_factory = sqlite3.Row
    return conn

def generate_pie_chart(data):
    counts = Counter(data)
    labels, sizes = zip(*counts.items()) if counts else ([], [])
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def get_usage_data():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT DATE(timestamp) FROM results")
    dates = [row[0] for row in c.fetchall()]
    conn.close()
    counts = Counter(dates)
    labels = sorted(counts.keys())
    data = [counts[date] for date in labels]
    return labels, data

def generate_line_graph(labels, data):
    fig, ax = plt.subplots()
    ax.plot(labels, data, marker='o', linestyle='-')
    ax.set_title("Waste Trends Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Wastes")
    plt.xticks(rotation=45)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def clear_database():
    try:
        conn = sqlite3.connect(DATABASE) # Update with your database file
        cursor = conn.cursor()
        cursor.execute("DELETE FROM results")  # Replace 'waste_data' with your table name
        conn.commit()
        conn.close()
        print("Database cleared successfully.")
    except Exception as e:
        print(f"Error clearing the database: {e}")

# Function to clear the upload folder
def clear_upload_folder():
    try:
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
        print("Upload folder cleared successfully.")
    except Exception as e:
        print(f"Error clearing the upload folder: {e}")

# Reset route
@app.route('/reset', methods=['POST'])
def reset():
    clear_database()
    clear_upload_folder()
    return redirect(url_for('index'))

@app.route('/')
def index():
    recent_classifications, all_data = get_classification_data()
    labels, usage_data = get_usage_data()
    pie_chart = generate_pie_chart(all_data) if all_data else None
    bin_data=bin_capacity()
    line_graph = generate_line_graph(labels, usage_data) if labels else None
    return render_template('index.html',
                           bin_data=bin_data,pie_chart=pie_chart,
                           line_graph=line_graph,
                           recent_classifications=recent_classifications,
                           labels=labels,
                           usage_data=usage_data)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Classify the uploaded image
    classification = model.predict_image(file_path)

    # Convert the image to base64 for display
    with open(file_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    # Store result in the database
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("INSERT INTO results (classification) VALUES (?)", (classification,))
    conn.commit()

    # Return JSON response instead of rendering an HTML page
    if request.headers.get('Accept') == 'application/json':
        return jsonify({"classification": classification})

    # Default behavior: render the result page
    return render_template('result.html',
                           image_data=encoded_image,
                           classification=classification)

@app.route('/update_capacity', methods=['POST'])
def update_capacity():
    data = request.get_json()

    # Validate the input data
    bio_fill_level = data.get('bio_fill_level')
    non_bio_fill_level = data.get('non_bio_fill_level')

    if bio_fill_level is None or non_bio_fill_level is None:
        return jsonify({"error": "Invalid data, bio_fill_level and non_bio_fill_level are required"}), 400

    # Store the data in the database
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO bin_capacity (bio_fill_level, non_bio_fill_level)
            VALUES (?, ?)
        ''', (bio_fill_level, non_bio_fill_level))
        conn.commit()
        conn.close()
        return jsonify({"message": "Bin capacity updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Error updating bin capacity: {str(e)}"}), 500


@app.route('/status')
def status():
    return jsonify({"online": True})

def bin_capacity():
    try:
        # Connect to the database
        conn = get_db()
        cursor = conn.cursor()
        
        # Fetch the most recent bin capacity data
        cursor.execute('SELECT bio_fill_level, non_bio_fill_level, timestamp FROM bin_capacity ORDER BY timestamp DESC LIMIT 1')
        data = cursor.fetchone()  # Get the most recent record
        
        # If no data is found, set the data to None
        if not data:
            data = {'bio_fill_level': 0, 'non_bio_fill_level': 0, 'timestamp': 'N/A'}
        
        conn.close()

        # Pass data to the template
        return data
    
    except Exception as e:
        return jsonify({"error": f"Error fetching bin capacity data: {str(e)}"}), 500


@app.route('/clear', methods=['POST'])
def clear_data():
    conn = sqlite3.connect(DATABASE)
    conn.execute("DELETE FROM results")
    conn.execute("DELETE FROM bin_capacity")
    conn.commit()
    conn.close()
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=8080)

