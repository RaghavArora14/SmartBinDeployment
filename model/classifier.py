import tensorflow as tf
import numpy as np
import os
from pathlib import Path

class BiodegradableClassifier:
    def __init__(self, model_path):
        """Initialize the classifier with a saved model."""
        # Configure GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")

        # Load the model
        try:
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.model.compile(
                optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy']
            )
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {e}")

    def preprocess_image(self, image):
        """Preprocess the image for prediction."""
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.expand_dims(image, 0)
        return image

    def predict_image(self, image_path):
        """Make a prediction on a single image."""
        try:
            img = tf.io.read_file(image_path)
            img = tf.image.decode_image(img, channels=3)
            processed_img = self.preprocess_image(img)
            
            with tf.device('/GPU:0'):
                prediction = self.model(processed_img, training=False)
            
            probability = float(prediction[0][0])
            is_biodegradable = probability > 0.5

            return {
                'probability': probability,
                'is_biodegradable': is_biodegradable,
                'class_label': 'Biodegradable' if is_biodegradable else 'Non-biodegradable',
                'confidence': probability if is_biodegradable else (1 - probability)
            }
        except Exception as e:
            raise Exception(f"Error during prediction: {e}")
