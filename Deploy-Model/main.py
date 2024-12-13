import os
import logging
import sys
from google.cloud import storage
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'capstone-project-442216-2278d0dc8f4c.json'

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import StandardScaler
import string
import re
from datetime import datetime
from werkzeug.utils import secure_filename

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# Configurations
UPLOAD_FOLDER = '/tmp/uploads'
BUCKET_NAME = 'capstone-upload-images'  # Ganti dengan nama bucket GCS Anda
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CLASSIFIER_PATH = os.path.join(os.getcwd(), 'question_classifier_model.h5')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_to_gcs(file, bucket_name):
    """Upload file to Google Cloud Storage"""
    try:
        logger.info(f"Starting upload to bucket: {bucket_name}")
        storage_client = storage.Client()
        
        # Log available buckets
        buckets = list(storage_client.list_buckets())
        logger.info(f"Available buckets: {[b.name for b in buckets]}")
        
        bucket = storage_client.bucket(bucket_name)
        if not bucket.exists():
            logger.error(f"Bucket {bucket_name} does not exist")
            return None
            
        blob_name = f"math_questions/{file.filename}"
        blob = bucket.blob(blob_name)
        
        # Reset file pointer
        file.seek(0)
        
        # Upload file
        logger.info(f"Uploading file: {file.filename}")
        blob.upload_from_file(file)
        
        # Get public URL
        public_url = blob.public_url
        logger.info(f"Upload successful. URL: {public_url}")
        return public_url
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return None

def ensure_required_files_exist():
    required_files = [
        CLASSIFIER_PATH,
        'expert.xlsx',
    ]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")

class ModelState:
    def __init__(self):
        self.question_classifier_model = None
        self.preprocessor = None
        self.experts_df = None
        self.is_initialized = False

    def initialize(self):
        try:
            logger.info("Starting model initialization...")
            
            # Create upload directory
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            
            # Initialize preprocessor
            logger.info("Initializing preprocessor")
            self.preprocessor = DataPreprocessor()
            
            # Load classifier
            logger.info(f"Loading classifier from {CLASSIFIER_PATH}")
            self.question_classifier_model = tf.keras.models.load_model(CLASSIFIER_PATH)
            
            # Load expert data
            logger.info("Loading expert data")
            self.experts_df = pd.read_excel('expert.xlsx')
            
            self.is_initialized = True
            logger.info("Model initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            self.is_initialized = False
            return False

    def ensure_initialized(self):
        if not self.is_initialized:
            return self.initialize()
        return True

# Initialize global state
model_state = ModelState()

class DataPreprocessor:
    def __init__(self):
        self.features = None
        self.scaler = StandardScaler()
        
    def extract_text_features(self, text):
        text = str(text).lower()
        feature_vector = np.zeros(502)
        
        char_count = len(text)
        word_count = len(text.split())
        punct_count = sum([1 for char in text if char in string.punctuation])
        number_count = sum([1 for char in text if char.isdigit()])
        words = text.split()
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        special_char_count = sum([1 for char in text if not char.isalnum() and char not in string.punctuation])
        uppercase_count = sum([1 for char in str(text) if char.isupper()])
        sentence_count = len(re.split(r'[.!?]+', text))
        
        feature_vector[0:8] = [char_count, word_count, punct_count, number_count, 
                             avg_word_length, special_char_count, uppercase_count, sentence_count]
        
        char_to_index = {chr(i): i-8 for i in range(32, 127)}
        for i, char in enumerate(text):
            if char in char_to_index and char_to_index[char] < 494:
                feature_vector[8 + char_to_index[char]] = 1
        
        return feature_vector
        
    def preprocess_new_data(self, question_data):
        features = self.extract_text_features(question_data['Problem'])
        processed_data = pd.DataFrame([features])
        
        if hasattr(self, 'scaler') and hasattr(self.scaler, 'mean_'):
            processed_data = pd.DataFrame(
                self.scaler.transform(processed_data),
                columns=processed_data.columns
            )
        
        return processed_data

class QuestionClassifier:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.difficulty_labels = ['Easy', 'Medium', 'Hard']
        
    def predict(self, question_text):
        try:
            question_data = {'Problem': question_text}
            processed_data = self.preprocessor.preprocess_new_data(question_data)
            prediction = self.model.predict(processed_data)
            predicted_class = np.argmax(prediction)
            confidence = float(np.max(prediction))
            
            return {
                'difficulty': self.difficulty_labels[predicted_class],
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {'error': str(e)}

def get_expert_recommendations(experts_df, top_n=3):
    try:
        expertise_list = experts_df['expertise'].tolist()
        if expertise_list:
            unique_expertise = list(set(expertise_list))
            recommendations = unique_expertise[:top_n]
            logger.info(f"Found {len(recommendations)} recommendations")
            return recommendations
        return []
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        return []

@app.route('/')
def index():
    status = {
        "status": "Server is running",
        "server_time": datetime.now().isoformat(),
        "model_status": "initialized" if model_state.is_initialized else "not initialized",
        "endpoints": {
            "health": "/health",
            "process": "/process"
        }
    }
    return jsonify(status)

@app.route('/process', methods=['POST'])
def process():
    try:
        # Ensure models are initialized
        if not model_state.ensure_initialized():
            return jsonify({"error": "Failed to initialize models"}), 500

        # Handle text input
        recognized_text = request.form.get('text', '')
        logger.info(f"Received text input: {recognized_text}")

        # Handle image upload
        gcs_url = None
        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                # Upload to GCS
                gcs_url = upload_to_gcs(file, BUCKET_NAME)
                if not gcs_url:
                    return jsonify({"error": "Failed to upload image"}), 500

        # Validate input
        if not recognized_text and not gcs_url:
            return jsonify({"error": "No input provided"}), 400

        # Perform classification if text is provided
        classification_result = None
        if recognized_text:
            classifier = QuestionClassifier(
                model_state.question_classifier_model,
                model_state.preprocessor
            )
            classification_result = classifier.predict(recognized_text)

        # Get expert recommendations
        recommendations = get_expert_recommendations(model_state.experts_df)

        # Add forum link
        forum_link = "https://histudent-web.web.app/"  # Ganti dengan URL forum Anda

        # Construct response
        response = {
            'text': recognized_text,
            'image_url': gcs_url,
            'classification': classification_result,
            'recommendations': recommendations,
            'forum_link': forum_link
        }
        return jsonify(response)

    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy" if model_state.is_initialized else "initializing",
        "models_loaded": model_state.is_initialized,
        "preprocessor_status": model_state.preprocessor is not None
    })

@app.route('/test-upload', methods=['POST'])
def test_upload():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file and allowed_file(file.filename):
            logger.info(f"Processing file: {file.filename}")
            gcs_url = upload_to_gcs(file, BUCKET_NAME)
            
            if gcs_url is None:
                return jsonify({
                    'error': 'Upload failed, check server logs for details'
                }), 500
                
            return jsonify({
                'success': True,
                'message': 'Upload successful',
                'url': gcs_url,
                'filename': file.filename
            })
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        logger.error(f"Upload test error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting application")
    try:
        ensure_required_files_exist()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
        
    if not model_state.initialize():
        logger.error("Failed to initialize models")
        sys.exit(1)
        
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)