import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import os

class EmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = None
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']
        self.class_indices = {}
        
        # Try to load the trained model
        model_paths = [
            'checkpoints/best_emotion_model.h5',
            'checkpoints/emotion_model.h5', 
            'checkpoints/best_image_model.h5',
            'emotion_model.h5'
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    self.model = load_model(model_path)
                    print(f"✅ Emotion model loaded from: {model_path}")
                    
                    # Load class indices if available
                    class_indices_path = 'checkpoints/class_indices.json'
                    if os.path.exists(class_indices_path):
                        with open(class_indices_path, 'r') as f:
                            self.class_indices = json.load(f)
                        # Reverse mapping for prediction
                        self.idx_to_emotion = {v: k for k, v in self.class_indices.items()}
                        print(f"✅ Class indices loaded: {self.class_indices}")
                    else:
                        # Use default mapping
                        self.idx_to_emotion = {i: emotion for i, emotion in enumerate(self.emotions)}
                    break
                except Exception as e:
                    print(f"Failed to load {model_path}: {e}")
                    continue
        
        if self.model is None:
            print("⚠️ No trained model found. Using pattern-based detection.")
            self.idx_to_emotion = {i: emotion for i, emotion in enumerate(self.emotions)}
    
    def preprocess_face(self, face_img):
        """Preprocess face image for emotion detection"""
        # Resize to 48x48 (standard for emotion detection)
        face_img = cv2.resize(face_img, (48, 48))
        
        # Ensure grayscale
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Normalize pixel values to [0, 1]
        face_img = face_img.astype('float32') / 255.0
        
        # Add batch and channel dimensions
        face_img = np.expand_dims(face_img, axis=0)  # Batch dimension
        face_img = np.expand_dims(face_img, axis=-1)  # Channel dimension
        
        return face_img
    
    def detect_emotion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with better parameters
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        emotions_detected = []
        
        for (x, y, w, h) in faces:
            # Extract face region
            roi_gray = gray[y:y+h, x:x+w]
            
            if self.model:
                try:
                    # Preprocess face
                    processed_face = self.preprocess_face(roi_gray)
                    
                    # Predict emotion
                    prediction = self.model.predict(processed_face, verbose=0)
                    emotion_idx = np.argmax(prediction[0])
                    confidence = float(prediction[0][emotion_idx])
                    
                    # Map index to emotion name
                    emotion = self.idx_to_emotion.get(emotion_idx, 'neutral')
                    
                    emotions_detected.append({
                        'emotion': emotion,
                        'confidence': confidence,
                        'bbox': [int(x), int(y), int(w), int(h)]
                    })
                    
                except Exception as e:
                    print(f"Prediction error: {e}")
                    # Fallback to pattern-based detection
                    emotion = self.pattern_based_detection(roi_gray)
                    emotions_detected.append({
                        'emotion': emotion,
                        'confidence': 0.7,
                        'bbox': [int(x), int(y), int(w), int(h)]
                    })
            else:
                # Pattern-based fallback detection
                emotion = self.pattern_based_detection(roi_gray)
                emotions_detected.append({
                    'emotion': emotion,
                    'confidence': 0.6,
                    'bbox': [int(x), int(y), int(w), int(h)]
                })
        
        return emotions_detected
    
    def pattern_based_detection(self, face_roi):
        """Simple pattern-based emotion detection as fallback"""
        h, w = face_roi.shape
        
        # Analyze different face regions
        mouth_region = face_roi[int(h*0.6):int(h*0.9), int(w*0.3):int(w*0.7)]
        eye_region = face_roi[int(h*0.2):int(h*0.5), int(w*0.2):int(w*0.8)]
        
        # Simple intensity analysis
        mouth_intensity = np.mean(mouth_region)
        eye_intensity = np.mean(eye_region)
        
        # Basic emotion classification based on intensity patterns
        if mouth_intensity > 120:
            return 'happy'
        elif mouth_intensity < 80:
            return 'sad'
        elif eye_intensity < 90:
            return 'angry'
        elif np.std(face_roi) > 50:
            return 'surprised'
        else:
            return 'neutral'

# Flask API for real-time detection
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64

app = Flask(__name__)
CORS(app)
detector = EmotionDetector()

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion_api():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect emotions
        emotions = detector.detect_emotion(frame)
        
        return jsonify({
            'success': True,
            'emotions': emotions,
            'dominant_emotion': emotions[0]['emotion'] if emotions else 'neutral'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)