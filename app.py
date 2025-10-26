import os
from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import librosa
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Fake Models (replace with your own) ---
emotions_image = ["angry", "disgust", "fear", "happy"]
emotions_voice = ["euphoric", "sad", "joyful", "surprised"]

def predict_image(filepath):
    # TODO: replace with your trained image model
    # Right now: return random class
    return np.random.choice(emotions_image)

def predict_voice(filepath):
    # Load audio
    y, sr = librosa.load(filepath, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.mean(mfccs.T, axis=0)

    # TODO: replace with your trained voice model
    # Right now: return random class
    return np.random.choice(emotions_voice)

# --- Routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze_image", methods=["POST"])
def analyze_image():
    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(filepath)
    emotion = predict_image(filepath)
    return jsonify({"emotion": emotion})

@app.route("/analyze_voice", methods=["POST"])
def analyze_voice():
    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(filepath)
    emotion = predict_voice(filepath)
    return jsonify({"emotion": emotion})

if __name__ == "__main__":
    app.run(debug=True)
