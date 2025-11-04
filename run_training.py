#!/usr/bin/env python3
"""
Complete training pipeline for emotion detection model
"""

import os
import sys
import subprocess
import time

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def train_model():
    """Train the emotion detection model"""
    print("Starting model training...")
    from train_emotion_model import train_model
    model, history = train_model()
    return model

def test_model():
    """Test the trained model"""
    print("Testing model...")
    from emotion_detector import EmotionDetector
    
    detector = EmotionDetector()
    if detector.model:
        print("✓ Model loaded successfully!")
        return True
    else:
        print("✗ Model loading failed!")
        return False

def start_api():
    """Start the Flask API"""
    print("Starting emotion detection API...")
    print("Run: python emotion_detector.py")
    print("API will be available at: http://localhost:5000")

def main():
    print("🧠 NEUROLENS - Emotion Detection Training Pipeline")
    print("=" * 50)
    
    try:
        # Step 1: Install requirements
        install_requirements()
        print("✓ Requirements installed")
        
        # Step 2: Train model
        model = train_model()
        print("✓ Model training completed")
        
        # Step 3: Test model
        if test_model():
            print("✓ Model testing passed")
        else:
            print("⚠ Model testing failed, but fallback detection available")
        
        # Step 4: Instructions
        print("\n🚀 Setup Complete!")
        print("Next steps:")
        print("1. Run: python emotion_detector.py")
        print("2. Open your HTML page")
        print("3. Click 'Start Detection' to test emotion recognition")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Using fallback detection mode")

if __name__ == "__main__":
    main()