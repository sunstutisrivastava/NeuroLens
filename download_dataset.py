import os
import requests
import zipfile
from pathlib import Path

def download_fer2013():
    """Download FER2013 dataset from Kaggle"""
    print("Downloading FER2013 dataset...")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Download dataset (you need to get this from Kaggle)
    url = "https://www.kaggle.com/datasets/msambare/fer2013"
    print(f"Please download FER2013 dataset from: {url}")
    print("Extract it to 'data/fer2013/' folder")
    
    # Alternative: Create sample data structure
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    for split in ['train', 'test']:
        for emotion in emotions:
            path = f"data/fer2013/{split}/{emotion}"
            os.makedirs(path, exist_ok=True)
    
    print("Dataset structure created. Please add images to respective folders.")

if __name__ == "__main__":
    download_fer2013()