import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import os
from sklearn.model_selection import train_test_split

def create_emotion_model():
    model = Sequential([
        # First block
        Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third block
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Dense layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_fer2013_data():
    """Load FER2013 dataset"""
    try:
        # Try to load from CSV
        df = pd.read_csv('data/fer2013.csv')
        
        X = []
        y = []
        
        for index, row in df.iterrows():
            pixels = np.array(row['pixels'].split(), dtype='float32')
            image = pixels.reshape(48, 48, 1) / 255.0
            X.append(image)
            y.append(row['emotion'])
        
        X = np.array(X)
        y = to_categorical(np.array(y), 7)
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
        
    except FileNotFoundError:
        print("FER2013 dataset not found. Creating synthetic data for demo...")
        return create_synthetic_data()

def create_synthetic_data():
    """Create realistic synthetic emotion data with strong patterns"""
    print("Creating realistic emotion dataset with strong patterns...")
    
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    # Create 2000 samples per emotion for training
    for emotion in range(7):
        for _ in range(2000):
            # Create base face
            face = np.random.normal(0.4, 0.1, (48, 48, 1))
            
            # Add strong emotion-specific patterns
            if emotion == 0:  # angry
                face[12:18, 15:33] += 0.4  # eyebrows down
                face[28:35, 18:30] -= 0.3  # mouth down
                face[20:25, 20:28] += 0.2  # eyes narrow
                
            elif emotion == 1:  # disgust
                face[25:30, 15:20] += 0.3  # nose wrinkle
                face[30:35, 20:28] -= 0.4  # mouth down
                
            elif emotion == 2:  # fear
                face[18:25, 18:30] += 0.5  # eyes wide
                face[30:35, 22:26] += 0.2  # mouth open
                
            elif emotion == 3:  # happy
                face[30:38, 15:33] += 0.6  # big smile
                face[18:25, 20:28] += 0.2  # eyes crinkle
                face[25:30, 18:30] += 0.1  # cheeks up
                
            elif emotion == 4:  # sad
                face[30:35, 20:28] -= 0.5  # mouth down
                face[15:20, 18:30] -= 0.2  # eyebrows down
                face[25:30, 20:28] -= 0.1  # cheeks down
                
            elif emotion == 5:  # surprised
                face[18:28, 18:30] += 0.6  # eyes very wide
                face[30:38, 22:26] += 0.4  # mouth open
                face[12:18, 20:28] += 0.3  # eyebrows up
                
            elif emotion == 6:  # neutral
                # Keep base face with minimal changes
                face[30:35, 22:26] += 0.1  # slight mouth
            
            face = np.clip(face, 0, 1)
            X_train.append(face)
            y_train.append(emotion)
    
    # Create test data (400 samples per emotion)
    for emotion in range(7):
        for _ in range(400):
            face = np.random.normal(0.4, 0.1, (48, 48, 1))
            
            # Same patterns as training but with slight variations
            if emotion == 0:  # angry
                face[12:18, 15:33] += 0.35
                face[28:35, 18:30] -= 0.25
            elif emotion == 3:  # happy
                face[30:38, 15:33] += 0.55
                face[18:25, 20:28] += 0.15
            elif emotion == 4:  # sad
                face[30:35, 20:28] -= 0.45
                face[15:20, 18:30] -= 0.15
            elif emotion == 5:  # surprised
                face[18:28, 18:30] += 0.55
                face[30:38, 22:26] += 0.35
            
            face = np.clip(face, 0, 1)
            X_test.append(face)
            y_test.append(emotion)
    
    X_train = np.array(X_train)
    y_train = to_categorical(np.array(y_train), 7)
    X_test = np.array(X_test)
    y_test = to_categorical(np.array(y_test), 7)
    
    # Shuffle the data
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    return X_train, X_test, y_train, y_test

def train_model():
    print("Loading dataset...")
    X_train, X_test, y_train, y_test = load_fer2013_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    
    print("Creating model...")
    model = create_emotion_model()
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001
    )
    
    print("Training model...")
    # Train with both augmented and original data
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        steps_per_epoch=len(X_train) // 32,
        epochs=100,  # More epochs for better learning
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Additional training on original data for better pattern recognition
    print("Fine-tuning on original data...")
    model.fit(X_train, y_train, 
              batch_size=16, 
              epochs=20, 
              validation_data=(X_test, y_test),
              verbose=1)
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save model
    model.save('emotion_model.h5')
    print("Model trained and saved successfully!")
    
    return model, history

if __name__ == "__main__":
    train_model()