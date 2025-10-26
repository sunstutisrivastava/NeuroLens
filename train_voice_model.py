import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 📁 Path to your dataset
data_dir = "models/voice"
emotions = ["euphoric", "sad", "joyfully", "surprised"]

# 🎧 Extract MFCC features from each audio file
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"⚠️ Could not process {file_path}: {e}")
        return None

# 🧾 Prepare dataset
features, labels = [], []

for idx, emotion in enumerate(emotions):
    emotion_dir = os.path.join(data_dir, emotion)
    if not os.path.exists(emotion_dir):
        print(f"⚠️ Folder not found: {emotion_dir}")
        continue

    for file_name in os.listdir(emotion_dir):
        file_path = os.path.join(emotion_dir, file_name)
        if file_path.endswith(".wav") or file_path.endswith(".mp3"):
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(idx)

# ✅ Convert to numpy arrays
features = np.array(features)
labels = np.array(labels)

if len(features) == 0:
    raise ValueError("❌ No audio data found! Make sure .wav or .mp3 files exist in your folders.")

# 🧠 One-hot encode labels
labels = to_categorical(labels, num_classes=len(emotions))

# 📊 Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# 🧱 Model architecture
model = Sequential([
    Dense(256, activation='relu', input_shape=(40,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(emotions), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🚀 Train the model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=8,
    validation_data=(X_test, y_test)
)

# 💾 Save trained model
os.makedirs("checkpoints", exist_ok=True)
model.save("checkpoints/best_voice_model.h5")

print("✅ Voice emotion model saved successfully at checkpoints/best_voice_model.h5")
