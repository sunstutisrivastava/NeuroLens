import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import os
import numpy as np

# Paths
train_dir = "models/images"
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

# Check if dataset exists
if not os.path.exists(train_dir):
    print(f"Dataset not found at {train_dir}")
    exit()

# Count classes
emotion_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
num_classes = len(emotion_classes)
print(f"Found {num_classes} emotion classes: {emotion_classes}")

# Image parameters
img_size = (48, 48)  # Standard for emotion detection
batch_size = 32

# Enhanced data preprocessing with augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

# Data generators
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    color_mode='grayscale'  # Emotion detection typically uses grayscale
)

val_gen = val_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    color_mode='grayscale'
)

print(f"Training samples: {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")
print(f"Class indices: {train_gen.class_indices}")

# Build improved CNN model
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
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=8,
        min_lr=0.00001,
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(save_dir, 'best_emotion_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train model
print("Starting training...")
history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // batch_size,
    validation_data=val_gen,
    validation_steps=val_gen.samples // batch_size,
    epochs=100,
    callbacks=callbacks,
    verbose=1
)

# Save final model
final_model_path = os.path.join(save_dir, 'emotion_model.h5')
model.save(final_model_path)

# Save class indices for later use
import json
class_indices_path = os.path.join(save_dir, 'class_indices.json')
with open(class_indices_path, 'w') as f:
    json.dump(train_gen.class_indices, f)

print(f"✅ Training completed!")
print(f"✅ Best model saved to: {os.path.join(save_dir, 'best_emotion_model.h5')}")
print(f"✅ Final model saved to: {final_model_path}")
print(f"✅ Class indices saved to: {class_indices_path}")

# Evaluate model
print("\nEvaluating model...")
val_loss, val_accuracy = model.evaluate(val_gen, verbose=0)
print(f"Final validation accuracy: {val_accuracy:.4f}")
print(f"Final validation loss: {val_loss:.4f}")