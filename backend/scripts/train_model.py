"""
Accuracy: 60.5%


import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LSTM, Reshape, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
import random

# -----------------------------
# Audio Augmentation Functions
# -----------------------------
def add_noise(data, noise_factor=0.005):
    return data + noise_factor * np.random.randn(len(data))

def time_stretch(data, rate=None):
    if rate is None:
        rate = np.random.uniform(0.8, 1.2)
    return librosa.effects.time_stretch(y=data, rate=rate)

def pitch_shift(data, sr, n_steps=None):
    if n_steps is None:
        n_steps = np.random.randint(-2, 3)
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)

def augment_audio(file_path, output_dir, num_aug=5):
    y, sr = librosa.load(file_path, sr=None)

    augmentations = [add_noise, time_stretch, lambda d: pitch_shift(d, sr)]
    
    for i in range(num_aug):
        aug_func = random.choice(augmentations)
        try:
            y_aug = aug_func(y)
            out_file = os.path.join(output_dir, f"aug_{i}_{os.path.basename(file_path)}")
            sf.write(out_file, y_aug, sr)
        except Exception as e:
            print(f"⚠️ Augmentation failed for {file_path}: {e}")

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(file_path, max_pad_len=174):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc
    except Exception as e:
        print(f"⚠️ Feature extraction failed for {file_path}: {e}")
        return None

# -----------------------------
# Model Definition
# -----------------------------
def create_model(input_shape, num_classes):
    model = Sequential([
        Reshape((*input_shape, 1), input_shape=input_shape),
        
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Reshape((-1, 64)),
        LSTM(64, return_sequences=False),
        Dropout(0.3),

        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -----------------------------
# Evaluation - ALL CLASSES
# -----------------------------
def evaluate_all_classes(model, X_test, y_test, class_names):
    print("\n🔍 COMPREHENSIVE EVALUATION - ALL CLASSES")
    print("="*45)

    pred_probs = model.predict(X_test, verbose=0)
    pred_classes = np.argmax(pred_probs, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    # Overall accuracy
    accuracy = accuracy_score(true_classes, pred_classes)
    print(f"Overall accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

    # Prediction distribution
    unique_preds, pred_counts = np.unique(pred_classes, return_counts=True)
    print(f"\n📊 PREDICTION DISTRIBUTION:")
    for pred_class, count in zip(unique_preds, pred_counts):
        class_name = class_names[pred_class]
        percentage = count / len(pred_classes) * 100
        print(f"  {class_name:12s}: {count:3d}/{len(pred_classes)} ({percentage:5.1f}%)")

    classes_predicted = len(unique_preds)
    print(f"\n🎯 CLASSES ACTUALLY PREDICTED: {classes_predicted}/{len(class_names)}")

    # Per-class analysis
    print(f"\n📋 PER-CLASS PERFORMANCE:")
    for i, class_name in enumerate(class_names):
        mask = (true_classes == i)
        if np.sum(mask) > 0:
            correct = np.sum((true_classes == i) & (pred_classes == i))
            total = np.sum(mask)
            predicted = np.sum(pred_classes == i)

            precision = correct / predicted if predicted > 0 else 0
            recall = correct / total
            class_acc = recall * 100

            status = "✅" if predicted > 0 else "❌"
            print(f"{status} {class_name:12s}: {correct:2d}/{total:2d} ({class_acc:5.1f}%) | Precision: {precision:.3f}")

# -----------------------------
# Main Training
# -----------------------------
if __name__ == "__main__":
    data_dir = "backend\data\Baby Cry Sence Dataset"  # Update if needed
    class_names = sorted(os.listdir(data_dir))

    X, y = [], []
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(".wav")]

        # Augment minority classes
        if len(files) < 50:
            for f in files:
                augment_audio(f, class_dir, num_aug=5)

        for f in files:
            feat = extract_features(f)
            if feat is not None:
                X.append(feat)
                y.append(idx)

    X = np.array(X)
    y = to_categorical(np.array(y), num_classes=len(class_names))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model
    input_shape = X_train.shape[1:]
    model = create_model(input_shape, len(class_names))

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        ModelCheckpoint("best_model.h5", save_best_only=True)
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    evaluate_all_classes(model, X_test, y_test, class_names)

# Accuracy: 94% model

import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

OUTPUT_PATH = "backend/data/Processed Baby Cry Sence Dataset"
MODEL_PATH = "models/cnn_lstm.h5"
CLASSES = ["belly_pain", "burping", "discomfort", "hungry", "tired"]

def build_model(input_shape, num_classes):
    model = models.Sequential()

    # CNN part
    model.add(layers.Conv2D(32, (3,3), activation="relu", input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (3,3), activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.3))

    # Reshape for LSTM
    model.add(layers.Reshape((-1, 64)))  # Adjust based on feature size

    # LSTM part
    model.add(layers.LSTM(64, return_sequences=False))
    model.add(layers.Dropout(0.3))

    # Dense layers
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    # Load features
    X, y = joblib.load(os.path.join(OUTPUT_PATH, "features.pkl"))
    print("Loaded dataset:", X.shape, y.shape)

    # Expand dims for CNN input
    X = np.expand_dims(X, -1)  # (samples, n_mfcc, time, 1)

    # Log class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\n📊 Class Distribution after augmentation:")
    for cls, cnt in zip(unique, counts):
        print(f"{CLASSES[cls]}: {cnt}")

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y
    )
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}
    print("\n⚖️ Computed Class Weights:", class_weights_dict)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Build model
    input_shape = (X.shape[1], X.shape[2], 1)
    model = build_model(input_shape, num_classes=len(CLASSES))

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weights_dict,
        callbacks=callbacks,
        verbose=1
    )

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\n✅ Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    main()"""

import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import accuracy_score

# -----------------------------
# Paths & Config
# -----------------------------
OUTPUT_PATH = "backend/data/Processed Baby Cry Sence Dataset"
MODEL_PATH = "models/cnn_lstm.keras"   # use .keras format (recommended)
CLASSES = ["belly_pain", "burping", "discomfort", "hungry", "tired"]

# -----------------------------
# Model Definition
# -----------------------------
def build_model(input_shape, num_classes):
    model = models.Sequential()

    # CNN part
    model.add(layers.Conv2D(32, (3,3), activation="relu", input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (3,3), activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.3))

    # Reshape for LSTM
    model.add(layers.Reshape((-1, 64)))

    # LSTM part
    model.add(layers.LSTM(64, return_sequences=False))
    model.add(layers.Dropout(0.3))

    # Dense layers
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_all_classes(model, X_test, y_test, class_names):
    print("\n🔍 COMPREHENSIVE EVALUATION - ALL CLASSES")
    print("="*45)

    pred_probs = model.predict(X_test, verbose=0)
    pred_classes = np.argmax(pred_probs, axis=1)

    # Overall accuracy
    accuracy = accuracy_score(y_test, pred_classes)
    print(f"Overall accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

    # Prediction distribution
    unique_preds, pred_counts = np.unique(pred_classes, return_counts=True)
    print(f"\n📊 PREDICTION DISTRIBUTION:")
    for pred_class, count in zip(unique_preds, pred_counts):
        class_name = class_names[pred_class]
        percentage = count / len(pred_classes) * 100
        print(f"  {class_name:12s}: {count:3d}/{len(pred_classes)} ({percentage:5.1f}%)")

    classes_predicted = len(unique_preds)
    print(f"\n🎯 CLASSES ACTUALLY PREDICTED: {classes_predicted}/{len(class_names)}")

    # Per-class analysis
    print(f"\n📋 PER-CLASS PERFORMANCE:")
    for i, class_name in enumerate(class_names):
        mask = (y_test == i)
        if np.sum(mask) > 0:
            correct = np.sum((y_test == i) & (pred_classes == i))
            total = np.sum(mask)
            predicted = np.sum(pred_classes == i)

            precision = correct / predicted if predicted > 0 else 0
            recall = correct / total
            class_acc = recall * 100

            status = "✅" if predicted > 0 else "❌"
            print(f"{status} {class_name:12s}: {correct:2d}/{total:2d} ({class_acc:5.1f}%) | Precision: {precision:.3f}")

# -----------------------------
# Main Training
# -----------------------------
def main():
    # Load pre-extracted features
    X, y = joblib.load(os.path.join(OUTPUT_PATH, "features.pkl"))
    print("✅ Loaded dataset:", X.shape, y.shape)

    # Expand dims for CNN input
    X = np.expand_dims(X, -1)  # (samples, n_mfcc, time, 1)

    # Log class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\n📊 Class Distribution after augmentation:")
    for cls, cnt in zip(unique, counts):
        print(f"{CLASSES[cls]}: {cnt}")

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y
    )
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}
    print("\n⚖️ Computed Class Weights:", class_weights_dict)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Build model
    input_shape = (X.shape[1], X.shape[2], 1)
    model = build_model(input_shape, num_classes=len(CLASSES))

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1)  # save best model
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weights_dict,
        callbacks=callbacks,
        verbose=1
    )

    # Final save (best already saved by ModelCheckpoint)
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\n✅ Final model saved at {MODEL_PATH}")

    # Evaluate
    evaluate_all_classes(model, X_val, y_val, CLASSES)

if __name__ == "__main__":
    main()
