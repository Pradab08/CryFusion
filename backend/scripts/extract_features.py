"""
Accuracy: 60.5% model


import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Paths
DATASET_PATH = r"E:\Capstone Project\CryFusion Project\backend\data\Baby Cry Sence Dataset"   # root folder containing class subfolders
OUTPUT_FEATURES = "features.npy"
OUTPUT_LABELS = "labels.npy"
OUTPUT_CLASSES = "class_names.npy"

# Parameters
n_mfcc = 40
sample_rate = 22050

def extract_features(dataset_path):
    features = []
    labels = []
    class_names = sorted(os.listdir(dataset_path))  # 5 classes

    max_len = 0
    temp_features = []

    # Pass 1: Extract MFCCs and track longest sequence
    print("Extracting MFCCs...")
    for label in class_names:
        folder = os.path.join(dataset_path, label)
        if not os.path.isdir(folder):
            continue

        for file in os.listdir(folder):
            if not file.endswith(".wav"):
                continue

            file_path = os.path.join(folder, file)
            signal, sr = librosa.load(file_path, sr=sample_rate)

            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
            mfcc = mfcc.T  # shape: (time, n_mfcc)

            temp_features.append((mfcc, label))
            max_len = max(max_len, mfcc.shape[0])

    print(f"Max sequence length found: {max_len} frames")

    # Pass 2: Pad sequences
    for mfcc, label in temp_features:
        if mfcc.shape[0] < max_len:
            pad_width = max_len - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode="constant")

        features.append(mfcc)
        labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    # Encode labels
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)

    print("Feature shape:", features.shape)  # (samples, max_len, n_mfcc)
    print("Labels shape:", labels_categorical.shape)
    print("Classes:", encoder.classes_)

    # Save everything
    np.save(OUTPUT_FEATURES, features)
    np.save(OUTPUT_LABELS, labels_categorical)
    np.save(OUTPUT_CLASSES, encoder.classes_)

    print(f"Saved: {OUTPUT_FEATURES}, {OUTPUT_LABELS}, {OUTPUT_CLASSES}")

if __name__ == "__main__":
    extract_features(DATASET_PATH)"""

import os
import numpy as np
import librosa
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import joblib

DATASET_PATH = "backend/data/Baby Cry Sence Dataset"
OUTPUT_PATH = "backend/data/Processed Baby Cry Sence Dataset"

# Classes
CLASSES = ["belly_pain", "burping", "discomfort", "hungry", "tired"]

# Augmentation functions
def pitch_shift(y, sr):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=random.choice([-2, -1, 1, 2]))

def time_stretch(y):
    rate = random.choice([0.9, 1.1])
    return librosa.effects.time_stretch(y=y, rate=rate)

def add_noise(y):
    noise = np.random.randn(len(y))
    return y + 0.005 * noise

def extract_features(file_path, sr=22050, n_mfcc=40, max_len=300):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, :max_len]
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Main feature extraction with augmentation
def main():
    X, y = [], []

    for label_idx, label in enumerate(CLASSES):
        folder = os.path.join(DATASET_PATH, label)
        files = os.listdir(folder)

        for file in tqdm(files, desc=f"Processing {label}"):
            file_path = os.path.join(folder, file)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(label_idx)

                # Augment minority classes
                if label != "hungry":  # Augment all except majority
                    y_raw, sr = librosa.load(file_path, sr=22050)

                    # Pitch shift
                    aug1 = pitch_shift(y_raw, sr)
                    feat1 = extract_features(file_path)
                    if feat1 is not None:
                        X.append(feat1)
                        y.append(label_idx)

                    # Time stretch
                    aug2 = time_stretch(y_raw)
                    mfcc2 = librosa.feature.mfcc(y=aug2, sr=sr, n_mfcc=40)
                    if mfcc2.shape[1] < 300:
                        pad = 300 - mfcc2.shape[1]
                        mfcc2 = np.pad(mfcc2, pad_width=((0,0),(0,pad)), mode="constant")
                    else:
                        mfcc2 = mfcc2[:, :300]
                    X.append(mfcc2)
                    y.append(label_idx)

                    # Add noise
                    aug3 = add_noise(y_raw)
                    mfcc3 = librosa.feature.mfcc(y=aug3, sr=sr, n_mfcc=40)
                    if mfcc3.shape[1] < 300:
                        pad = 300 - mfcc3.shape[1]
                        mfcc3 = np.pad(mfcc3, pad_width=((0,0),(0,pad)), mode="constant")
                    else:
                        mfcc3 = mfcc3[:, :300]
                    X.append(mfcc3)
                    y.append(label_idx)

    X = np.array(X)
    y = np.array(y)

    print("Final dataset shape:", X.shape, y.shape)

    # Save processed data
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    joblib.dump((X, y), os.path.join(OUTPUT_PATH, "features.pkl"))
    print("✅ Features saved at:", os.path.join(OUTPUT_PATH, "features.pkl"))

if __name__ == "__main__":
    main()
