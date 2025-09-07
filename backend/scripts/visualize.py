import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model

# -----------------------------
# Paths & Config
# -----------------------------
OUTPUT_PATH = r"E:\Capstone Project\CryFusion Project\backend\data\Processed Baby Cry Sence Dataset"
MODEL_PATH = r"E:\Capstone Project\CryFusion Project\backend\models\cnn_lstm.keras"
CLASSES = ["belly_pain", "burping", "discomfort", "hungry", "tired"]

# -----------------------------
# Model Definition (same as training script)
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

    return model

# -----------------------------
# Main
# -----------------------------
def main():
    # Load dataset to get input shape
    X, y = joblib.load(os.path.join(OUTPUT_PATH, "features.pkl"))
    X = np.expand_dims(X, -1)  # (samples, n_mfcc, time, 1)

    input_shape = (X.shape[1], X.shape[2], 1)

    # Build model
    model = build_model(input_shape, num_classes=len(CLASSES))

    # Save architecture diagram
    os.makedirs("models", exist_ok=True)
    plot_model(
        model,
        to_file="models/model_architecture.png",
        show_shapes=True,
        show_layer_names=True,
        expand_nested=True,
        dpi=120
    )
    print("✅ Model architecture diagram saved at models/model_architecture.png")

if __name__ == "__main__":
    main()
