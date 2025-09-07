import numpy as np
import time
import tensorflow as tf

LABELS = ["hungry","tired","discomfort","belly_pain","burping","lonely","scared"]

def load_model_stub():
    # placeholder that imitates loading
    return {"type":"stub", "loaded_at": time.time()}

MODEL = MODEL = tf.keras.models.load_model(r"E:\Capstone Project\CryFusion Project\backend\models\cnn_lstm_frozen.keras")

def predict_from_mfcc(mfcc):
    # deterministic-ish stub based on energy -> pick label
    energy = np.mean(np.abs(mfcc))
    # map energy to label index
    idx = int((energy * 1000) % len(LABELS))
    confidence = float(min(0.98, 0.5 + (energy % 1)))
    return {"label": LABELS[idx], "confidence": confidence}
