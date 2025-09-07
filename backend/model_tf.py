# backend/model_tf.py
import io
import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf
from typing import Dict

# load your frozen model (update path)
MODEL_PATH = r"E:\Capstone Project\CryFusion Project\backend\models\cnn_lstm_frozen.keras"
MODEL = tf.keras.models.load_model(MODEL_PATH)

# update to your labels order exactly as training
LABELS = ["belly_pain", "burping", "discomfort", "hungry", "tired"]

# model input assumptions
N_MFCC = 40
MODEL_FRAMES = 300  # time dimension (frames)
CHANNELS_LAST = True  # model expects (batch, 40, 300, 1)

def wav_bytes_to_np(wav_bytes: bytes, target_sr=22050):
    # read via soundfile (works for wav). If bytes are not wav, ffmpeg already produced wav before calling this.
    bio = io.BytesIO(wav_bytes)
    data, sr = sf.read(bio, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return data, sr

def extract_mfcc_from_wave_bytes(wav_bytes: bytes, n_mfcc=N_MFCC, model_frames=MODEL_FRAMES):
    y, sr = wav_bytes_to_np(wav_bytes, target_sr=22050)
    # trim leading/trailing silence
    y_trim, _ = librosa.effects.trim(y)
    # compute mfcc: shape (n_mfcc, frames)
    mfcc = librosa.feature.mfcc(y=y_trim, sr=sr, n_mfcc=n_mfcc)
    # pad/truncate along frames dimension to model_frames
    frames = mfcc.shape[1]
    if frames < model_frames:
        pad_cols = model_frames - frames
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_cols)), mode="constant")
    else:
        mfcc = mfcc[:, :model_frames]
    return mfcc  # shape (n_mfcc, model_frames)

def predict_from_audio_bytes(wav_bytes: bytes) -> Dict:
    """
    Input: wav audio bytes (mono or stereo)
    Output: {"label": str, "confidence": float, "probs": {label:prob,...}}
    """
    mfcc = extract_mfcc_from_wave_bytes(wav_bytes)  # (n_mfcc, frames)
    # reshape according to your model: (1, 40, 300, 1)
    inp = np.expand_dims(mfcc, axis=0)  # (1, 40, 300)
    if len(MODEL.input_shape) == 4 and MODEL.input_shape[-1] == 1:
        inp = np.expand_dims(inp, -1)  # (1, 40, 300, 1)
    # optionally normalize / standardize as per training (mean/std)
    probs = MODEL.predict(inp)[0]  # 1D array
    idx = int(np.argmax(probs))
    label = LABELS[idx]
    confidence = float(probs[idx])
    probs_by_label = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    return {"label": label, "confidence": confidence, "probs": probs_by_label}
