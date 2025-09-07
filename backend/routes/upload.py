"""from fastapi import APIRouter, UploadFile, File
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
import io

router = APIRouter()

# Load model (reuse the same as realtime)
MODEL_PATH = r"E:\Capstone Project\CryFusion Project\backend\models\cnn_lstm_frozen.keras"
model = tf.keras.models.load_model(MODEL_PATH)
LABELS = ["belly_pain", "burping", "discomfort", "hungry", "tired"]
MFCC_MAX_LEN = 300

@router.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    try:
        # Read uploaded file
        contents = await file.read()
        audio, sr = sf.read(io.BytesIO(contents))

        # Resample to 16kHz
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Extract MFCC and pad/truncate
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        if mfcc.shape[1] < MFCC_MAX_LEN:
            pad_width = MFCC_MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :MFCC_MAX_LEN]

        mfcc = np.expand_dims(mfcc, axis=0)
        mfcc = np.expand_dims(mfcc, axis=-1)

        # Predict
        prediction = model.predict(mfcc, verbose=0)
        predicted_label = LABELS[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return {"prediction": predicted_label, "confidence": confidence}

    except Exception as e:
        return {"error": str(e)}
"""

from fastapi import APIRouter, UploadFile, File

router = APIRouter()

@router.post("/audio")
async def upload_audio(file: UploadFile = File(...)):
    # Dummy response for testing
    return {
        "filename": file.filename,
        "status": "uploaded",
        "prediction": "hungry (dummy prediction)"
    }
