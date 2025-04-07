from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import librosa
import os
import pickle
from tensorflow.keras.models import load_model
import tempfile
import traceback
from pydub import AudioSegment
import uuid
import sklearn

# Initialize router
router = APIRouter()

UPLOAD_DIR = "recordings"
os.makedirs(UPLOAD_DIR, exist_ok=True)

print(f"Scikit-learn version: {sklearn.__version__}")  # Add this before loading


# Load model and label encoder
model_path = os.path.join("models", "indian_accent_emotion_model_updated.h5")
label_encoder_path = os.path.join("models", "indian_accent_label_encoder_updated.pkl")

try:
    model = load_model(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❗ Error loading model: {e}")
    raise HTTPException(status_code=500, detail="Failed to load the model")

try:
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    print("✅ Label encoder loaded successfully!")
except Exception as e:
    print(f"❗ Error loading label encoder: {e}")
    raise HTTPException(status_code=500, detail="Failed to load the label encoder")

# Feature extraction function
def extract_features(file_path, sr=16000, n_mfcc=40, max_pad_len=174):
    try:
        y, sr = librosa.load(file_path, sr=sr, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-10)
        return mfcc
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Emotion prediction function
def predict_emotion(file_path):
    mfcc = extract_features(file_path)
    if mfcc is None:
        return None
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.transpose(mfcc, (0, 2, 1))
    prediction = model.predict(mfcc)
    predicted_index = np.argmax(prediction, axis=1)[0]
    emotion = label_encoder.inverse_transform([predicted_index])[0]
    return emotion

# API endpoint for prediction
@router.post("/predict-indian")
async def predict_indian(file: UploadFile = File(...)):
    # Save the incoming file (regardless of extension)
    file_extension = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    print(f"File saved at: {file_path}")
    
    # Convert to WAV if necessary using pydub
    if file_extension.lower() != "wav":
        try:
            audio = AudioSegment.from_file(file_path)
            wav_file_path = file_path.rsplit(".", 1)[0] + ".wav"
            audio.export(wav_file_path, format="wav")
            file_path = wav_file_path
            print(f"Converted file to WAV at: {file_path}")
        except Exception as e:
            print(f"Error converting file to WAV: {e}")
            raise HTTPException(status_code=400, detail="Failed to convert file to WAV")
    
    try:
        emotion = predict_emotion(file_path)
        if emotion is None:
            raise Exception("Failed to extract features or predict emotion")
        print(f"Emotion Prediction: {emotion}", flush=True)
        # Optionally remove the file after processing:
        # os.remove(file_path)
        return {"emotion": emotion}
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❗ Error in predict-indian: {error_details}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))
