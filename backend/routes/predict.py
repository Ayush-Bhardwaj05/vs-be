from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import librosa
import tensorflow as tf
import os
import traceback
import uuid
import asyncio
from pathlib import Path

router = APIRouter(prefix="")

# Configuration
SAMPLE_RATE = 16000  # Must match frontend conversion
DURATION = 3         # Seconds of audio needed
OFFSET = 0.5         # Initial offset
MFCC_FEATURES = 40   # Number of MFCC coefficients

# Create necessary directories
UPLOAD_DIR = Path("recordings")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# Load model once at startup
try:
    model = tf.keras.models.load_model("models/speech_emotion_recognition_model.h5")
    print("✅ Model loaded successfully!")
    print(f"Model input shape: {model.input_shape}")
except Exception as e:
    print(f"❗ Failed to load model: {str(e)}")
    raise RuntimeError("Model loading failed")

# Emotion labels (ensure order matches training)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']

def validate_audio(y: np.ndarray, sr: int):
    """Validate audio meets processing requirements"""
    if len(y) == 0:
        raise ValueError("Audio is empty")
    if sr != SAMPLE_RATE:
        raise ValueError(f"Invalid sample rate {sr}. Expected {SAMPLE_RATE}")
    if y.dtype != np.float32:
        raise ValueError(f"Invalid audio dtype {y.dtype}. Expected float32")

def extract_mfcc(file_path: Path):
    """Extract MFCC features with proper error handling"""
    try:
        # Load with strict parameters matching frontend conversion
        y, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            mono=True,
            duration=DURATION,
            offset=OFFSET,
            dtype=np.float32
        )
        
        validate_audio(y, sr)
        
        # Calculate MFCC features
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=MFCC_FEATURES,
            n_fft=512,
            hop_length=256,
            center=True
        )
        
        # Process features
        mfcc = np.mean(mfcc.T, axis=0)
        
        if mfcc.shape != (MFCC_FEATURES,):
            raise ValueError(f"Invalid MFCC shape {mfcc.shape}. Expected ({MFCC_FEATURES},)")
            
        return mfcc
        
    except Exception as e:
        print(f"❗ Feature extraction failed: {str(e)}")
        raise

async def predict_emotion(file_path: Path):
    """Async wrapper for prediction with proper typing"""
    try:
        # Extract features
        mfcc_features = extract_mfcc(file_path)
        
        # Reshape for model input
        input_data = mfcc_features[np.newaxis, ..., np.newaxis]
        
        # Validate input shape
        if input_data.shape[1:] != model.input_shape[1:]:
            raise ValueError(
                f"Input shape mismatch. Model expects {model.input_shape[1:]}, "
                f"got {input_data.shape[1:]}"
            )
        
        # Run prediction
        prediction = await asyncio.to_thread(model.predict, input_data, verbose=0)
        predicted_index = np.argmax(prediction, axis=1)[0]
        
        if predicted_index >= len(emotion_labels):
            raise ValueError(f"Predicted index {predicted_index} out of range")
            
        return emotion_labels[predicted_index]
        
    except Exception as e:
        print(f"❗ Prediction failed: {str(e)}")
        raise

@router.post("/predict-emotion/", response_model=dict)
async def predict_emotion_endpoint(file: UploadFile = File(...)):
    """Endpoint for emotion prediction with proper error handling"""
    temp_file = None
    
    try:
        # Validate input
        if not file.filename.lower().endswith(".wav"):
            raise HTTPException(400, "Only WAV files are accepted")
            
        # Create temp file
        temp_file = TEMP_DIR / f"{uuid.uuid4()}.wav"
        
        # Read content asynchronously
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(400, "Empty file received")
        
        # Write content synchronously
        with temp_file.open("wb") as buffer:
            buffer.write(content)
            
        # Process file
        emotion = await predict_emotion(temp_file)
        
        return {"emotion": emotion}
        
    except HTTPException as he:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Processing error: {str(e)}")
    finally:
        # Cleanup temp file
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
                print(f"🗑️ Cleaned up temp file: {temp_file}")
            except Exception as cleanup_error:
                print(f"❗ Temp file cleanup failed: {cleanup_error}")
