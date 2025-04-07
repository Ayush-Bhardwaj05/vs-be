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
from pydantic import BaseModel
from emotion_st import save_emotion, get_emotion_heatmap

router = APIRouter(prefix="")


class EmotionRequest(BaseModel):
    user_id: str
    emotion: str
    confidence: float

@router.post("/api/save_emotion")
async def record_emotion(data: EmotionRequest):
    if not all([data.user_id, data.emotion, data.confidence]):
        raise HTTPException(status_code=400, detail="Missing fields")
    result = save_emotion(data.user_id, data.emotion, data.confidence)
    return result

@router.get("/api/heatmap/{user_id}")
async def get_heatmap(user_id: str):
    result = get_emotion_heatmap(user_id)
    return result