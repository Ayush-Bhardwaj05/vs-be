from fastapi import APIRouter, File, UploadFile, HTTPException
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import io
from PIL import Image

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('models/model.h5')
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
ANGRY_THRESHOLD = 0.3

router = APIRouter(prefix="")

@router.post("/video/predict")
async def predict_emotion(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image_stream = io.BytesIO(contents)
        img = Image.open(image_stream).convert('RGB')
        img = np.array(img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                angry_prob = prediction[0]
                if angry_prob > ANGRY_THRESHOLD:
                    return {"emotion": "Angry"}
                return {"emotion": emotion_labels[prediction.argmax()]}

        return {"emotion": "No Face"}

    except Exception as e:
        return {"error": str(e)}