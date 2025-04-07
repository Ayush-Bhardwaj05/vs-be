from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
import google.generativeai as genai
import os
import traceback
from pydub import AudioSegment
import uuid

# Create a FastAPI router
convo_router = APIRouter(prefix="/convo", tags=["Conversation"])

# Configure the API key
genai.configure(api_key="AIzaSyCyVeQ2RU1jmNjbLmgTSuivNRFEz-aSjio")

UPLOAD_DIR = "recordings"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class TextRequest(BaseModel):
    text: str

@convo_router.post("/process-text")
async def process_text(data: TextRequest):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        content_response = model.generate_content(data.text)
        return {"response": content_response.text}
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❗ Error in process-text: {error_details}")
        raise HTTPException(status_code=500, detail=str(e))

def convert_webm_to_wav(input_path, output_path):
    try:
        audio = AudioSegment.from_file(input_path, format="webm")
        audio.export(output_path, format="wav")
        print(f"✅ Converted to WAV: {output_path}")
        return output_path
    except Exception as e:
        print(f"❗ Error during conversion: {e}")
        raise HTTPException(status_code=500, detail="Error converting file to WAV.")


@convo_router.post("/process-audio")
async def upload_audio(file: UploadFile = File(...), responses: str = Form(...)):
    print(f"Received responses: {responses}")

    file_extension = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    print(f"File saved at: {file_path}")

    try:
        myfile = genai.upload_file(file_path)
        model = genai.GenerativeModel("gemini-2.0-flash")

        # Format the responses for conversation context
        response_list = responses.split('|') if responses else []
        conversation_prompt = "\n".join(response_list)

        # Build the final prompt
        prompt = (
            f"This is the conversation so far: {conversation_prompt}\n"
            "analyze the provided audio file."
            "If you dont see any conversation above, respond strictly with whether the user seems happy, sad,angry or neutral"
            "If this is a follow-up interaction, respond with a follow-up question or statement based on the previous response"
            "i need you to act and respond  like a human after the first response"
            
        )

        response = model.generate_content(contents=[prompt, myfile])
        print(f"Gemini response: {response.text}", flush=True)

        # Delete the recording after processing
        os.remove(file_path)

        # Return the response to the client
        return {
            "message": "File uploaded and processed",
            "file_path": filename,
            "response": response.text
        }
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❗ Error in process-audio: {error_details}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@convo_router.get("/get-audio/{filename}")
async def get_audio(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    return {"error": "File not found"}
