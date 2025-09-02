# Bu sınıf microfon kullanımı için
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import openai
import os

# =========================================================

openai.api_key = os.getenv("OPENAI_API_KEY")

# =========================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    with open("temp_audio.wav", "wb") as f:
        f.write(await file.read())

    with open("temp_audio.wav", "rb") as audio_file:
        transcript = openai.Audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return {"text": transcript.text}

# =========================================================

@app.post("/api/tts")
async def text_to_speech(text: str = Form(...)):
    response = openai.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text
    )
    return StreamingResponse(response, media_type="audio/mpeg")

# =========================================================
