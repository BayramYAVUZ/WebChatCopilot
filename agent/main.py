import uvicorn
import os
import openai
from openai import APIStatusError, APIConnectionError
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# ==========================================================

openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==========================================================

app = FastAPI(title="Sample Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        with open("temp_audio.wav", "wb") as f:
            f.write(await file.read())
        
        with open("temp_audio.wav", "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return {"text": transcript.text}
    except Exception as e:
        print(f"Transcription error: {e}")
        return {"text": ""}
    finally:
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")

# ==========================================================

@app.post("/api/tts")
async def text_to_speech(text: str = Form(...)):
    try:
        response = client.audio.speech.create(
            model="tts-1",  # veya "tts-1-hd"
            voice="alloy",
            input=text
        )
        return StreamingResponse(response.iter_bytes(), media_type="audio/mpeg")
    except APIStatusError as e:
        raise HTTPException(status_code=e.status_code, detail=f"API Error: {e.message}")
    except APIConnectionError as e:
        raise HTTPException(status_code=503, detail="Connection Error: Unable to connect to OpenAI.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# ==========================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8123, reload=True)

# ==========================================================    
