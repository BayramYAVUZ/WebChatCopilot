import uvicorn
import os
import openai
from openai import APIStatusError, APIConnectionError
from fastapi import FastAPI, UploadFile, File, Form, Body, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage # type: ignore
from agent import graph
from copilot_kit import CopilotKit, LangchainAdapter # type: ignore

# ==========================================================

openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Sample Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# CopilotKit Backend Entegrasyonu
# Mevcut langgraph agent'ınızı CopilotKit'e bağlıyoruz.
# ==========================================================

copilot = CopilotKit(
    langchain_adapter=LangchainAdapter(graph)
)

@app.post("/copilotkit")
async def handle_copilot_chat(request: Request):
    """
    Bu endpoint, frontend'den gelen tüm CopilotKit isteklerini (chat, actions vb.)
    işler ve agent'a yönlendirir.
    """
    return await copilot.handle_request(request)

# ==========================================================
# Mikrofon için Gerekli Endpoint'ler (Değiştirilmedi)
# ==========================================================

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Frontend'den gönderilen sesi metne çevirir.
    """
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

@app.post("/api/tts")
async def text_to_speech(text: str = Form(...)):
    """
    Frontend'den gönderilen metni sese çevirir.
    """
    try:
        response = client.audio.speech.create(
            model="tts-1",  # veya "tts-1-hd" gibi başka bir model
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