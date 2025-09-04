import uvicorn
import os
import openai
from openai import APIStatusError, APIConnectionError, InternalServerError
from fastapi import FastAPI, UploadFile, File, Form, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage # type: ignore
from agent import graph

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

@app.post("/agent_chat")
async def agent_chat(messages: list[str] = Body(...)):
    """
    Endpoint to interact with the agent.
    """
    try:
        langgraph_messages = [HumanMessage(content=m) for m in messages]
        result = await graph.ainvoke({"messages": langgraph_messages})
        return {"response": [msg.content for msg in result.get("messages", [])]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred in the agent chat: {str(e)}")

# ==========================================================

@app.post("/transcribeAudioUrl")
async def transcribe_audio_url(audio_url: str = Body(..., embed=True)):
    # Bu endpoint sadece dummy bir yanıt döndürüyor,
    # gerçek bir API çağrısı olmadığı için try-except eklemeye gerek yok.
    return {"transcription": f"Transcription for: {audio_url}"}

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
    except APIStatusError as e:
        raise HTTPException(status_code=e.status_code, detail=f"API Error: {e.message}")
    except APIConnectionError as e:
        raise HTTPException(status_code=503, detail="Connection Error: Unable to connect to OpenAI.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# ==========================================================

@app.post("/textToSpeechUrl")
async def text_to_speech_url(text: str = Body(..., embed=True)):
    # Bu endpoint de dummy bir yanıt döndürüyor.
    return {"audio_url": f"https://dummy-audio.com/{text}.mp3"}

@app.post("/api/tts")
async def text_to_speech(text: str = Form(...)):
    try:
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text
        )
        # response'u StreamingResponse'a dönüştürme mantığı burada kalmalı
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