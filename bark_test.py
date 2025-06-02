from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from bark import generate_audio, preload_models
from scipy.io.wavfile import write
import uuid
import os

app = FastAPI()
preload_models()

@app.post("/tts")
async def tts(request: Request):
    data=await request.json()
    text=data["text"]
    audio_array=generate_audio(text)

    filename=f"/tmp/{uuid.uuid4()}.wav"
    write(filename, 24000, audio_array)

    return FileResponse(filename, media_type="audio/wav")
