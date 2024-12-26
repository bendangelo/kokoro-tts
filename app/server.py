import time
from io import BytesIO
import base64
from fastapi.security.api_key import APIKeyHeader
import os
from fastapi import FastAPI, Security, HTTPException, Depends
from pydantic import BaseModel, Field
from starlette.status import HTTP_403_FORBIDDEN
from typing import Optional
import uvicorn
from __version__ import __version__
import logging
import magic
from pydub import AudioSegment
from fastapi.responses import StreamingResponse
import torch
from models import build_model
import numpy as np


# Setup logging
log_level = os.getenv("LOG_LEVEL", "INFO")
log_level = log_level.upper()
if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
    log_level = "INFO"
logging.basicConfig(level=getattr(logging, log_level))

# Server configuration
host = os.getenv("HOST", "*")
port = os.getenv("PORT", "4321")
port = int(port)

# Model initialization
device = 'cuda' if torch.cuda.is_available() else 'cpu'
kokoro_model = build_model('kokoro-v0_19.pth', device)
voicepack = torch.load('voices/af.pt', weights_only=True).to(device)
af_bella = torch.load('voices/af_bella.pt', weights_only=True).to(device)
af_sarah = torch.load('voices/af_sarah.pt', weights_only=True).to(device)

from kokoro import generate, tokenize, VOCAB

KOKORO_API_KEY = os.getenv("KOKORO_API_KEY")
if not KOKORO_API_KEY:
    logging.info("Environment variable KOKORO_API_KEY is not set")
else:
    logging.info(f"Environment variable KOKORO_API_KEY is {KOKORO_API_KEY}")

warmup_text = "This is an inference API for TTS. It is now warming up..."

load_start = time.perf_counter()
logging.info(f"Models loaded in {time.perf_counter() - load_start} seconds.")

app = FastAPI()

API_KEY_HEADER = APIKeyHeader(name="API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if KOKORO_API_KEY == None:
        return KOKORO_API_KEY

    if not api_key:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="No API key provided"
        )
    
    if api_key != KOKORO_API_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Invalid API key"
        )
    
    return api_key

def get_wav_length_from_bytesio(bytes_io):
    bytes_io.seek(0)
    audio = AudioSegment.from_file(bytes_io, format="wav")
    duration_seconds = len(audio) / 1000.0
    return audio, duration_seconds

@app.get("/hc")
def health_check():
    return {"status": "ok", "version": __version__}

def get_voice_model(voice_name: str):
    """Helper function to get the appropriate voice model"""
    voice_mapping = {
        "bella": af_bella,
        "sarah": af_sarah,
        "default": voicepack
    }
    return voice_mapping.get(voice_name, voicepack)

class TTSRequest(BaseModel):
    text: str = Field(..., title="Text to convert to speech")
    phonetics: Optional[str] = Field(
        None,
        title="Custom phonetics",
        description="Custom phonetics string for pronunciation. If not provided, will be generated automatically."
    )
    voice: Optional[str] = Field(
        "default",
        title="Voice selection",
        description="Select voice to use: 'bella', 'sarah', or 'default'"
    )
    output_sample_rate: Optional[int] = 24000
    speed: Optional[float] = Field(
        1.0,
        title="Speech speed",
        description="Speech speed factor. 1.0 is normal speed."
    )
    output_format: Optional[str] = "mp3"

@app.post("/generate")
def generate_speech(request: TTSRequest, api_key: str = Depends(verify_api_key)):
    start = time.perf_counter()
    wav_bytes = BytesIO()
    
    # Get the appropriate voice model
    voice_model = get_voice_model(request.voice)
    
    # Generate phonetics from text
    audio, _ = generate(kokoro_model, request.text, voice_model, speed=request.speed, ps=request.phonetics)

    if audio is None:
        raise HTTPException(status_code=500, detail="Failed to generate audio")
        
    # Convert numpy array to wav
    import scipy.io.wavfile as wav
    wav.write(wav_bytes, request.output_sample_rate, (audio * 32767).astype(np.int16))
        
    inference_time = time.perf_counter() - start
    logging.info(f"Generated audio in {inference_time} seconds.")
    
    audio, duration_seconds = get_wav_length_from_bytesio(wav_bytes)
    
    headers = {
        "x-inference-time": str(inference_time),
        "x-audio-length": str(duration_seconds),
        "x-realtime-factor": str(duration_seconds / inference_time),
    }
    
    return_bytes = BytesIO()
    audio.export(return_bytes, format=request.output_format)
    return_bytes.seek(0)

    return StreamingResponse(
        return_bytes,
        media_type=f"audio/{request.output_format}",
        headers=headers,
    )

if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port)
