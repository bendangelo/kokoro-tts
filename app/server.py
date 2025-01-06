import time
from io import BytesIO
import base64
from fastapi.security.api_key import APIKeyHeader
import os
from fastapi import FastAPI, Security, HTTPException, Depends
from pydantic import BaseModel, Field
from starlette.status import HTTP_403_FORBIDDEN
from typing import Optional, Literal
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
host = os.getenv("HOST", "0.0.0.0")
port = os.getenv("PORT", "4321")
port = int(port)

# Available voices configuration
VOICE_NAMES = [
    'af',  # Default voice (50-50 mix of Bella & Sarah)
    'af_bella', 'af_sarah', 'am_adam', 'am_michael',
    'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis',
    'af_nicole', 'af_sky'
]

# Model initialization
device = 'cuda' if torch.cuda.is_available() else 'cpu'
kokoro_model = build_model('kokoro-v0_19.pth', device)

# Load all voice models
voice_models = {}
for voice_name in VOICE_NAMES:
    try:
        voice_models[voice_name] = torch.load(f'voices/{voice_name}.pt', weights_only=True).to(device)
        logging.info(f"Loaded voice: {voice_name}")
    except Exception as e:
        logging.error(f"Failed to load voice {voice_name}: {str(e)}")

from kokoro import generate, tokenize, VOCAB

KOKORO_API_KEY = os.getenv("KOKORO_API_KEY")
if not KOKORO_API_KEY:
    logging.info("Environment variable KOKORO_API_KEY is not set")
else:
    logging.info(f"Environment variable KOKORO_API_KEY is {KOKORO_API_KEY}")

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

def get_language_from_voice(voice_name: str) -> tuple[str, str]:
    """
    Determine language based on voice name prefix
    Returns a tuple of (display_language, phonemizer_code)
    """
    if voice_name.startswith('a'):
        return "en-us", "a"
    elif voice_name.startswith('b'):
        return "en-gb", "b"
    else:
        return "en-us", "a"  # default to US English

class TTSRequest(BaseModel):
    text: str = Field(..., title="Text to convert to speech")
    phonetics: Optional[str] = Field(
        None,
        title="Custom phonetics",
        description="Custom phonetics string for pronunciation. If not provided, will be generated automatically."
    )
    voice: str = Field(
        "af",
        title="Voice selection",
        description="Select voice to use from available options",
    )
    output_sample_rate: Optional[int] = 24000
    speed: Optional[float] = Field(
        1.0,
        title="Speech speed",
        description="Speech speed factor. 1.0 is normal speed."
    )
    output_format: Optional[str] = "mp3"

@app.get("/voices")
def list_voices():
    """Endpoint to list all available voices"""
    return {
        "voices": [
            {
                "name": voice,
                "language": get_language_from_voice(voice)[0]
            }
            for voice in VOICE_NAMES
        ]
    }

@app.post("/generate")
def generate_speech(request: TTSRequest, api_key: str = Depends(verify_api_key)):
    start = time.perf_counter()
    wav_bytes = BytesIO()
    
    if request.voice not in voice_models:
        raise HTTPException(status_code=400, detail=f"Voice '{request.voice}' not found. Available voices: {', '.join(VOICE_NAMES)}")
    
    voice_model = voice_models[request.voice]
    display_language, phonemizer_code = get_language_from_voice(request.voice)
    
    # Generate audio with phonemizer code
    audio, phonemes = generate(
        kokoro_model, 
        request.text, 
        voice_model, 
        speed=request.speed, 
        # ps=request.phonetics,
        lang=phonemizer_code
    )

    if audio is None:
        raise HTTPException(status_code=500, detail="Failed to generate audio")
        
    # Convert numpy array to wav
    import scipy.io.wavfile as wav
    wav.write(wav_bytes, request.output_sample_rate, (audio * 32767).astype(np.int16))
        
    inference_time = time.perf_counter() - start
    logging.info(f"Generated audio in {inference_time} seconds with voice {request.voice}")
    
    audio, duration_seconds = get_wav_length_from_bytesio(wav_bytes)
    
     # Base64 encode the phonemes to ensure they can be safely transmitted in headers
    encoded_phonemes = base64.b64encode(phonemes.encode('utf-8')).decode('ascii')
    
    headers = {
        "x-inference-time": str(inference_time),
        "x-audio-length": str(duration_seconds),
        "x-realtime-factor": str(duration_seconds / inference_time),
        "x-phonemes-base64": encoded_phonemes,
        "x-language": display_language
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
