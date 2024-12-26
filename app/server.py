import time
from io import BytesIO
import base64
import os
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
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

warmup_text = "This is an inference API for TTS. It is now warming up..."

load_start = time.perf_counter()
logging.info(f"Models loaded in {time.perf_counter() - load_start} seconds.")

app = FastAPI()

def process_voice(voice_sample: str):
    audio_data = base64.b64decode(voice_sample)
    audio_buffer = BytesIO(audio_data)
    file_type = magic.from_buffer(audio_buffer.read(2048), mime=True)
    if file_type == "video/mp4":
        file_type = "m4a"
    audio_buffer.seek(0)
    audio = AudioSegment.from_file(audio_buffer, format=file_type)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio.export(f.name, format="wav")
        return f.name

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
    voice_sample: Optional[str] = Field(
        None,
        title="Base64 encoded voice sample",
        description="If provided, the model will attempt to match the voice of the provided sample. 3-5s of sample audio is recommended.",
    )
    output_sample_rate: Optional[int] = 24000
    speed: Optional[float] = Field(
        1.0,
        title="Speech speed",
        description="Speech speed factor. 1.0 is normal speed."
    )
    alpha: Optional[float] = Field(0.3, title="Alpha (StyleTTS2 only)")
    beta: Optional[float] = Field(0.7, title="Beta (StyleTTS2 only)")
    diffusion_steps: Optional[int] = Field(5, title="Diffusion steps (StyleTTS2 only)")
    embedding_scale: Optional[float] = Field(1, title="Embedding scale (StyleTTS2 only)")
    output_format: Optional[str] = "mp3"

@app.post("/generate")
def generate_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    start = time.perf_counter()
    wav_bytes = BytesIO()
    
    # Get the appropriate voice model
    voice_model = get_voice_model(request.voice)
    
    # Initialize audio variable
    audio = None
    
    if request.phonetics:
        # Use provided phonetics
        tokens = tokenize(request.phonetics)
        if not tokens:
            raise HTTPException(status_code=400, detail="Invalid phonetics string")
        # Generate audio using phonetics
        audio, _ = generate(kokoro_model, request.text, voice_model, speed=request.speed, phonetics=request.phonetics)
    else:
        # Generate phonetics from text
        audio, phonetics = generate(kokoro_model, request.text, voice_model, speed=request.speed)
    
    if audio is None:
        raise HTTPException(status_code=500, detail="Failed to generate audio")
        
    # Convert numpy array to wav
    import scipy.io.wavfile as wav
    wav.write(wav_bytes, 24000, (audio * 32767).astype(np.int16))
        
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
