import time
from io import BytesIO
import base64
import os
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import uvicorn
from __version__ import __version__
from styletts2 import tts
import logging
import magic
from pydub import AudioSegment
from fastapi.responses import StreamingResponse
import tempfile

logging.basicConfig(level=logging.INFO)

host = os.getenv("HOST", "*")
port = os.getenv("PORT", "4321")
port = int(port)

warmup_text = "This is an inference API for StyleTTS2. It is now warming up..."

load_start = time.perf_counter()
model = tts.StyleTTS2()
model.inference(warmup_text)
logging.info(f"Model loaded in {time.perf_counter() - load_start} seconds.")

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
    # Ensure the buffer's position is at the start
    bytes_io.seek(0)
    audio = AudioSegment.from_file(bytes_io, format="wav")

    # Calculate the duration in milliseconds, then convert to seconds
    duration_seconds = len(audio) / 1000.0
    return audio, duration_seconds


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    output_sample_rate: Optional[int] = 24000
    alpha: Optional[float] = 0.3
    beta: Optional[float] = 0.7
    diffusion_steps: Optional[int] = 5
    embedding_scale: Optional[int] = 1
    output_format: Optional[str] = "wav"


@app.get("/hc")
def health_check():
    return {"status": "ok", "version": __version__}


@app.post("/generate")
def generate(request: TTSRequest, background_tasks: BackgroundTasks):
    start = time.perf_counter()
    params = request.model_dump()
    output_format = params["output_format"]
    del params["output_format"]
    if "voice" in params and params["voice"] is not None:
        wav_buffer = process_voice(request.voice)
        params["target_voice_path"] = wav_buffer
    del params["voice"]
    wav_bytes = BytesIO()
    model.inference(
        **params,
        output_wav_file=wav_bytes,
    )
    inference_time = time.perf_counter() - start
    logging.info(f"Generated audio in {inference_time} seconds.")
    audio, duration_seconds = get_wav_length_from_bytesio(wav_bytes)
    background_tasks.add_task(os.remove, wav_buffer)
    headers = {
        "x-inference-time": str(inference_time),
        "x-audio-length": str(duration_seconds),
        "x-realtime-factor": str(duration_seconds / inference_time),
    }
    return_bytes = BytesIO()
    audio.export(return_bytes, format=output_format)
    return_bytes.seek(0)

    return StreamingResponse(
        return_bytes,
        media_type=f"audio/{output_format}",
        headers=headers,
    )


if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port)
