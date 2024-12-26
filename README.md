# Kokoro TTS API
Text-To-Speech Inference Server for [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M/tree/main).

## Usage

```
docker compose up

curl -X POST "http://localhost:4321/generate" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world","phonetics": "həˈloʊ wˈɜːld"}' \
     --output audio.mp3
```
