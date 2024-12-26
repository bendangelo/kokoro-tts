# Kokoro TTS
Text-To-Speech Inference Server for [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M/tree/main).

## Usage

```
docker compose up

curl -X POST "http://localhost:4321/generate" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world"}' \
     --output audio.mp3

# optional phonetics to specify pronunciation

curl -X POST "http://localhost:4321/generate" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world","phonetics": "həˈloʊ wˈɜːld"}' \
     --output audio.mp3

# can optionally set an api key
docker run -e KOKORO_API_KEY=your_secret_key_here .

curl -X POST "http://localhost:4321/generate" \
     -H "API-Key: your_secret_key_here" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world"}' \
     --output audio.mp3
```

## API Options

* `text`: Text to convert to speech.
* `phonetics`: **optional** Custom phonetics string for pronunciation. If not provided, will be generated automatically.
* `voice`: **optional** Select voice to use: 'bella', 'sarah', or 'default'.
* `speed`: **optional** Speech speed factor. 1.0 is normal speed.
* `output_format`: **optional** mp3 (can use wav).
