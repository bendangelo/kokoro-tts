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

# get voices
curl "http://localhost:4321/voices" \
     -H "Content-Type: application/json"

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
* `voice`: **optional** Select voice to use. See below.
* `speed`: **optional** Speech speed factor. 1.0 is normal speed.
* `output_format`: **optional** mp3 (can use wav).

## Available Voices

```
VOICE_NAMES = [
    'af',  # Default voice (50-50 mix of Bella & Sarah)
    'af_bella', 'af_sarah', 'am_adam', 'am_michael',
    'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis',
    'af_nicole', 'af_sky'
]
```
