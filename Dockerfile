FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app
ENV TZ=etc/UTC DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
  git \
  build-essential \
  ffmpeg \
  libmagic1 \
  espeak-ng

RUN git clone https://huggingface.co/hexgrad/Kokoro-82M
RUN cd Kokoro-82M
RUN pip install -q phonemizer torch transformers scipy munch

COPY kokoro-v0_19.pth /app/
COPY voices/af.pt /app/voices/
COPY models.py /app/
COPY kokoro.py /app/

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

COPY app/ .

CMD ["python", "server.py"]
