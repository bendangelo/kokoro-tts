FROM --platform=$BUILDPLATFORM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app
ENV TZ=etc/UTC DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    build-essential \
    ffmpeg \
    libmagic1 \
    espeak-ng

# Clone Kokoro and install its dependencies first
WORKDIR /app/Kokoro-82M
RUN git lfs install
RUN git clone https://huggingface.co/hexgrad/Kokoro-82M .

# Switch back to app directory and install other dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

# Copy application code
COPY app/ .

# Create symbolic links to Kokoro files
RUN ln -s /app/Kokoro-82M/kokoro.py . && \
    ln -s /app/Kokoro-82M/models.py . && \
    ln -s /app/Kokoro-82M/kokoro-v0_19.pth . && \
    ln -s /app/Kokoro-82M/plbert.py . && \
    ln -s /app/Kokoro-82M/istftnet.py . && \
    ln -s /app/Kokoro-82M/voices . && \
    ln -s /app/Kokoro-82M/config.json .

EXPOSE 4321

CMD ["python", "server.py"]
