# SoulX-Singer SVC RunPod Serverless Worker
FROM runpod/pytorch:2.2.0-py3.11-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Clone SoulX-Singer
RUN git clone --depth 1 https://github.com/Soul-AILab/SoulX-Singer.git /app/SoulX-Singer

# Install dependencies
RUN pip install --no-cache-dir \
    runpod \
    requests \
    pedalboard \
    && pip install --no-cache-dir -r /app/SoulX-Singer/requirements.txt

# Download models
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Soul-AILab/SoulX-Singer', local_dir='/app/SoulX-Singer/pretrained_models/SoulX-Singer'); \
print('SoulX-Singer model downloaded')"

RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Soul-AILab/SoulX-Singer-Preprocess', local_dir='/app/SoulX-Singer/pretrained_models/SoulX-Singer-Preprocess'); \
print('Preprocess models downloaded')"

# Copy handler
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
