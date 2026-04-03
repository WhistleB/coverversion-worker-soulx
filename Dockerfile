# SoulX-Singer SVC RunPod Serverless Worker
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Clone SoulX-Singer
RUN git clone --depth 1 https://github.com/Soul-AILab/SoulX-Singer.git /app/SoulX-Singer

# Install dependencies (skip torch/torchaudio to keep base image's torch 2.4 + CUDA 12.4)
RUN pip install --no-cache-dir \
    runpod \
    requests \
    pedalboard

# Install SoulX-Singer deps WITHOUT overwriting torch
RUN grep -v -E "^torch==|^torchaudio==" /app/SoulX-Singer/requirements.txt > /tmp/reqs_notorch.txt \
    && pip install --no-cache-dir -r /tmp/reqs_notorch.txt

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
