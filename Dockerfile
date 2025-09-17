# Multi-stage build for RunPod Photobooth Worker
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 as base

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    SHELL=/bin/bash \
    MODEL_ID=stablediffusionapi/turbovision_xl \
    HF_HOME=/models \
    TRANSFORMERS_CACHE=/models \
    HF_HUB_CACHE=/models

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install system dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip \
        git \
        git-lfs \
        wget \
        curl \
        unzip \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1-mesa-glx \
        libgoogle-perftools-dev \
        ffmpeg \
        procps && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Set Python 3.11 as default
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3

# Install uv for faster package installation
RUN pip install uv

# Create virtual environment
ENV PATH="/.venv/bin:${PATH}"
RUN uv venv --python 3.11 /.venv

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN uv pip install torch --extra-index-url https://download.pytorch.org/whl/cu121 && \
    uv pip install -r requirements.txt

# Copy source code
COPY config.py schemas.py handler.py download_weights.py ./

# Create model directories
RUN mkdir -p /models checkpoints/models

# Download models (with model selection support)
ARG MODEL_ID=stablediffusionapi/turbovision_xl
ENV MODEL_ID=${MODEL_ID}
RUN python download_weights.py

# Copy startup script
COPY start.sh .
RUN chmod +x start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start the worker
CMD ["./start.sh"]