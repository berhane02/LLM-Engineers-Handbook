# =============================
# Stage 1: Build / compile stage
# =============================
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1

# Install system dependencies for build
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    build-essential git cmake ninja-build curl libffi-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel packaging ninja

# Copy build-time requirements
COPY requirements.txt /tmp/requirements.txt

# Install PyTorch 2.4.1 (better Unsloth compatibility than 2.5.1)
RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Build flash-attn as wheel (compilation happens here)
RUN pip install flash-attn==2.6.3.post1 --no-build-isolation || \
    echo "Flash Attention build failed - continuing without it"

# Install remaining build-time packages
RUN pip install --no-cache-dir -r /tmp/requirements.txt || true

# Install SageMaker Training Toolkit in builder stage (needs build tools)
RUN pip install sagemaker-training

# =============================
# Stage 2: Runtime / final image
# =============================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install minimal runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Copy installed packages from builder stage (includes sagemaker-training)
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Suppress pip root user warning (Docker containers run as root by default)
ENV PIP_ROOT_USER_ACTION=ignore

# SageMaker-specific environment variables
ENV SAGEMAKER_PROGRAM=finetune.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
ENV SAGEMAKER_REGION=us-east-2

# Create SageMaker directories
RUN mkdir -p /opt/ml/code /opt/ml/model /opt/ml/output/data /opt/ml/input

# Set working directory
WORKDIR /opt/ml/code

# Clean up to reduce final image size
RUN rm -rf /root/.cache /tmp/* /var/tmp/*

# Default PATH for SageMaker
ENV PATH="/opt/ml/code:${PATH}"

# Use SageMaker Training Toolkit's train entrypoint for script mode
ENTRYPOINT ["train"]