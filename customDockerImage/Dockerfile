# =============================
# Stage 1: Builder (compile heavy packages)
# =============================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1

# Install basic build tools and Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3-venv \
    build-essential git cmake ninja-build curl libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel packaging ninja

# =============================
# Install heavy Python packages
# =============================
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    torchvision==0.15.2 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    flash-attn==2.6.1 \
    bitsandbytes==0.44.1 \
    triton==3.1.0 \
    numpy==1.26.4 \
    sentencepiece

# =============================
# Stage 2: Runtime image
# =============================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Copy heavy packages from builder stage
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Suppress pip root warning
ENV PIP_ROOT_USER_ACTION=ignore

# SageMaker env
ENV SAGEMAKER_PROGRAM=finetune.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
ENV SAGEMAKER_REGION=us-east-2

# SageMaker directories
RUN mkdir -p /opt/ml/code /opt/ml/model /opt/ml/output/data /opt/ml/input
WORKDIR /opt/ml/code
ENV PATH="/opt/ml/code:${PATH}"

# =============================
# Runtime packages installed at SageMaker job start
# =============================
COPY requirements_runtime.txt /tmp/requirements_runtime.txt
RUN pip install --no-cache-dir -r /tmp/requirements_runtime.txt

# Default entrypoint for SageMaker training
ENTRYPOINT ["train"]