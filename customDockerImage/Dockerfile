# =============================
# Stage 1: Builder (optional heavy packages compilation)
# =============================
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1

# Install Python 3.10 + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3-venv \
    build-essential git cmake ninja-build curl libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel packaging ninja

# Copy requirements
COPY requirements.txt /tmp/requirements.txt

# Install all packages from requirements
RUN pip install --no-cache-dir -r /tmp/requirements.txt || true

# =============================
# Stage 2: Runtime image
# =============================
FROM ubuntu:22.04

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

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Suppress pip root user warning
ENV PIP_ROOT_USER_ACTION=ignore

# SageMaker environment
ENV SAGEMAKER_PROGRAM=finetune.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
ENV SAGEMAKER_REGION=us-east-2

RUN mkdir -p /opt/ml/code /opt/ml/model /opt/ml/output/data /opt/ml/input
WORKDIR /opt/ml/code
ENV PATH="/opt/ml/code:${PATH}"

# Copy code (your training scripts)
COPY . /opt/ml/code

# Default entrypoint for SageMaker training
ENTRYPOINT ["train"]