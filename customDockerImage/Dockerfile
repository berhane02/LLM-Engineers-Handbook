# =============================
# Ultra-lightweight Docker image for SageMaker
# =============================
FROM ubuntu:22.04

# Build arguments for cache busting
ARG BUILD_DATE
ARG BUILD_VERSION
ENV BUILD_DATE=${BUILD_DATE}
ENV BUILD_VERSION=${BUILD_VERSION}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install minimal dependencies + build tools for sagemaker-training
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip git curl ca-certificates \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install SagePy compatible NumPy first, then SageMaker Training Toolkit
RUN pip install "numpy>=1.24.0,<2.0.0"
RUN pip install sagemaker-training

# Copy requirements.txt from finetuning directory for runtime installation
COPY llm_engineering/model/finetuning/requirements.txt /opt/ml/code/requirements.txt

# Suppress pip root user warning
ENV PIP_ROOT_USER_ACTION=ignore

# SageMaker environment
ENV SAGEMAKER_PROGRAM=finetune.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
ENV SAGEMAKER_REGION=us-east-2

RUN mkdir -p /opt/ml/code /opt/ml/model /opt/ml/output/data /opt/ml/input
WORKDIR /opt/ml/code
ENV PATH="/opt/ml/code:${PATH}"

# Default entrypoint for SageMaker training
ENTRYPOINT ["train"]
