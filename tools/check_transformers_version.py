#!/usr/bin/env python3
"""Check transformers version from model config."""

from huggingface_hub import hf_hub_download
import json
from loguru import logger

from llm_engineering import settings

config_path = hf_hub_download(
    repo_id="berhaneio/TwinLlama-3.1-8B", filename="config.json", token=settings.HUGGINGFACE_ACCESS_TOKEN
)

with open(config_path) as f:
    config = json.load(f)

transformers_version = config.get("transformers_version", "Not specified")
logger.info(f"Transformers version used for training: {transformers_version}")
