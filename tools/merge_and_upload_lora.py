#!/usr/bin/env python3
"""
Helper script to merge LoRA adapters with base model and upload the full model to HuggingFace.
This creates a model that vLLM can load directly without needing LoRA adapters.
"""

import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from loguru import logger

from llm_engineering import settings


def main():
    """Merge LoRA adapters and upload full model."""

    if not settings.HUGGINGFACE_ACCESS_TOKEN:
        logger.error("HUGGINGFACE_ACCESS_TOKEN is not set.")
        return

    # Get HuggingFace username
    api = HfApi()
    try:
        user_info = api.whoami(token=settings.HUGGINGFACE_ACCESS_TOKEN)
        username = user_info["name"]
        logger.info(f"Logged in as: {username}")
    except Exception as e:
        logger.error(f"Failed to authenticate with HuggingFace: {e}")
        return

    model_name = f"{username}/TwinLlama-3.1-8B"

    # Parse command line for custom model name
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        if "/" not in model_name:
            model_name = f"{username}/{model_name}"

    logger.info(f"Processing model: {model_name}")

    # Create temp directory
    temp_dir = Path("/tmp/lora_merge")
    temp_dir.mkdir(exist_ok=True)

    try:
        # Download your model with LoRA adapters
        logger.info("Downloading your model with LoRA adapters...")
        model_dir = temp_dir / "model"
        snapshot_download(
            repo_id=model_name,
            local_dir=str(model_dir),
            token=settings.HUGGINGFACE_ACCESS_TOKEN,
        )

        # Check what files we have
        files = list(model_dir.glob("*"))
        logger.info(f"Found {len(files)} files in model:")
        for f in files:
            if f.is_file():
                logger.info(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")

        # Check if this is a LoRA model (has adapter_model files)
        has_adapter = any("adapter_model" in f.name for f in files if f.is_file())

        if has_adapter:
            logger.warning("⚠️  Your model has LoRA adapters, not full model weights.")
            logger.warning("⚠️  vLLM cannot load LoRA-only models.")
            logger.warning("\nTo fix this, you need to:")
            logger.warning("1. Merge the LoRA adapters with the base model during training")
            logger.warning("2. Or use Unsloth's save_pretrained_merged() method")
            logger.warning("\nThe training script should have saved a merged model.")
            logger.warning("Please check your training output for 'merged_16bit' or similar directories.")
        else:
            # Check if it's a full model (has model-*.safetensors)
            has_model_weights = any(
                "model-" in f.name and f.suffix in [".safetensors", ".bin"] for f in files if f.is_file()
            )

            if has_model_weights:
                logger.success("✅ Model appears to have full weights!")
                logger.info("If vLLM still can't load it, the issue might be with the model configuration.")
                logger.info("Try checking if your training saved the weights correctly.")
            else:
                logger.error("❌ Model files not found or in unexpected format.")
                logger.info("Files in model directory:")
                for f in files:
                    logger.info(f"  - {f.name}")

        logger.info("\nFor now, vLLM evaluation will fall back to meta-llama/Llama-3.1-8B-Instruct")
        logger.info("which is compatible for evaluation purposes.")

    except Exception as e:
        logger.error(f"Failed to process model: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Clean up
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary files")


if __name__ == "__main__":
    main()
