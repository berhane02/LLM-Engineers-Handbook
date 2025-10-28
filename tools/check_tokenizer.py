#!/usr/bin/env python3
"""
Check tokenizer files in HuggingFace model.
"""

import sys

from huggingface_hub import HfApi
from loguru import logger

from llm_engineering import settings


def main():
    """Check tokenizer files."""

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

    logger.info(f"Checking tokenizer files for: {model_name}")

    try:
        # Get model info
        model_info = api.model_info(model_name)

        # List all files
        files = [file.rfilename for file in model_info.siblings]

        logger.info(f"\nTotal files in model: {len(files)}")
        logger.info("\nTokenzier files:")
        tokenizer_files = [f for f in files if "tokenizer" in f.lower()]
        for file in tokenizer_files:
            logger.info(f"  ✓ {file}")

        logger.info("\nConfig files:")
        config_files = [f for f in files if "config" in f.lower()]
        for file in config_files:
            logger.info(f"  ✓ {file}")

        logger.info("\nModel weight files:")
        weight_files = [f for f in files if any(ext in f for ext in [".safetensors", ".bin"]) and "adapter" not in f]
        for file in weight_files:
            logger.info(f"  ✓ {file}")

        logger.info("\nLoRA adapter files:")
        adapter_files = [f for f in files if "adapter" in f.lower()]
        for file in adapter_files:
            logger.info(f"  ✓ {file}")

        # Check required tokenizer files
        required_tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
        missing_files = []
        for required_file in required_tokenizer_files:
            if required_file not in files:
                missing_files.append(required_file)

        if missing_files:
            logger.warning(f"\n⚠️  Missing required tokenizer files: {missing_files}")
            logger.info("Run: poetry run python tools/fix_tokenizer.py")
        else:
            logger.success("\n✅ All required tokenizer files are present!")

        # Check for vocab.json which is also needed
        vocab_files = [f for f in files if "vocab.json" in f]
        if not vocab_files:
            logger.warning("⚠️  vocab.json is missing (might be needed)")
        else:
            logger.info("  ✓ vocab.json found")

    except Exception as e:
        logger.error(f"Failed to check model: {e}")


if __name__ == "__main__":
    main()
