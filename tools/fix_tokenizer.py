#!/usr/bin/env python3
"""
Helper script to fix missing tokenizer files in your HuggingFace model.
Downloads the official Llama 3.1 8B tokenizer and uploads it to your model.
"""

import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download, upload_file
from loguru import logger

from llm_engineering import settings


def main():
    """Fix tokenizer files for your model."""

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

    logger.info(f"Fixing tokenizer for: {model_name}")

    # Create temp directory
    temp_dir = Path("/tmp/tokenizer_fix")
    temp_dir.mkdir(exist_ok=True)

    try:
        # Download official Llama 3.1 8B Instruct tokenizer
        logger.info("Downloading official Llama 3.1 8B Instruct tokenizer...")
        snapshot_download(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            local_dir=str(temp_dir / "official"),
            token=settings.HUGGINGFACE_ACCESS_TOKEN,
            allow_patterns=[
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "special_tokens_map.json",
                "added_tokens.json",
            ],
        )

        logger.success("Downloaded tokenizer files successfully")

        # List what we got
        official_dir = temp_dir / "official"
        tokenizer_files = list(official_dir.glob("*"))
        logger.info(f"Found {len(tokenizer_files)} tokenizer files")

        # Upload each tokenizer file to your model
        logger.info("Uploading tokenizer files to your model...")
        for file_path in tokenizer_files:
            if file_path.is_file():
                logger.info(f"Uploading {file_path.name}...")
                upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=file_path.name,
                    repo_id=model_name,
                    token=settings.HUGGINGFACE_ACCESS_TOKEN,
                    repo_type="model",
                    commit_message=f"Add missing tokenizer file: {file_path.name}",
                )

        logger.success(f"✅ Tokenizer files uploaded successfully to {model_name}!")
        logger.info("Your model should now work with vLLM.")

    except Exception as e:
        logger.error(f"Failed to fix tokenizer: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Clean up
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary files")


if __name__ == "__main__":
    main()
