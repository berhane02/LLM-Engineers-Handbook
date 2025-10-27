#!/usr/bin/env python3
"""
Helper script to merge LoRA adapters with base model and upload to HuggingFace.
This creates a full model that vLLM can load.
"""

import sys
import torch
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download, upload_folder, login
from loguru import logger
import shutil
import tempfile

from llm_engineering import settings


def main():
    """Merge LoRA adapters with base model."""

    if not settings.HUGGINGFACE_ACCESS_TOKEN:
        logger.error("HUGGINGFACE_ACCESS_TOKEN is not set.")
        return

    # Login to HuggingFace
    login(token=settings.HUGGINGFACE_ACCESS_TOKEN)

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

    logger.info(f"Merging LoRA adapters for: {model_name}")

    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        try:
            # Download your LoRA model
            logger.info("Downloading LoRA adapters...")
            lora_dir = temp_path / "lora_model"
            snapshot_download(
                repo_id=model_name,
                local_dir=str(lora_dir),
                token=settings.HUGGINGFACE_ACCESS_TOKEN,
            )

            # Check if this is actually a LoRA model
            files = list(lora_dir.glob("*"))
            has_adapter = any("adapter_model" in f.name for f in files if f.is_file())

            if not has_adapter:
                logger.warning("Model doesn't have LoRA adapters. It might already be a full model.")
                return

            logger.info("Found LoRA adapters. Downloading base model...")

            # Download the base model
            base_dir = temp_path / "base_model"
            snapshot_download(
                repo_id="meta-llama/Llama-3.1-8B",
                local_dir=str(base_dir),
                token=settings.HUGGINGFACE_ACCESS_TOKEN,
            )

            logger.info("Loading models...")

            # Load base model
            from transformers import LlamaForCausalLM, LlamaTokenizer
            from peft import PeftModel

            base_model = LlamaForCausalLM.from_pretrained(
                str(base_dir),
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

            # Load LoRA adapters
            logger.info("Loading LoRA adapters...")
            model = PeftModel.from_pretrained(base_model, str(lora_dir))

            # Merge and save
            logger.info("Merging adapters...")
            merged_model = model.merge_and_unload()

            # Save merged model
            output_dir = temp_path / "merged_model"
            logger.info(f"Saving merged model to {output_dir}...")

            merged_model.save_pretrained(
                str(output_dir),
                safe_serialization=True,
                max_shard_size="5GB",
            )

            # Copy tokenizer from LoRA model or base model
            tokenizer = LlamaTokenizer.from_pretrained(str(lora_dir))
            if tokenizer.tokenizer_file is not None:
                # Use LoRA tokenizer
                tokenizer.save_pretrained(str(output_dir))
            else:
                # Use base tokenizer
                tokenizer = LlamaTokenizer.from_pretrained(str(base_dir))
                tokenizer.save_pretrained(str(output_dir))

            logger.success("Merged model saved successfully!")

            # Upload to HuggingFace
            logger.info(f"Uploading merged model to {model_name}...")
            upload_folder(
                repo_id=model_name,
                folder_path=str(output_dir),
                token=settings.HUGGINGFACE_ACCESS_TOKEN,
                commit_message="Merge LoRA adapters into full model for vLLM compatibility",
            )

            logger.success(f"✅ Merged model uploaded to {model_name}!")
            logger.info("Your model should now work with vLLM!")

        except Exception as e:
            logger.error(f"Failed to merge adapters: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
