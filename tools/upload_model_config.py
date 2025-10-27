#!/usr/bin/env python3
"""
Helper script to generate and upload config.json to your HuggingFace model.
This ensures your trained model can be used with vLLM and other frameworks.

Usage:
    python tools/upload_model_config.py [--model-name MODEL_NAME] [--all] [--custom-config PATH]
"""

import json
import sys
from pathlib import Path
from huggingface_hub import HfApi, upload_file
from loguru import logger

from llm_engineering import settings


# Standard Llama 3.1 8B configuration
LLAMA_3_1_8B_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "max_position_embeddings": 8192,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 4096,
        "rope_type": "llama3",
    },
    "rope_theta": 500000.0,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.44.0",
    "unsloth_version": "2024.9",
    "use_cache": True,
    "vocab_size": 128256,
}


def get_user_models(username: str) -> list[str]:
    """Get list of models from HuggingFace user."""
    api = HfApi()
    try:
        models = api.list_models(author=username)
        return [model.id for model in models]
    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")
        return []


def upload_config_to_hf(model_name: str, config: dict, token: str) -> bool:
    """Upload config.json to HuggingFace model repository."""
    try:
        # Save config to temporary file
        temp_path = Path("/tmp/config.json")
        with open(temp_path, "w") as f:
            json.dump(config, f, indent=2)

        # Upload to HuggingFace
        upload_file(
            path_or_fileobj=str(temp_path),
            path_in_repo="config.json",
            repo_id=model_name,
            token=token,
            repo_type="model",
            commit_message="Add model configuration for vLLM compatibility",
        )

        logger.success(f"Successfully uploaded config.json to {model_name}")
        temp_path.unlink()  # Clean up
        return True

    except Exception as e:
        logger.error(f"Failed to upload config.json: {e}")
        return False


def main():
    """Main entry point for the script."""
    # Parse command line arguments
    model_name = None
    upload_all = False
    custom_config_path = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--model-name" and i + 1 < len(args):
            model_name = args[i + 1]
            i += 2
        elif args[i] == "--all":
            upload_all = True
            i += 1
        elif args[i] == "--custom-config" and i + 1 < len(args):
            custom_config_path = args[i + 1]
            i += 2
        elif args[i] in ["--help", "-h"]:
            print(__doc__)
            sys.exit(0)
        else:
            logger.error(f"Unknown argument: {args[i]}")
            logger.info(
                "Usage: python tools/upload_model_config.py [--model-name MODEL_NAME] [--all] [--custom-config PATH]"
            )
            sys.exit(1)

    # Execute main logic
    if not settings.HUGGINGFACE_ACCESS_TOKEN:
        logger.error("HUGGINGFACE_ACCESS_TOKEN is not set. Please set it in your environment or .env file.")
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

    # Load custom config if provided
    if custom_config_path:
        with open(custom_config_path, "r") as f:
            config = json.load(f)
        logger.info(f"Using custom config from {custom_config_path}")
    else:
        config = LLAMA_3_1_8B_CONFIG
        logger.info("Using default Llama 3.1 8B config")

    # Handle different scenarios
    if upload_all:
        # Upload to all TwinLlama models
        models = get_user_models(username)
        twin_models = [m for m in models if "TwinLlama" in m]

        if not twin_models:
            logger.warning("No TwinLlama models found for your account.")
            return

        logger.info(f"Found {len(twin_models)} TwinLlama model(s)")
        for model_id in twin_models:
            logger.info(f"Uploading config to {model_id}...")
            upload_config_to_hf(model_id, config, settings.HUGGINGFACE_ACCESS_TOKEN)

    elif model_name:
        # Upload to specific model
        if "/" not in model_name:
            model_name = f"{username}/{model_name}"

        logger.info(f"Uploading config to {model_name}...")
        success = upload_config_to_hf(model_name, config, settings.HUGGINGFACE_ACCESS_TOKEN)

        if success:
            logger.success(f"✅ Config uploaded successfully!")
            logger.info(f"You can now use {model_name} in the evaluation pipeline.")
        else:
            logger.error("Failed to upload config.")

    else:
        # List available models
        logger.info("Listing your models...")
        models = get_user_models(username)

        if not models:
            logger.info("No models found for your account.")
            return

        twin_models = [m for m in models if "TwinLlama" in m]

        if twin_models:
            logger.info(f"\nYour TwinLlama models:")
            for i, model_id in enumerate(twin_models, 1):
                logger.info(f"  {i}. {model_id}")

            logger.info(f"\nTo upload config to a specific model, run:")
            logger.info(f"  python tools/upload_model_config.py --model-name {twin_models[0]}")
            logger.info(f"\nOr upload to all TwinLlama models:")
            logger.info(f"  python tools/upload_model_config.py --all")
        else:
            logger.info(f"Found {len(models)} models, but none are TwinLlama models.")
            logger.info(f"List of all models:")
            for i, model_id in enumerate(models[:10], 1):
                logger.info(f"  {i}. {model_id}")
            if len(models) > 10:
                logger.info(f"  ... and {len(models) - 10} more")


if __name__ == "__main__":
    main()
