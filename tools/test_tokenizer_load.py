#!/usr/bin/env python3
"""
Test if tokenizer can be loaded from HuggingFace model.
"""

from transformers import AutoTokenizer
from loguru import logger
import sys

from llm_engineering import settings


def main():
    """Test tokenizer loading."""

    if not settings.HUGGINGFACE_ACCESS_TOKEN:
        logger.error("HUGGINGFACE_ACCESS_TOKEN is not set.")
        return

    model_name = "berhaneio/TwinLlama-3.1-8B"

    # Parse command line for custom model name
    if len(sys.argv) > 1:
        model_name = sys.argv[1]

    logger.info(f"Testing tokenizer loading for: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=settings.HUGGINGFACE_ACCESS_TOKEN, trust_remote_code=True
        )
        logger.success("✅ Tokenizer loaded successfully!")

        # Test encoding
        test_text = "Hello, world!"
        tokens = tokenizer(test_text)
        logger.info(f"Test encoding: '{test_text}' -> {len(tokens['input_ids'])} tokens")

    except Exception as e:
        logger.error(f"❌ Failed to load tokenizer: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
