#!/usr/bin/env python3
"""
LLM Fine-tuning Script for SageMaker
This script performs SFT (Supervised Fine-Tuning) or DPO (Direct Preference Optimization) training.
"""

# CRITICAL: Import sys and os FIRST before using them
import sys
import os

# CRITICAL: Add debugging at the very beginning before any imports
print("=" * 80)
print("🚀 SCRIPT STARTUP - BEFORE IMPORTS")
print("=" * 80)
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Script location: {__file__}")
print("=" * 80)

import argparse
import platform
import traceback
from pathlib import Path

# Environment detection for cross-platform compatibility
print("=" * 60)
print("ENVIRONMENT DETECTION")
print("=" * 60)
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.architecture()}")
print(f"Machine: {platform.machine()}")
print(f"Python: {sys.version}")
print(f"Python Platform: {sys.platform}")
print(f"Build Date: {os.environ.get('BUILD_DATE', 'unknown')}")
print(f"Build Version: {os.environ.get('BUILD_VERSION', 'unknown')}")
print("=" * 60)

# CRITICAL: Step-by-step import debugging
print("=" * 60)
print("STEP-BY-STEP IMPORT DEBUGGING")
print("=" * 60)

try:
    print("✅ Step 1: Basic imports successful")
    print("✅ Step 2: Starting Unsloth import check...")
except Exception as e:
    print(f"❌ Step 1-2 failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Try to import Unsloth - make it optional for Python 3.10 compatibility
# Can be disabled with DISABLE_UNSLOTH=1 environment variable for compatibility
print("=" * 60)
print("CHECKING UNSLOTH AVAILABILITY")
print("=" * 60)

try:
    print("✅ Step 3: Starting Unsloth import check...")
    UNSLOTH_AVAILABLE = False
    if os.environ.get("DISABLE_UNSLOTH", "0") != "1":
        try:
            print("Attempting to import Unsloth...")
            from unsloth import PatchDPOTrainer
            print("Unsloth imported successfully. Applying patches...")
            PatchDPOTrainer()
            UNSLOTH_AVAILABLE = True
            print("✅ Unsloth loaded successfully. Using Unsloth optimizations.")
        except ImportError as e:
            UNSLOTH_AVAILABLE = False
            print(f"⚠️  Warning: Unsloth not available ({e}). Training will proceed without Unsloth optimizations.")
        except Exception as e:
            UNSLOTH_AVAILABLE = False
            print(f"⚠️  Warning: Unsloth failed to initialize ({e}). Training will proceed without Unsloth optimizations.")
    else:
        print("Unsloth disabled via DISABLE_UNSLOTH environment variable. Using standard transformers.")
    print("✅ Step 3: Unsloth check completed")
except Exception as e:
    print(f"❌ Step 3 failed: {e}")
    traceback.print_exc()
    sys.exit(1)
print("=" * 60)

from typing import Any, List, Literal, Optional  # noqa: E402

try:
    print("✅ Step 4: Starting PyTorch import...")
    import torch  # noqa
    print("✅ Step 4: PyTorch imported successfully")
except Exception as e:
    print(f"❌ Step 4 failed (PyTorch import): {e}")
    traceback.print_exc()
    sys.exit(1)

# CUDA and PyTorch environment detection
print("=" * 60)
print("CUDA & PYTORCH DETECTION")
print("=" * 60)
try:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        print(f"CUDA capability: {torch.cuda.get_device_capability()}")
    else:
        print("⚠️  CUDA not available - training will use CPU (very slow)")
except Exception as e:
    print(f"❌ Error detecting CUDA/PyTorch: {e}")
print("=" * 60)

try:
    print("✅ Step 5: Starting core dependencies import...")
    print("=" * 60)
    print("IMPORTING CORE DEPENDENCIES")
    print("=" * 60)
    
    print("Importing datasets...")
    from datasets import concatenate_datasets, load_dataset  # noqa: E402
    print("✅ datasets imported successfully")
    
    print("Importing huggingface_hub...")
    from huggingface_hub import HfApi  # noqa: E402
    from huggingface_hub.utils import RepositoryNotFoundError  # noqa: E402
    print("✅ huggingface_hub imported successfully")
    
    print("Importing transformers...")
    from transformers import TextStreamer, TrainingArguments  # noqa: E402
    print("✅ transformers imported successfully")
    
    print("Importing trl...")
    from trl import DPOConfig, DPOTrainer, SFTTrainer  # noqa: E402
    print("✅ trl imported successfully")
    
    print("✅ Step 5: Core dependencies imported successfully")
    
except Exception as e:
    print(f"❌ Step 5 failed (Core dependencies import): {e}")
    traceback.print_exc()
    sys.exit(1)
print("=" * 60)

# Import Unsloth components if available
try:
    print("✅ Step 6: Starting Unsloth components import...")
    if UNSLOTH_AVAILABLE:
        try:
            print("Importing Unsloth components...")
            from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: E402
            from unsloth.chat_templates import get_chat_template  # noqa: E402
            print("✅ Unsloth components imported successfully")
        except Exception as e:
            print(f"❌ Error importing Unsloth components: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Fallback to standard transformers
        try:
            print("Importing standard transformers components...")
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
            print("✅ Standard transformers components imported successfully")

            def is_bfloat16_supported():
                return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        except Exception as e:
            print(f"❌ Error importing standard transformers components: {e}")
            traceback.print_exc()
            sys.exit(1)
    print("✅ Step 6: Unsloth/Transformers components imported successfully")
except Exception as e:
    print(f"❌ Step 6 failed (Unsloth/Transformers components): {e}")
    traceback.print_exc()
    sys.exit(1)

print("=" * 80)
print("🎉 ALL IMPORTS SUCCESSFUL - SCRIPT READY TO RUN")
print("=" * 80)

# Removed Alpaca template - using raw text formatting


def load_model(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
    chat_template: str,
) -> tuple:
    if UNSLOTH_AVAILABLE:
        # Use Unsloth for optimized loading (2-5x faster training)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=None,  # Auto-detect optimal dtype (bfloat16 or float16)
            device_map="auto",  # Proper device mapping for meta tensor handling
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            use_gradient_checkpointing=True,  # Simplified to boolean
            use_rslora=False,  # Use standard LoRA (not rank-stabilized)
            loftq_config=None,  # No LoftQ quantization
        )

        tokenizer = get_chat_template(
            tokenizer,
            chat_template=chat_template,
        )
    else:
        # Use standard transformers + PEFT
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Configure quantization if needed
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,  # Simplified to avoid bfloat16 issues
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,  # Simplified to avoid bfloat16 issues
            )
            model = prepare_model_for_kbit_training(model)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,  # Simplified to avoid bfloat16 issues
            )

        # Apply LoRA
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    return model, tokenizer


def finetune(
    finetuning_type: Literal["sft", "dpo"],
    model_name: str,
    output_dir: str,
    dataset_huggingface_workspace: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = False,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],  # noqa: B006
    chat_template: str = "chatml",
    learning_rate: float = 3e-4,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    beta: float = 0.5,  # Only for DPO
    is_dummy: bool = True,
) -> tuple:
    model, tokenizer = load_model(
        model_name, max_seq_length, load_in_4bit, lora_rank, lora_alpha, lora_dropout, target_modules, chat_template
    )
    EOS_TOKEN = tokenizer.eos_token
    print(f"Setting EOS_TOKEN to {EOS_TOKEN}")  # noqa

    if is_dummy is True:
        num_train_epochs = 1
        print(f"Training in dummy mode. Setting num_train_epochs to '{num_train_epochs}'")  # noqa
        print(f"Training in dummy mode. Reducing dataset size to '400'.")  # noqa

    if finetuning_type == "sft":

        def format_samples_sft(examples):
            text = []
            for instruction, output in zip(examples["instruction"], examples["output"], strict=False):
                # Use raw text formatting without Alpaca template
                message = f"{instruction}\n\n{output}" + EOS_TOKEN
                text.append(message)

            return {"text": text}

        dataset1 = load_dataset(f"{dataset_huggingface_workspace}/llmtwin", split="train")
        dataset2 = load_dataset("mlabonne/FineTome-Alpaca-100k", split="train[:10000]")
        dataset = concatenate_datasets([dataset1, dataset2])
        if is_dummy:
            try:
                dataset = dataset.select(range(400))
            except Exception:
                print("Dummy mode active. Failed to trim the dataset to 400 samples.")  # noqa
        print(f"Loaded dataset with {len(dataset)} samples.")  # noqa

        dataset = dataset.map(format_samples_sft, batched=True, remove_columns=dataset.column_names)
        dataset = dataset.train_test_split(test_size=0.05)

        print("Training dataset example:")  # noqa
        print(dataset["train"][0])  # noqa

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            args=TrainingArguments(
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                fp16=True,  # Simplified to avoid bfloat16 detection issues
                logging_steps=1,
                optim="adamw_torch",  # Changed from adamw_8bit for compatibility
                weight_decay=0.01,
                lr_scheduler_type="linear",
                per_device_eval_batch_size=per_device_train_batch_size,
                warmup_steps=10,
                output_dir=output_dir,
                seed=0,
            ),
        )
    elif finetuning_type == "dpo":
        if UNSLOTH_AVAILABLE:
            PatchDPOTrainer()

        def format_samples_dpo(example):
            # Use raw text formatting without Alpaca template
            example["prompt"] = example["prompt"]
            example["chosen"] = example["chosen"] + EOS_TOKEN
            example["rejected"] = example["rejected"] + EOS_TOKEN

            return {"prompt": example["prompt"], "chosen": example["chosen"], "rejected": example["rejected"]}

        dataset = load_dataset(f"{dataset_huggingface_workspace}/llmtwin-dpo", split="train")
        if is_dummy:
            try:
                dataset = dataset.select(range(400))
            except Exception:
                print("Dummy mode active. Failed to trim the dataset to 400 samples.")  # noqa
        print(f"Loaded dataset with {len(dataset)} samples.")  # noqa

        dataset = dataset.map(format_samples_dpo)
        dataset = dataset.train_test_split(test_size=0.05)

        print("Training dataset example:")  # noqa
        print(dataset["train"][0])  # noqa

        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            beta=beta,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            max_length=max_seq_length // 2,
            max_prompt_length=max_seq_length // 2,
            args=DPOConfig(
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                fp16=True,  # Simplified to avoid bfloat16 detection issues
                optim="adamw_torch",  # Changed from adamw_8bit for compatibility
                weight_decay=0.01,
                lr_scheduler_type="linear",
                per_device_eval_batch_size=per_device_train_batch_size,
                warmup_steps=10,
                output_dir=output_dir,
                eval_steps=50,  # Changed from 0.2 to integer
                logging_steps=1,
                seed=0,
            ),
        )
    else:
        raise ValueError("Invalid finetuning_type. Choose 'sft' or 'dpo'.")

    trainer.train()

    return model, tokenizer


def inference(
    model: Any,
    tokenizer: Any,
    prompt: str = "Write a paragraph to introduce supervised fine-tuning.",
    max_new_tokens: int = 256,
) -> None:
    if UNSLOTH_AVAILABLE:
        model = FastLanguageModel.for_inference(model)
    else:
        model.eval()

    # Use raw text formatting without Alpaca template
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=max_new_tokens, use_cache=True)


def save_model(model: Any, tokenizer: Any, output_dir: str, push_to_hub: bool = False, repo_id: Optional[str] = None):
    if UNSLOTH_AVAILABLE:
        # Use Unsloth's optimized save method
        model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
        if push_to_hub and repo_id:
            print(f"Saving model to '{repo_id}'")  # noqa
            model.push_to_hub_merged(repo_id, tokenizer, save_method="merged_16bit")
    else:
        # Use standard PEFT save method
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        if push_to_hub and repo_id:
            print(f"Saving model to '{repo_id}'")  # noqa
            model.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)


def check_if_huggingface_model_exists(model_id: str, default_value: str = "mlabonne/TwinLlama-3.1-8B") -> str:
    api = HfApi()

    try:
        api.model_info(model_id)
    except RepositoryNotFoundError:
        print(f"Model '{model_id}' does not exist.")  # noqa
        model_id = default_value
        print(f"Defaulting to '{model_id}'")  # noqa
        print("Train your own 'TwinLlama-3.1-8B' to avoid this behavior.")  # noqa

    return model_id


if __name__ == "__main__":
    try:
        print("=" * 60)
        print("STARTING TRAINING SCRIPT")
        print("=" * 60)
        
        parser = argparse.ArgumentParser()

        parser.add_argument("--num_train_epochs", type=int, default=3)
        parser.add_argument("--per_device_train_batch_size", type=int, default=2)
        parser.add_argument("--learning_rate", type=float, default=3e-4)
        parser.add_argument("--dataset_huggingface_workspace", type=str, default="mlabonne")
        parser.add_argument("--model_output_huggingface_workspace", type=str, default="mlabonne")
        parser.add_argument("--is_dummy", type=bool, default=False, help="Flag to reduce the dataset size for testing")
        parser.add_argument(
            "--finetuning_type",
            type=str,
            choices=["sft", "dpo"],
            default="sft",
            help="Parameter to choose the finetuning stage.",
        )

        parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
        parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
        parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])

        args = parser.parse_args()
        
        print(f"Arguments parsed successfully:")
        print(f"  finetuning_type: {args.finetuning_type}")
        print(f"  num_train_epochs: {args.num_train_epochs}")
        print(f"  per_device_train_batch_size: {args.per_device_train_batch_size}")
        print(f"  learning_rate: {args.learning_rate}")
        print(f"  dataset_huggingface_workspace: {args.dataset_huggingface_workspace}")
        print(f"  model_output_huggingface_workspace: {args.model_output_huggingface_workspace}")
        print(f"  is_dummy: {args.is_dummy}")
        print("=" * 60)

        print(f"Num training epochs: '{args.num_train_epochs}'")  # noqa
        print(f"Per device train batch size: '{args.per_device_train_batch_size}'")  # noqa
        print(f"Learning rate: {args.learning_rate}")  # noqa
        print(f"Datasets will be loaded from Hugging Face workspace: '{args.dataset_huggingface_workspace}'")  # noqa
        print(f"Models will be saved to Hugging Face workspace: '{args.model_output_huggingface_workspace}'")  # noqa
        print(f"Training in dummy mode? '{args.is_dummy}'")  # noqa
        print(f"Finetuning type: '{args.finetuning_type}'")  # noqa
        print(f"Output data dir: '{args.output_data_dir}'")  # noqa
        print(f"Model dir: '{args.model_dir}'")  # noqa
        print(f"Number of GPUs: '{args.n_gpus}'")  # noqa
        print("=" * 60)

        if args.finetuning_type == "sft":
            print("Starting SFT training...")  # noqa
            base_model_name = "meta-llama/Llama-3.1-8B"
            print(f"Training from base model '{base_model_name}'")  # noqa

            output_dir_sft = Path(args.model_dir) / "output_sft"
            model, tokenizer = finetune(
                finetuning_type="sft",
                model_name=base_model_name,
                output_dir=str(output_dir_sft),
                dataset_huggingface_workspace=args.dataset_huggingface_workspace,
                num_train_epochs=args.num_train_epochs,
                per_device_train_batch_size=args.per_device_train_batch_size,
                learning_rate=args.learning_rate,
            )
            inference(model, tokenizer)

            sft_output_model_repo_id = f"{args.model_output_huggingface_workspace}/TwinLlama-3.1-8B"
            save_model(model, tokenizer, "model_sft", push_to_hub=True, repo_id=sft_output_model_repo_id)
        elif args.finetuning_type == "dpo":
            print("Starting DPO training...")  # noqa

            sft_base_model_repo_id = f"{args.model_output_huggingface_workspace}/TwinLlama-3.1-8B"
            sft_base_model_repo_id = check_if_huggingface_model_exists(sft_base_model_repo_id)
            print(f"Training from base model '{sft_base_model_repo_id}'")  # noqa

            output_dir_dpo = Path(args.model_dir) / "output_dpo"
            model, tokenizer = finetune(
                finetuning_type="dpo",
                model_name=sft_base_model_repo_id,
                output_dir=str(output_dir_dpo),
                dataset_huggingface_workspace=args.dataset_huggingface_workspace,
                num_train_epochs=1,
                per_device_train_batch_size=args.per_device_train_batch_size,
                learning_rate=2e-6,
                is_dummy=args.is_dummy,
            )
            inference(model, tokenizer)

            dpo_output_model_repo_id = f"{args.model_output_huggingface_workspace}/TwinLlama-3.1-8B-DPO"
            save_model(model, tokenizer, "model_dpo", push_to_hub=True, repo_id=dpo_output_model_repo_id)
        
        print("=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print("=" * 60)
        print("TRAINING FAILED WITH ERROR:")
        print("=" * 60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("=" * 60)
        print("FULL TRACEBACK:")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
        sys.exit(1)
