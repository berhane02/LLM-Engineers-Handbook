from pathlib import Path

from huggingface_hub import HfApi
from loguru import logger

try:
    from sagemaker.estimator import Estimator
    import boto3
except ModuleNotFoundError:
    logger.warning("Couldn't load SageMaker imports. Run 'poetry install --with aws' to support AWS.")

from llm_engineering.settings import settings

finetuning_dir = Path(__file__).resolve().parent

# Note: SageMaker will automatically install requirements.txt from finetuning_dir
# The base Docker image has (BAKED IN):
#   - PyTorch 2.4.1 + CUDA 12.1
#   - Flash Attention 2.6.3 (compiled)
#   - bitsandbytes, triton, numpy, sentencepiece
#   - SageMaker Training Toolkit
# Runtime installation (from requirements.txt):
#   - Transformers, Unsloth, PEFT, TRL, datasets, etc.


def run_finetuning_on_sagemaker(
    finetuning_type: str = "sft",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 3e-4,
    dataset_huggingface_workspace: str = "mlabonne",
    is_dummy: bool = False,
) -> None:
    assert settings.HUGGINGFACE_ACCESS_TOKEN, "Hugging Face access token is required."
    assert settings.AWS_ARN_ROLE, "AWS ARN role is required."

    if not finetuning_dir.exists():
        raise FileNotFoundError(f"The directory {finetuning_dir} does not exist.")

    api = HfApi()
    user_info = api.whoami(token=settings.HUGGINGFACE_ACCESS_TOKEN)
    huggingface_user = user_info["name"]
    logger.info(f"Current Hugging Face user: {huggingface_user}")

    # Get AWS account ID for custom image URI
    sts = boto3.client("sts", region_name=settings.AWS_REGION)
    aws_account_id = sts.get_caller_identity()["Account"]

    # Custom Docker image URI
    image_uri = f"{aws_account_id}.dkr.ecr.{settings.AWS_REGION}.amazonaws.com/llm-training-python310:latest"
    logger.info(f"Using custom Docker image: {image_uri}")

    hyperparameters = {
        "finetuning_type": finetuning_type,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "learning_rate": learning_rate,
        "dataset_huggingface_workspace": dataset_huggingface_workspace,
        "model_output_huggingface_workspace": huggingface_user,
    }
    if is_dummy:
        hyperparameters["is_dummy"] = True

    # Create SageMaker estimator with custom Docker image
    # Using custom image with:
    #   - Python 3.10 + PyTorch 2.4.1 + CUDA 12.1
    #   - Unsloth for 2-5x faster training
    #   - Flash Attention for optimized attention
    #   - 4-bit quantization support
    # SageMaker will automatically install requirements.txt from source_dir at runtime
    estimator = Estimator(
        image_uri=image_uri,
        role=settings.AWS_ARN_ROLE,
        instance_count=1,
        instance_type="ml.g5.2xlarge",  # A10G GPU with 24GB VRAM
        volume_size=100,  # GB for model storage
        max_run=86400,  # 24 hours max
        hyperparameters=hyperparameters,
        source_dir=str(finetuning_dir),
        entry_point="finetune.py",
        environment={
            "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
            "COMET_API_KEY": settings.COMET_API_KEY,
            "COMET_PROJECT_NAME": settings.COMET_PROJECT,
            # Unsloth is enabled by default (auto-detected in finetune.py)
            # To disable: add "DISABLE_UNSLOTH": "1"
        },
    )

    # Start the training job on SageMaker
    logger.info("Starting SageMaker training job with custom Docker image...")
    estimator.fit()


if __name__ == "__main__":
    run_finetuning_on_sagemaker()
