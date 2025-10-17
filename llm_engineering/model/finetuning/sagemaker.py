from pathlib import Path

from huggingface_hub import HfApi
from loguru import logger

try:
    from sagemaker.estimator import Estimator
    import boto3
except ModuleNotFoundError:
    logger.warning("Couldn't load SageMaker imports. Run 'poetry install --with aws' to support AWS.")

from llm_engineering.settings import settings
from zenml.client import Client

finetuning_dir = Path(__file__).resolve().parent

# Note: SageMaker will automatically install requirements.txt from finetuning_dir
# Using Docker image from config (training.yaml) with (BAKED IN):
#   - PyTorch 2.2.0 + CUDA 12.1
#   - Python 3.10
#   - SageMaker Training Toolkit
#   - All CUDA libraries (Flash Attention compiled)
#   - transformers, tokenizers, huggingface-hub, numpy, accelerate, datasets, etc.
# Runtime installation (from requirements.txt):
#   - Only missing packages: sentencepiece, peft, trl, rich, pydantic, comet-ml, unsloth


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

    # Use AWS public SageMaker PyTorch image from ZenML config
    # PyTorch 2.2.0 + Python 3.10 + CUDA 12.1
    try:
        # Get Docker parent image from ZenML configuration
        zenml_client = Client()
        docker_config = zenml_client.active_stack.components.get("docker")
        image_uri = docker_config.configuration.parent_image
        logger.info(f"Using Docker image from ZenML config: {image_uri}")
    except Exception as e:
        # Fallback to hardcoded image if ZenML config fails
        image_uri = "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:2.2.0-gpu-py310-cu121-ubuntu20.04-sagemaker"
        logger.warning(f"Failed to get image from ZenML config ({e}), using fallback: {image_uri}")

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

    # Create SageMaker estimator with Docker image from config
    # Using image from training.yaml with:
    #   - Python 3.10 + PyTorch 2.2.0 + CUDA 12.1
    #   - All CUDA libraries (Flash Attention compiled)
    #   - SageMaker Training Toolkit
    #   - Most ML packages pre-installed
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
    logger.info("Starting SageMaker training job with Docker image from config...")
    estimator.fit()


if __name__ == "__main__":
    run_finetuning_on_sagemaker()
