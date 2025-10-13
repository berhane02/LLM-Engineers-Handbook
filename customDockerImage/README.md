# Custom Docker Image for LLM Training

This directory contains a custom Docker image with:
- **Python 3.10** (your requirement)
- **PyTorch 2.5.1** with CUDA 12.1
- **Transformers 4.45.2**
- **Unsloth** for 2-5x faster training
- All training dependencies

## 🚀 Quick Start

### Step 1: Build and Push to ECR

```bash
cd customDockerImage
./build_and_push.sh us-east-2 llm-training-python310 latest
```

This will:
1. Create ECR repository (if needed)
2. Build the Docker image
3. Push to AWS ECR

### Step 2: Update sagemaker.py

Replace the HuggingFace estimator configuration with:

```python
# Get your image URI
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
image_uri = f"{AWS_ACCOUNT_ID}.dkr.ecr.us-east-2.amazonaws.com/llm-training-python310:latest"

# Use custom image instead of pre-built HuggingFace container
from sagemaker.estimator import Estimator

estimator = Estimator(
    image_uri=image_uri,
    role=settings.AWS_ARN_ROLE,
    instance_count=1,
    instance_type="ml.g5.2xlarge",
    hyperparameters=hyperparameters,
    environment={
        "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
        "COMET_API_KEY": settings.COMET_API_KEY,
        "COMET_PROJECT_NAME": settings.COMET_PROJECT,
    },
)
```

### Step 3: Run Training

```bash
poetry run python -m tools.run --run-training
```

## 📦 What's Included

### Base Image
- NVIDIA CUDA 12.1.0 + cuDNN 8
- Ubuntu 22.04
- Python 3.10

### ML Libraries
- PyTorch 2.5.1 (CUDA 12.1)
- Transformers 4.45.2
- Datasets 3.x
- PEFT 0.14.0 (LoRA)
- TRL 0.22.2 (SFT/DPO)
- Accelerate 0.34.0

### Optimizations
- Flash Attention 2.7.4
- BitsAndBytes 0.44.1 (quantization)
- Unsloth (2-5x speedup!)

### Utilities
- Comet ML (experiment tracking)
- SentencePiece (tokenization)
- NumPy, Pandas, etc.

## 🔧 Customization

### Modify Dependencies

Edit `requirements.txt` and rebuild:

```bash
# Edit requirements.txt
vim requirements.txt

# Rebuild and push
./build_and_push.sh us-east-2 llm-training-python310 v2
```

### Change Python Version

Edit `Dockerfile` line 14:
```dockerfile
python3.10  → python3.11
```

### Change PyTorch Version

Edit `Dockerfile` line 34:
```dockerfile
torch==2.5.1 → torch==2.4.0
```

## 🐛 Troubleshooting

### Build Fails

```bash
# Build without cache
docker build --no-cache -t test .

# Check logs
docker build -t test . 2>&1 | tee build.log
```

### Test Locally

```bash
# Build locally
docker build -t llm-training:test .

# Test Python version
docker run --rm llm-training:test python --version

# Test PyTorch
docker run --rm llm-training:test \
  python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Test Unsloth
docker run --rm llm-training:test \
  python -c "import unsloth; print('Unsloth installed!')"
```

### Can't Push to ECR

```bash
# Login manually
aws ecr get-login-password --region us-east-2 | \
  docker login --username AWS --password-stdin \
  ACCOUNT_ID.dkr.ecr.us-east-2.amazonaws.com
```

## 📊 Image Size

Expected size: ~15-20 GB
- CUDA libraries: ~8-10 GB
- PyTorch: ~3-4 GB
- Other dependencies: ~2-3 GB

## 💰 Cost

- **ECR Storage**: ~$0.10/GB/month (~$1.50-2/month)
- **Data Transfer**: Minimal (SageMaker in same region)
- **Build Time**: ~10-15 minutes

## 🎯 Benefits

✅ **Python 3.10** - Your requirement met!
✅ **PyTorch 2.5.1** - Latest features
✅ **Unsloth** - 2-5x faster training
✅ **Full control** - Customize anything
✅ **Reproducible** - Same environment every time

## 📝 Notes

- Image builds for `linux/amd64` platform (SageMaker compatibility)
- Includes SageMaker environment variables
- Compatible with SageMaker Training Jobs
- Flash Attention compilation happens during build (~5-10 min)
- Unsloth installed from latest GitHub version

## 🔗 Next Steps

1. Build and push the image
2. Update `sagemaker.py` to use custom image
3. Run training
4. Monitor in CloudWatch and Comet ML
5. Deploy trained model

Happy Training! 🚀

