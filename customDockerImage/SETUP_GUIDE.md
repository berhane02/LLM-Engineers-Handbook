# Custom Docker Image Setup Guide

Complete guide to build and use your custom training image with Python 3.10 + PyTorch 2.5.1 + Unsloth.

## 🎯 What This Solves

**Problem**: SageMaker's pre-built containers with PyTorch 2.5.1 only support Python 3.11.

**Solution**: Custom Docker image with Python 3.10 + PyTorch 2.5.1 + all your exact package versions.

## 📋 Step-by-Step Setup

### Step 1: Build and Push Docker Image

```bash
cd customDockerImage

# Build and push to ECR
./build_and_push.sh us-east-2 llm-training-python310 latest
```

**What this does:**
- Creates ECR repository `llm-training-python310`
- Builds Docker image with Python 3.10 + PyTorch 2.5.1
- Pushes to your AWS ECR
- Takes ~10-15 minutes (Flash Attention compilation)

### Step 2: Update sagemaker.py

You have two options:

#### Option A: Replace the entire file (easiest)

```bash
# Backup original
cp ../llm_engineering/model/finetuning/sagemaker.py ../llm_engineering/model/finetuning/sagemaker.py.backup

# Use custom image version
cp sagemaker_custom_image.py ../llm_engineering/model/finetuning/sagemaker.py
```

#### Option B: Manual update (more control)

Edit `llm_engineering/model/finetuning/sagemaker.py`:

```python
# After imports, add:
import boto3

# In run_finetuning_on_sagemaker function, replace HuggingFace estimator with:

# Get AWS account ID
sts = boto3.client('sts', region_name=settings.AWS_REGION)
aws_account_id = sts.get_caller_identity()['Account']

# Custom Docker image URI
image_uri = f"{aws_account_id}.dkr.ecr.{settings.AWS_REGION}.amazonaws.com/llm-training-python310:latest"

# Use custom Estimator instead of HuggingFace
from sagemaker.estimator import Estimator

estimator = Estimator(
    image_uri=image_uri,
    role=settings.AWS_ARN_ROLE,
    instance_count=1,
    instance_type="ml.g5.2xlarge",
    volume_size=100,
    hyperparameters=hyperparameters,
    source_dir=str(finetuning_dir),
    entry_point="finetune.py",
    environment={
        "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
        "COMET_API_KEY": settings.COMET_API_KEY,
        "COMET_PROJECT_NAME": settings.COMET_PROJECT,
    },
)

# Start training
estimator.fit()
```

### Step 3: Run Training

```bash
poetry run python -m tools.run --run-training
```

## 🧪 Testing

### Test Build Locally

```bash
cd customDockerImage

# Build locally
docker build -t llm-training:test .

# Test Python version
docker run --rm llm-training:test python --version
# Should show: Python 3.10.x

# Test PyTorch
docker run --rm llm-training:test python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
"

# Test all packages
docker run --rm llm-training:test python -c "
import torch
import transformers
import peft
import trl
import unsloth
print('✓ All packages imported successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
"
```

### Test on SageMaker

Create a test script `test_training.py`:

```python
from llm_engineering.model.finetuning.sagemaker import run_finetuning_on_sagemaker

run_finetuning_on_sagemaker(
    finetuning_type="sft",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    learning_rate=3e-4,
    is_dummy=True,  # Use small dataset for testing
)
```

Run it:
```bash
poetry run python test_training.py
```

## 🔄 Update Workflow

When you need to update dependencies:

```bash
cd customDockerImage

# 1. Edit requirements.txt
vim requirements.txt

# 2. Rebuild with new version tag
./build_and_push.sh us-east-2 llm-training-python310 v1.1

# 3. Update image tag in sagemaker.py
#    Change: llm-training-python310:latest
#    To:     llm-training-python310:v1.1

# 4. Run training
poetry run python -m tools.run --run-training
```

## 📊 Configuration Summary

| Component | Version | Why |
|-----------|---------|-----|
| Python | 3.10 | Your requirement |
| PyTorch | 2.5.1 | Latest, best features |
| CUDA | 12.1 | GPU acceleration |
| Transformers | 4.45.2 | Model support |
| Unsloth | Latest | 2-5x speedup |
| Flash Attention | 2.7.4 | Faster attention |

## ⚠️ Important Notes

1. **Build Time**: First build takes 10-15 minutes (Flash Attention compilation)
2. **Image Size**: ~15-20 GB (CUDA + PyTorch + dependencies)
3. **ECR Costs**: ~$2/month for storage
4. **Compatibility**: Works with SageMaker Training Jobs
5. **Updates**: Rebuild and push when changing dependencies

## 🆘 Troubleshooting

### "Repository does not exist"
```bash
aws ecr create-repository \
  --repository-name llm-training-python310 \
  --region us-east-2
```

### "No basic auth credentials"
```bash
aws ecr get-login-password --region us-east-2 | \
  docker login --username AWS --password-stdin \
  $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-2.amazonaws.com
```

### "Flash attention build failed"
Edit `requirements.txt` and comment out:
```
# flash-attn==2.7.4.post1
```

Training will still work, just slightly slower.

### "Unsloth not compatible"
Edit `requirements.txt` and comment out:
```
# unsloth[torch] @ git+https://github.com/unslothai/unsloth.git
```

Training will work with standard PEFT (slower but functional).

## 🎉 Success!

Once built and pushed, your training will use:
- ✅ Python 3.10 (your requirement!)
- ✅ PyTorch 2.5.1 (latest)
- ✅ Unsloth (2-5x faster)
- ✅ All your exact package versions
- ✅ Full control over the environment

Happy Training! 🚀

