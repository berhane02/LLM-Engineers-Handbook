#!/bin/bash

# Push Docker Image to AWS ECR
# Usage: ./push_to_ecr.sh [IMAGE_NAME] [IMAGE_TAG] [AWS_REGION]

set -e

# Default values
IMAGE_NAME=${1:-"llm-training-python310"}
IMAGE_TAG=${2:-"latest"}
AWS_REGION=${3:-"us-east-2"}

echo "================================================"
echo "Pushing Docker Image to AWS ECR"
echo "================================================"
echo "Image Name: $IMAGE_NAME"
echo "Image Tag: $IMAGE_TAG"
echo "AWS Region: $AWS_REGION"
echo "================================================"

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "AWS Account ID: $AWS_ACCOUNT_ID"

# ECR repository name (same as image name)
ECR_REPO_NAME=$IMAGE_NAME

# Full ECR repository URI
ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME"
echo "ECR URI: $ECR_URI"

# Check if ECR repository exists, create if it doesn't
echo ""
echo "Checking if ECR repository exists..."
if ! aws ecr describe-repositories --repository-names $ECR_REPO_NAME --region $AWS_REGION 2>&1 > /dev/null; then
    echo "Creating ECR repository: $ECR_REPO_NAME"
    aws ecr create-repository \
        --repository-name $ECR_REPO_NAME \
        --region $AWS_REGION \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256
    echo "✓ ECR repository created"
else
    echo "✓ ECR repository already exists"
fi

# Login to ECR
echo ""
echo "Logging in to Amazon ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_URI
echo "✓ Logged in to ECR"

# Check if local image exists
echo ""
echo "Checking if local image exists..."
if ! docker images $IMAGE_NAME:$IMAGE_TAG | grep -q $IMAGE_NAME; then
    echo "❌ Error: Local image $IMAGE_NAME:$IMAGE_TAG not found"
    echo "Please build the image first:"
    echo "  ./build.sh $IMAGE_NAME $IMAGE_TAG"
    exit 1
fi
echo "✓ Local image found"

# Tag the image for ECR
echo ""
echo "Tagging image for ECR..."
docker tag $IMAGE_NAME:$IMAGE_TAG $ECR_URI:$IMAGE_TAG
echo "✓ Image tagged"

# Push to ECR
echo ""
echo "Pushing image to ECR..."
docker push $ECR_URI:$IMAGE_TAG
echo "✓ Image pushed"

# Also tag and push as 'latest' if a specific tag was provided
if [ "$IMAGE_TAG" != "latest" ]; then
    echo ""
    echo "Also tagging and pushing as 'latest'..."
    docker tag $IMAGE_NAME:$IMAGE_TAG $ECR_URI:latest
    docker push $ECR_URI:latest
    echo "✓ Latest tag pushed"
fi

echo ""
echo "================================================"
echo "Successfully pushed Docker image to ECR!"
echo "================================================"
echo "Image URI: $ECR_URI:$IMAGE_TAG"
if [ "$IMAGE_TAG" != "latest" ]; then
    echo "Latest URI: $ECR_URI:latest"
fi
echo "================================================"
echo ""
echo "Image is ready to use in SageMaker!"
echo "URI: $ECR_URI:$IMAGE_TAG"
echo "================================================"

