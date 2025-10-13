#!/bin/bash

# Build and Push Custom Training Docker Image to AWS ECR
# This is a convenience script that calls build.sh and push_to_ecr.sh
# Usage: ./build_and_push.sh [IMAGE_NAME] [IMAGE_TAG] [AWS_REGION]

set -e

# Default values
IMAGE_NAME=${1:-"llm-training-python310"}
IMAGE_TAG=${2:-"latest"}
AWS_REGION=${3:-"us-east-2"}

echo "================================================"
echo "Build and Push Pipeline"
echo "================================================"
echo "Image Name: $IMAGE_NAME"
echo "Image Tag: $IMAGE_TAG"
echo "AWS Region: $AWS_REGION"
echo "================================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Step 1: Build the image
echo ""
echo "Step 1/2: Building Docker image..."
echo "================================================"
"$SCRIPT_DIR/build.sh" "$IMAGE_NAME" "$IMAGE_TAG"

# Step 2: Push to ECR
echo ""
echo "Step 2/2: Pushing to ECR..."
echo "================================================"
"$SCRIPT_DIR/push_to_ecr.sh" "$IMAGE_NAME" "$IMAGE_TAG" "$AWS_REGION"

echo ""
echo "================================================"
echo "✅ Build and Push Complete!"
echo "================================================"

