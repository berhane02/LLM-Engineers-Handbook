#!/bin/bash

# Build Custom Training Docker Image (Local Build Only)
# Usage: ./build.sh [OPTIONS] [IMAGE_NAME] [IMAGE_TAG]
# Options:
#   --no-cache    Build without using Docker cache (clean rebuild)

set -e

# Parse flags
NO_CACHE=""
while [[ "$1" == --* ]]; do
    case "$1" in
        --no-cache)
            NO_CACHE="--no-cache"
            echo "🔄 Building without cache (clean rebuild)"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default values
IMAGE_NAME=${1:-"llm-training-python310"}
IMAGE_TAG=${2:-"latest"}

echo "================================================"
echo "Building Custom LLM Training Docker Image"
echo "Python 3.10 + PyTorch 2.4.1 + CUDA 12.1"
echo "================================================"
echo "Image Name: $IMAGE_NAME"
echo "Image Tag: $IMAGE_TAG"
if [ -n "$NO_CACHE" ]; then
    echo "Cache: DISABLED (clean rebuild)"
else
    echo "Cache: ENABLED (faster build)"
fi
echo "================================================"

# Build the Docker image
echo "Building Docker image..."
cd "$(dirname "$0")"
docker build \
    $NO_CACHE \
    --platform linux/amd64 \
    -t $IMAGE_NAME:$IMAGE_TAG \
    -f Dockerfile \
    .

echo "================================================"
echo "Build completed successfully!"
echo "================================================"
echo "Local image: $IMAGE_NAME:$IMAGE_TAG"
echo "Image size:"
docker images $IMAGE_NAME:$IMAGE_TAG --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
echo "================================================"
echo ""
echo "To test the image:"
echo "  docker run --rm $IMAGE_NAME:$IMAGE_TAG python --version"
echo "  docker run --rm $IMAGE_NAME:$IMAGE_TAG python -c 'import torch; print(torch.__version__)'"
echo ""
echo "To rebuild without cache (force clean build):"
echo "  ./build.sh --no-cache"
echo ""
echo "To push to ECR:"
echo "  ./push_to_ecr.sh $IMAGE_NAME $IMAGE_TAG"
echo "================================================"

