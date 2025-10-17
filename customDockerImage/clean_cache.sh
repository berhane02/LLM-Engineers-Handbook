#!/bin/bash

# Clean Docker Cache and Old Images
# Usage: ./clean_cache.sh [IMAGE_NAME]

set -e

IMAGE_NAME=${1:-"llm-training-torch"}

echo "================================================"
echo "Cleaning Docker Cache and Old Images"
echo "================================================"
echo "Target Image: $IMAGE_NAME"
echo "================================================"

# Remove dangling images
echo "Removing dangling images..."
docker image prune -f
echo "✓ Dangling images removed"

# Remove unused containers
echo "Removing stopped containers..."
docker container prune -f
echo "✓ Stopped containers removed"

# Remove unused volumes
echo "Removing unused volumes..."
docker volume prune -f
echo "✓ Unused volumes removed"

# Remove unused networks
echo "Removing unused networks..."
docker network prune -f
echo "✓ Unused networks removed"

# Remove specific image versions (keep latest)
echo "Removing old versions of $IMAGE_NAME..."
docker images $IMAGE_NAME --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}" | \
    grep -v "latest" | \
    awk 'NR>1 {print $3}' | \
    xargs -r docker rmi -f
echo "✓ Old image versions removed"

# Show remaining images
echo ""
echo "Remaining images:"
docker images $IMAGE_NAME --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

echo ""
echo "================================================"
echo "Cache cleanup completed!"
echo "================================================"
echo ""
echo "To rebuild with clean cache:"
echo "  ./build.sh --no-cache"
echo ""
echo "To push to ECR:"
echo "  ./push_to_ecr.sh"
echo "================================================"
