#!/bin/bash

set -e

echo "======================================"
echo "RunPod Photobooth Worker Build Script"
echo "======================================"

# Default values
IMAGE_NAME="photobooth-worker"
MODEL_ID="stablediffusionapi/turbovision_xl"
PUSH_TO_REGISTRY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_ID="$2"
            shift 2
            ;;
        --image-name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --push)
            PUSH_TO_REGISTRY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL_ID        Set the model ID (default: stablediffusionapi/turbovision_xl)"
            echo "  --image-name NAME       Set the Docker image name (default: photobooth-worker)"
            echo "  --push                  Push to registry after build"
            echo "  --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                           # Build with TurboVisionXL"
            echo "  $0 --model stabilityai/sdxl-turbo          # Build with SDXL Turbo"
            echo "  $0 --model stabilityai/stable-diffusion-xl-base-1.0  # Build with standard SDXL"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Model ID: $MODEL_ID"
echo "  Image Name: $IMAGE_NAME"
echo "  Push to Registry: $PUSH_TO_REGISTRY"
echo ""

# Build Docker image
echo "Building Docker image..."
docker build \
    --build-arg MODEL_ID="$MODEL_ID" \
    -t "$IMAGE_NAME" \
    .

echo "Build completed successfully!"

# Push to registry if requested
if [ "$PUSH_TO_REGISTRY" = true ]; then
    echo "Pushing to registry..."
    docker push "$IMAGE_NAME"
    echo "Push completed!"
fi

echo ""
echo "======================================"
echo "Build Summary"
echo "======================================"
echo "Image: $IMAGE_NAME"
echo "Model: $MODEL_ID"
echo ""
echo "To run locally:"
echo "  docker run --gpus all -p 8000:8000 $IMAGE_NAME"
echo ""
echo "To test:"
echo "  curl -X POST http://localhost:8000/runsync -H 'Content-Type: application/json' -d @test_input.json"