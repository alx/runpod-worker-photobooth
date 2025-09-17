# RunPod Photobooth Worker

A fast AI-powered photobooth worker that combines **TurboVisionXL** image generation with face swapping to create personalized avatars. This worker processes images in 3-5 steps for ultra-fast generation while maintaining high quality.

## Features

- **TurboVisionXL Integration**: Super fast SDXL generation (3-5 steps)
- **ControlNet Guidance**: Depth and OpenPose control for accurate generation
- **Face Swapping**: InsightFace-powered face replacement
- **Model Flexibility**: Support for multiple SDXL models
- **Optimized Performance**: Memory-efficient with xformers and FP16

## Workflow

1. **Input**: Receive base64 encoded image + text prompt
2. **Control Extraction**: Generate depth maps and OpenPose from source image
3. **Image Generation**: Create new image using TurboVisionXL with ControlNet guidance
4. **Face Swapping**: Replace faces in generated image with source faces
5. **Output**: Return processed image as base64

## Model Selection

### Default Model: TurboVisionXL
```bash
# Uses stablediffusionapi/turbovision_xl by default
docker run -e MODEL_ID="stablediffusionapi/turbovision_xl" your-image
```

**Optimal Settings for TurboVisionXL:**
- Steps: 3-5
- CFG Scale: 1-2.25
- Scheduler: DPM++ SDE

### Alternative Models
```bash
# Standard SDXL (slower but more control)
docker run -e MODEL_ID="stabilityai/stable-diffusion-xl-base-1.0" your-image

# SDXL Turbo (fastest, 1 step)
docker run -e MODEL_ID="stabilityai/sdxl-turbo" your-image

# Any HuggingFace SDXL model
docker run -e MODEL_ID="your-username/custom-sdxl-model" your-image
```

## API Usage

### Input Schema

```json
{
  "image": "base64_encoded_image",
  "prompt": "professional headshot photo, studio lighting",

  // Model selection
  "model_id": "stablediffusionapi/turbovision_xl",

  // Generation parameters
  "num_inference_steps": 5,
  "guidance_scale": 2.0,
  "width": 1024,
  "height": 1024,
  "seed": 42,

  // ControlNet options
  "use_controlnet": true,
  "controlnet_types": ["depth", "openpose"],
  "controlnet_conditioning_scale": 0.5,

  // Face swapping
  "swap_faces": true,
  "source_indexes": "-1",
  "target_indexes": "-1",

  // Output
  "output_format": "JPEG"
}
```

### Response Format

```json
{
  "image": "base64_encoded_result_image",
  "format": "JPEG"
}
```

### Example Request

```python
import requests
import base64

# Load and encode image
with open("portrait.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

payload = {
    "input": {
        "image": image_base64,
        "prompt": "professional business portrait, corporate headshot, studio lighting, high quality",
        "num_inference_steps": 4,
        "guidance_scale": 2.0
    }
}

response = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT/runsync",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json=payload
)

result = response.json()
output_image = base64.b64decode(result["output"]["image"])
```

## Development

### Local Testing

```bash
# Build Docker image
docker build -t photobooth-worker .

# Run locally
docker run --gpus all -p 8000:8000 photobooth-worker

# Test with curl
curl -X POST http://localhost:8000/runsync \
  -H "Content-Type: application/json" \
  -d @test_input.json
```

### Model Switching

Build with different models:
```bash
# Build with TurboVisionXL (default)
docker build -t photobooth-worker .

# Build with standard SDXL
docker build --build-arg MODEL_ID="stabilityai/stable-diffusion-xl-base-1.0" -t photobooth-worker-sdxl .

# Build with SDXL Turbo
docker build --build-arg MODEL_ID="stabilityai/sdxl-turbo" -t photobooth-worker-turbo .
```

### Environment Variables

- `MODEL_ID`: HuggingFace model ID (default: stablediffusionapi/turbovision_xl)
- `HF_HOME`: HuggingFace cache directory
- `PYTORCH_CUDA_ALLOC_CONF`: CUDA memory allocation settings

## Hardware Requirements

- **GPU**: NVIDIA with 16GB+ VRAM (24GB recommended)
- **Memory**: 32GB RAM recommended
- **Storage**: 50GB+ for models and cache

## Performance

- **TurboVisionXL**: 15-30 seconds per image (3-5 steps)
- **Standard SDXL**: 45-90 seconds per image (25+ steps)
- **SDXL Turbo**: 10-20 seconds per image (1 step)

## License

- **Code**: MIT License
- **TurboVisionXL Model**: Non-commercial use only (Stability AI license)
- **Standard SDXL**: CreativeML OpenRAIL-M

## Deployment on RunPod

1. Create GitHub repository with this code
2. Create GitHub release
3. Go to RunPod Hub console
4. Add repository URL
5. Follow UI steps to publish

## Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/your-repo/issues)
- Documentation: See `CLAUDE.md` for development guidance

## Credits

Built using vendor examples:
- `vendor/runpod-worker-inswapper`: Face swapping implementation
- `vendor/worker-sdxl`: SDXL generation pipeline
- `vendor/worker-template`: RunPod worker structure