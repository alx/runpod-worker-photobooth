#!/usr/bin/env python3

import os
import sys
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
)
from controlnet_aux import (
    OpenposeDetector,
    MidasDetector,
    CannyDetector
)
import insightface
from huggingface_hub import hf_hub_download
from config import (
    MODEL_ID,
    CONTROLNET_MODELS,
    FACE_ANALYSER_MODEL,
    CACHE_DIR
)

def fetch_pretrained_model(model_class, model_name, **kwargs):
    """Fetches a pretrained model from the HuggingFace model hub."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(model_name, **kwargs)
        except Exception as err:
            if attempt < max_retries - 1:
                print(f"Error: {err}. Retrying {attempt + 1}/{max_retries}...")
            else:
                print(f"Failed to download {model_name}: {err}")
                raise

def download_sdxl_models():
    """Download SDXL generation models."""
    print(f"Downloading SDXL model: {MODEL_ID}")

    common_args = {
        "torch_dtype": torch.float16,
        "use_safetensors": True,
        "cache_dir": CACHE_DIR,
    }

    # Download VAE
    print("Downloading VAE...")
    vae = fetch_pretrained_model(
        AutoencoderKL,
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR,
    )

    # Download main model
    print(f"Downloading main model: {MODEL_ID}")

    # Special handling for TurboVisionXL and SDXL-Turbo models
    if "turbo" in MODEL_ID.lower():
        # These models typically don't have a variant
        pipeline = fetch_pretrained_model(
            StableDiffusionXLPipeline,
            MODEL_ID,
            vae=vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            cache_dir=CACHE_DIR,
        )
    else:
        # Standard SDXL models with fp16 variant
        pipeline = fetch_pretrained_model(
            StableDiffusionXLPipeline,
            MODEL_ID,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            cache_dir=CACHE_DIR,
        )

    # Download refiner if not using Turbo model
    if "turbo" not in MODEL_ID.lower():
        print("Downloading SDXL refiner...")
        refiner = fetch_pretrained_model(
            StableDiffusionXLImg2ImgPipeline,
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            cache_dir=CACHE_DIR,
        )

    print("SDXL models downloaded successfully")

def download_controlnet_models():
    """Download ControlNet models."""
    print("Downloading ControlNet models...")

    for control_type, model_id in CONTROLNET_MODELS.items():
        print(f"Downloading ControlNet {control_type}: {model_id}")
        try:
            controlnet = ControlNetModel.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                cache_dir=CACHE_DIR,
            )
            print(f"  {control_type} downloaded successfully")
        except Exception as e:
            print(f"  Warning: Failed to download {control_type}: {e}")

    # Download ControlNet preprocessors
    print("Initializing ControlNet preprocessors...")

    try:
        print("  Loading OpenPose detector...")
        openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    except Exception as e:
        print(f"  Warning: Failed to load OpenPose detector: {e}")

    try:
        print("  Loading depth detector...")
        depth = MidasDetector.from_pretrained("lllyasviel/Annotators")
    except Exception as e:
        print(f"  Warning: Failed to load depth detector: {e}")

    try:
        print("  Loading Canny detector...")
        canny = CannyDetector()
    except Exception as e:
        print(f"  Warning: Failed to load Canny detector: {e}")

    print("ControlNet models downloaded")

def download_insightface_models():
    """Download InsightFace models for face swapping."""
    print("Downloading InsightFace models...")

    # Create checkpoints directory
    os.makedirs("checkpoints/models", exist_ok=True)

    # Download inswapper model
    print("Downloading inswapper_128.onnx...")
    try:
        hf_hub_download(
            repo_id="ashleykleynhans/inswapper",
            filename="inswapper_128.onnx",
            local_dir="checkpoints",
            cache_dir=CACHE_DIR,
        )
        print("  inswapper model downloaded")
    except Exception as e:
        print(f"  Warning: Failed to download inswapper: {e}")

    # Download buffalo_l model
    print("Downloading buffalo_l model...")
    try:
        # Download all buffalo_l model files
        files = [
            "1k3d68.onnx",
            "2d106det.onnx",
            "det_10g.onnx",
            "genderage.onnx",
            "w600k_r50.onnx"
        ]

        for file in files:
            hf_hub_download(
                repo_id="buffalo_l",
                filename=file,
                subfolder="",
                local_dir="checkpoints/models/buffalo_l",
                cache_dir=CACHE_DIR,
            )
        print("  buffalo_l model downloaded")
    except Exception as e:
        # Try alternative download method
        print(f"  Using alternative download for buffalo_l...")
        os.system("cd checkpoints/models && wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && unzip -q buffalo_l.zip")

    print("InsightFace models downloaded")

def download_codeformer_models():
    """Download CodeFormer models for face restoration."""
    print("Downloading CodeFormer models...")

    os.makedirs("CodeFormer/weights/CodeFormer", exist_ok=True)
    os.makedirs("CodeFormer/weights/facelib", exist_ok=True)
    os.makedirs("CodeFormer/weights/realesrgan", exist_ok=True)

    models = {
        "CodeFormer/weights/CodeFormer/codeformer.pth":
            "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "CodeFormer/weights/facelib/detection_Resnet50_Final.pth":
            "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        "CodeFormer/weights/facelib/parsing_parsenet.pth":
            "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",
        "CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth":
            "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth"
    }

    for path, url in models.items():
        if not os.path.exists(path):
            print(f"  Downloading {os.path.basename(path)}...")
            os.system(f"wget -q -O {path} {url}")

    print("CodeFormer models downloaded")

def main():
    """Main function to download all required models."""
    print("=" * 50)
    print("Photobooth Worker - Model Downloader")
    print(f"Selected model: {MODEL_ID}")
    print("=" * 50)

    try:
        # Set HuggingFace cache directory
        os.environ["HF_HOME"] = CACHE_DIR
        os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
        os.environ["HF_HUB_CACHE"] = CACHE_DIR

        # Download all models
        download_sdxl_models()
        download_controlnet_models()
        download_insightface_models()
        download_codeformer_models()

        print("=" * 50)
        print("All models downloaded successfully!")
        print("=" * 50)

    except Exception as e:
        print(f"Error during download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()