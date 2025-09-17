#!/usr/bin/env python3

import os
import io
import uuid
import base64
import traceback
from typing import List, Union, Optional, Tuple
import numpy as np
import cv2
import torch
from PIL import Image
import insightface
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_logger import RunPodLogger

# Diffusers imports
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
)

# ControlNet preprocessors
from controlnet_aux import (
    OpenposeDetector,
    MidasDetector,
    CannyDetector
)

# Local imports
from config import (
    MODEL_ID,
    CONTROLNET_MODELS,
    FACE_SWAP_MODEL,
    FACE_ANALYSER_MODEL,
    FACE_ANALYSER_ROOT,
    TMP_PATH,
    get_model_config,
    get_scheduler_class
)
from schemas import INPUT_SCHEMA

# Initialize logger
logger = RunPodLogger()

# Global model instances
DIFFUSION_PIPELINE = None
REFINER_PIPELINE = None
CONTROLNET_PIPELINES = {}
PREPROCESSORS = {}
FACE_SWAPPER = None
FACE_ANALYSER = None

def base64_to_image(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image.convert("RGB")

def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')

def load_diffusion_models(model_id: str = None):
    """Load SDXL diffusion models."""
    global DIFFUSION_PIPELINE, REFINER_PIPELINE

    current_model = model_id or MODEL_ID
    model_config = get_model_config(current_model)

    logger.info(f"Loading diffusion model: {current_model}")

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    # Load main pipeline
    common_args = {
        "vae": vae,
        "torch_dtype": torch.float16,
        "use_safetensors": True,
        "local_files_only": True,
    }

    if "turbo" in current_model.lower():
        # Turbo models don't have variant
        DIFFUSION_PIPELINE = StableDiffusionXLPipeline.from_pretrained(
            current_model,
            **common_args
        ).to("cuda")
    else:
        # Standard SDXL models
        DIFFUSION_PIPELINE = StableDiffusionXLPipeline.from_pretrained(
            current_model,
            variant="fp16",
            **common_args
        ).to("cuda")

    # Set scheduler
    scheduler_class = get_scheduler_class(model_config['scheduler'])
    DIFFUSION_PIPELINE.scheduler = scheduler_class.from_config(DIFFUSION_PIPELINE.scheduler.config)

    # Enable optimizations
    DIFFUSION_PIPELINE.enable_xformers_memory_efficient_attention()
    DIFFUSION_PIPELINE.enable_model_cpu_offload()

    # Load refiner if supported
    if model_config.get('use_refiner', False):
        logger.info("Loading SDXL refiner...")
        REFINER_PIPELINE = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            variant="fp16",
            **common_args
        ).to("cuda")

        REFINER_PIPELINE.enable_xformers_memory_efficient_attention()
        REFINER_PIPELINE.enable_model_cpu_offload()

    logger.info("Diffusion models loaded successfully")

def load_controlnet_models():
    """Load ControlNet models and preprocessors."""
    global CONTROLNET_PIPELINES, PREPROCESSORS

    logger.info("Loading ControlNet models...")

    # Load ControlNet models
    for control_type, model_id in CONTROLNET_MODELS.items():
        try:
            controlnet = ControlNetModel.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                local_files_only=True,
            )

            # Create ControlNet pipeline with current diffusion model
            pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                MODEL_ID,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                local_files_only=True,
            ).to("cuda")

            pipeline.enable_xformers_memory_efficient_attention()
            pipeline.enable_model_cpu_offload()

            CONTROLNET_PIPELINES[control_type] = pipeline
            logger.info(f"Loaded ControlNet: {control_type}")

        except Exception as e:
            logger.error(f"Failed to load ControlNet {control_type}: {e}")

    # Load preprocessors
    try:
        PREPROCESSORS['openpose'] = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        logger.info("Loaded OpenPose detector")
    except Exception as e:
        logger.error(f"Failed to load OpenPose: {e}")

    try:
        PREPROCESSORS['depth'] = MidasDetector.from_pretrained("lllyasviel/Annotators")
        logger.info("Loaded depth detector")
    except Exception as e:
        logger.error(f"Failed to load depth detector: {e}")

    try:
        PREPROCESSORS['canny'] = CannyDetector()
        logger.info("Loaded Canny detector")
    except Exception as e:
        logger.error(f"Failed to load Canny detector: {e}")

def load_face_models():
    """Load face swapping models."""
    global FACE_SWAPPER, FACE_ANALYSER

    logger.info("Loading face swapping models...")

    # Load face swapper
    try:
        FACE_SWAPPER = insightface.model_zoo.get_model(FACE_SWAP_MODEL)
        logger.info("Loaded face swapper model")
    except Exception as e:
        logger.error(f"Failed to load face swapper: {e}")

    # Load face analyser
    try:
        FACE_ANALYSER = insightface.app.FaceAnalysis(
            name=FACE_ANALYSER_MODEL,
            root=FACE_ANALYSER_ROOT,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("Loaded face analyser")
    except Exception as e:
        logger.error(f"Failed to load face analyser: {e}")

def get_control_image(image: Image.Image, control_type: str) -> Optional[Image.Image]:
    """Generate control image using specified preprocessor."""
    if control_type not in PREPROCESSORS:
        logger.warning(f"Preprocessor {control_type} not available")
        return None

    try:
        preprocessor = PREPROCESSORS[control_type]
        control_image = preprocessor(image)
        return control_image
    except Exception as e:
        logger.error(f"Failed to generate {control_type} control image: {e}")
        return None

def get_faces(image: Image.Image) -> List:
    """Extract faces from image."""
    if FACE_ANALYSER is None:
        return []

    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    faces = FACE_ANALYSER.get(image_cv)
    return sorted(faces, key=lambda x: x.bbox[0])  # Sort by x coordinate

def swap_faces(source_image: Image.Image, target_image: Image.Image,
               source_indexes: str = "-1", target_indexes: str = "-1") -> Image.Image:
    """Swap faces between source and target images."""
    if FACE_SWAPPER is None or FACE_ANALYSER is None:
        logger.warning("Face swapping models not loaded")
        return target_image

    try:
        # Get faces
        source_faces = get_faces(source_image)
        target_cv = cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR)
        target_faces = get_faces(target_image)

        if not source_faces or not target_faces:
            logger.warning("No faces detected in source or target image")
            return target_image

        # Parse indexes
        if source_indexes == "-1":
            source_indices = list(range(len(source_faces)))
        else:
            source_indices = [int(i) for i in source_indexes.split(",")]

        if target_indexes == "-1":
            target_indices = list(range(len(target_faces)))
        else:
            target_indices = [int(i) for i in target_indexes.split(",")]

        # Perform face swaps
        result_image = target_cv.copy()
        for i, (source_idx, target_idx) in enumerate(zip(source_indices, target_indices)):
            if source_idx < len(source_faces) and target_idx < len(target_faces):
                source_face = source_faces[source_idx]
                target_face = target_faces[target_idx]
                result_image = FACE_SWAPPER.get(result_image, target_face, source_face, paste_back=True)

        # Convert back to PIL
        result_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        return result_pil

    except Exception as e:
        logger.error(f"Face swapping failed: {e}")
        return target_image

def generate_image(prompt: str, negative_prompt: str = None,
                  control_images: dict = None, **kwargs) -> Image.Image:
    """Generate image using SDXL with optional ControlNet."""
    model_config = get_model_config(kwargs.get('model_id'))

    # Prepare generation parameters
    gen_kwargs = {
        "prompt": prompt,
        "height": kwargs.get('height', 1024),
        "width": kwargs.get('width', 1024),
        "num_inference_steps": kwargs.get('num_inference_steps', model_config['num_inference_steps']),
        "guidance_scale": kwargs.get('guidance_scale', model_config['guidance_scale']),
        "num_images_per_prompt": kwargs.get('num_images', 1),
    }

    # Add negative prompt if supported
    if model_config.get('supports_negative_prompt', True) and negative_prompt:
        gen_kwargs["negative_prompt"] = negative_prompt

    # Add seed if provided
    if kwargs.get('seed'):
        generator = torch.Generator(device="cuda").manual_seed(kwargs['seed'])
        gen_kwargs["generator"] = generator

    try:
        # Use ControlNet if control images provided
        if control_images and CONTROLNET_PIPELINES:
            # Use first available control type for now
            control_type = list(control_images.keys())[0]
            if control_type in CONTROLNET_PIPELINES:
                pipeline = CONTROLNET_PIPELINES[control_type]
                gen_kwargs["image"] = control_images[control_type]
                gen_kwargs["controlnet_conditioning_scale"] = kwargs.get('controlnet_conditioning_scale', 0.5)

                result = pipeline(**gen_kwargs)
            else:
                # Fallback to regular generation
                result = DIFFUSION_PIPELINE(**gen_kwargs)
        else:
            # Regular generation
            result = DIFFUSION_PIPELINE(**gen_kwargs)

        generated_image = result.images[0]

        # Apply refiner if available and not using Turbo model
        if REFINER_PIPELINE and model_config.get('use_refiner', False):
            refined_result = REFINER_PIPELINE(
                prompt=prompt,
                image=generated_image,
                num_inference_steps=kwargs.get('refiner_inference_steps', 50),
                strength=kwargs.get('strength', 0.3),
            )
            generated_image = refined_result.images[0]

        return generated_image

    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise

def process_photobooth(job_input: dict) -> dict:
    """Main photobooth processing function."""
    try:
        # Parse input
        source_image = base64_to_image(job_input['image'])
        prompt = job_input['prompt']

        # Generate control images if requested
        control_images = {}
        if job_input.get('use_controlnet', True):
            for control_type in job_input.get('controlnet_types', ['depth', 'openpose']):
                control_image = get_control_image(source_image, control_type)
                if control_image:
                    control_images[control_type] = control_image

        # Generate new image
        logger.info("Generating image...")
        generated_image = generate_image(
            prompt=prompt,
            negative_prompt=job_input.get('negative_prompt'),
            control_images=control_images,
            **job_input
        )

        # Swap faces if requested
        if job_input.get('swap_faces', True):
            logger.info("Swapping faces...")
            final_image = swap_faces(
                source_image=source_image,
                target_image=generated_image,
                source_indexes=job_input.get('source_indexes', "-1"),
                target_indexes=job_input.get('target_indexes', "-1")
            )
        else:
            final_image = generated_image

        # Convert to base64
        output_format = job_input.get('output_format', 'JPEG')
        result_base64 = image_to_base64(final_image, output_format)

        return {
            "image": result_base64,
            "format": output_format
        }

    except Exception as e:
        logger.error(f"Photobooth processing failed: {e}")
        raise

def handler(job):
    """RunPod serverless handler function."""
    try:
        # Validate input
        job_input = job["input"]
        validated_input = validate(job_input, INPUT_SCHEMA)

        if 'errors' in validated_input:
            return {"error": f"Input validation failed: {validated_input['errors']}"}

        # Process photobooth request
        result = process_photobooth(validated_input['validated_input'])

        return result

    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {"error": str(e)}

def initialize_models():
    """Initialize all models on startup."""
    logger.info("Initializing photobooth worker...")

    # Create temp directory
    os.makedirs(TMP_PATH, exist_ok=True)

    # Load all models
    load_diffusion_models()
    load_controlnet_models()
    load_face_models()

    logger.info("All models loaded successfully")

if __name__ == "__main__":
    logger.info("Starting RunPod Photobooth Worker")
    
    # Initialize models
    initialize_models()
    
    # Start serverless worker
    runpod.serverless.start({"handler": handler})
