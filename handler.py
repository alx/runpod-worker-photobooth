#!/usr/bin/env python3

import os
import io
import base64
import traceback
from typing import List, Union, Optional, Dict, Any
import numpy as np
import cv2
import torch
from PIL import Image

# RunPod imports
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_logger import RunPodLogger

# Initialize logger first
logger = RunPodLogger()

# Global model instances
DIFFUSION_PIPELINE = None
REFINER_PIPELINE = None
CONTROLNET_PIPELINES = {}
PREPROCESSORS = {}
FACE_SWAPPER = None
FACE_ANALYSER = None
MODELS_LOADED = False

def safe_import_and_load_models():
    """Safely import dependencies and load models with error handling."""
    global DIFFUSION_PIPELINE, REFINER_PIPELINE, CONTROLNET_PIPELINES
    global PREPROCESSORS, FACE_SWAPPER, FACE_ANALYSER, MODELS_LOADED

    if MODELS_LOADED:
        return True

    try:
        logger.info("Starting model initialization...")

        # Import dependencies
        try:
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
            logger.info("Dependencies imported successfully")
        except ImportError as e:
            logger.error(f"Failed to import dependencies: {e}")
            return False

        # Create temp directory
        os.makedirs(TMP_PATH, exist_ok=True)

        # Load diffusion models
        try:
            logger.info("Loading diffusion models...")
            model_config = get_model_config(MODEL_ID)

            # Load VAE
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=torch.float16,
            )

            # Load main pipeline
            common_args = {
                "vae": vae,
                "torch_dtype": torch.float16,
                "use_safetensors": True,
            }

            if "turbo" in MODEL_ID.lower():
                DIFFUSION_PIPELINE = StableDiffusionXLPipeline.from_pretrained(
                    MODEL_ID,
                    **common_args
                ).to("cuda")
            else:
                DIFFUSION_PIPELINE = StableDiffusionXLPipeline.from_pretrained(
                    MODEL_ID,
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
        except Exception as e:
            logger.error(f"Failed to load diffusion models: {e}")

        # Load ControlNet models (non-critical)
        try:
            logger.info("Loading ControlNet models...")
            for control_type, model_id in CONTROLNET_MODELS.items():
                try:
                    controlnet = ControlNetModel.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                    )

                    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                        MODEL_ID,
                        controlnet=controlnet,
                        torch_dtype=torch.float16,
                    ).to("cuda")

                    pipeline.enable_xformers_memory_efficient_attention()
                    pipeline.enable_model_cpu_offload()

                    CONTROLNET_PIPELINES[control_type] = pipeline
                    logger.info(f"Loaded ControlNet: {control_type}")
                except Exception as e:
                    logger.warn(f"Failed to load ControlNet {control_type}: {e}")

            # Load preprocessors
            try:
                PREPROCESSORS['openpose'] = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
                logger.info("Loaded OpenPose detector")
            except Exception as e:
                logger.warn(f"Failed to load OpenPose: {e}")

            try:
                PREPROCESSORS['depth'] = MidasDetector.from_pretrained("lllyasviel/Annotators")
                logger.info("Loaded depth detector")
            except Exception as e:
                logger.warn(f"Failed to load depth detector: {e}")

            try:
                PREPROCESSORS['canny'] = CannyDetector()
                logger.info("Loaded Canny detector")
            except Exception as e:
                logger.warn(f"Failed to load Canny detector: {e}")

        except Exception as e:
            logger.warn(f"ControlNet loading failed: {e}")

        # Load face models (non-critical)
        try:
            logger.info("Loading face swapping models...")

            FACE_SWAPPER = insightface.model_zoo.get_model(FACE_SWAP_MODEL)
            logger.info("Loaded face swapper model")

            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name=FACE_ANALYSER_MODEL,
                root=FACE_ANALYSER_ROOT,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("Loaded face analyser")

        except Exception as e:
            logger.warn(f"Face swapping models failed to load: {e}")

        MODELS_LOADED = True
        logger.info("Model initialization completed")
        return True

    except Exception as e:
        logger.error(f"Critical error during model loading: {e}")
        logger.error(traceback.format_exc())
        return False

def base64_to_image(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image.convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {e}")

def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')

def get_control_image(image: Image.Image, control_type: str) -> Optional[Image.Image]:
    """Generate control image using specified preprocessor."""
    if control_type not in PREPROCESSORS:
        logger.warn(f"Preprocessor {control_type} not available")
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

    try:
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        faces = FACE_ANALYSER.get(image_cv)
        return sorted(faces, key=lambda x: x.bbox[0])
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        return []

def swap_faces(source_image: Image.Image, target_image: Image.Image,
               source_indexes: str = "-1", target_indexes: str = "-1") -> Image.Image:
    """Swap faces between source and target images."""
    if FACE_SWAPPER is None or FACE_ANALYSER is None:
        logger.warn("Face swapping models not loaded")
        return target_image

    try:
        source_faces = get_faces(source_image)
        target_cv = cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR)
        target_faces = get_faces(target_image)

        if not source_faces or not target_faces:
            logger.warn("No faces detected in source or target image")
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
        for source_idx, target_idx in zip(source_indices, target_indices):
            if source_idx < len(source_faces) and target_idx < len(target_faces):
                source_face = source_faces[source_idx]
                target_face = target_faces[target_idx]
                result_image = FACE_SWAPPER.get(result_image, target_face, source_face, paste_back=True)

        result_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        return result_pil

    except Exception as e:
        logger.error(f"Face swapping failed: {e}")
        return target_image

def generate_image(prompt: str, negative_prompt: str = None,
                  control_images: dict = None, **kwargs) -> Image.Image:
    """Generate image using SDXL with optional ControlNet."""
    if DIFFUSION_PIPELINE is None:
        raise RuntimeError("Diffusion pipeline not loaded")

    try:
        from config import get_model_config
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

        # Use ControlNet if control images provided
        if control_images and CONTROLNET_PIPELINES:
            control_type = list(control_images.keys())[0]
            if control_type in CONTROLNET_PIPELINES:
                pipeline = CONTROLNET_PIPELINES[control_type]
                gen_kwargs["image"] = control_images[control_type]
                gen_kwargs["controlnet_conditioning_scale"] = kwargs.get('controlnet_conditioning_scale', 0.5)
                result = pipeline(**gen_kwargs)
            else:
                result = DIFFUSION_PIPELINE(**gen_kwargs)
        else:
            result = DIFFUSION_PIPELINE(**gen_kwargs)

        generated_image = result.images[0]

        # Apply refiner if available
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

def handler(job):
    """RunPod serverless handler function."""
    try:
        # Ensure models are loaded
        if not MODELS_LOADED:
            logger.info("Models not loaded, initializing...")
            if not safe_import_and_load_models():
                return {
                    "error": "Failed to load models during initialization",
                    "refresh_worker": True
                }

        # Validate input
        job_input = job.get("input", {})
        if not job_input:
            return {"error": "No input provided"}

        try:
            from schemas import INPUT_SCHEMA
            validated_input = validate(job_input, INPUT_SCHEMA)

            if 'errors' in validated_input:
                return {"error": f"Input validation failed: {validated_input['errors']}"}

            job_input = validated_input['validated_input']
        except Exception as e:
            logger.warn(f"Schema validation failed, using raw input: {e}")

        # Required fields check
        if 'image' not in job_input or 'prompt' not in job_input:
            return {"error": "Missing required fields: image and prompt"}

        # Parse input
        try:
            source_image = base64_to_image(job_input['image'])
            prompt = job_input['prompt']
        except Exception as e:
            return {"error": f"Failed to parse input: {e}"}

        # Generate control images if requested
        control_images = {}
        if job_input.get('use_controlnet', True) and PREPROCESSORS:
            for control_type in job_input.get('controlnet_types', ['depth', 'openpose']):
                control_image = get_control_image(source_image, control_type)
                if control_image:
                    control_images[control_type] = control_image

        # Generate new image
        logger.info("Generating image...")
        try:
            generated_image = generate_image(
                prompt=prompt,
                negative_prompt=job_input.get('negative_prompt'),
                control_images=control_images,
                **job_input
            )
        except Exception as e:
            return {
                "error": f"Image generation failed: {e}",
                "refresh_worker": True
            }

        # Swap faces if requested
        final_image = generated_image
        if job_input.get('swap_faces', True) and FACE_SWAPPER and FACE_ANALYSER:
            logger.info("Swapping faces...")
            try:
                final_image = swap_faces(
                    source_image=source_image,
                    target_image=generated_image,
                    source_indexes=job_input.get('source_indexes', "-1"),
                    target_indexes=job_input.get('target_indexes', "-1")
                )
            except Exception as e:
                logger.warn(f"Face swapping failed, returning generated image: {e}")

        # Convert to base64
        try:
            output_format = job_input.get('output_format', 'JPEG')
            result_base64 = image_to_base64(final_image, output_format)

            return {
                "image": result_base64,
                "format": output_format
            }
        except Exception as e:
            return {"error": f"Failed to encode output image: {e}"}

    except Exception as e:
        logger.error(f"Handler error: {e}")
        logger.error(traceback.format_exc())
        return {
            "error": f"Unexpected error: {e}",
            "refresh_worker": True
        }

# Load models at module level for RunPod serverless
logger.info("Initializing photobooth worker...")
safe_import_and_load_models()

# Start RunPod serverless
runpod.serverless.start({"handler": handler})
