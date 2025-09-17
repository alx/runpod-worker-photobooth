import os
from typing import Dict, Any

# Model configuration
DEFAULT_MODEL_ID = "stablediffusionapi/turbovision_xl"
MODEL_ID = os.getenv("MODEL_ID", DEFAULT_MODEL_ID)

# Model-specific configurations
MODEL_CONFIGS = {
    "stablediffusionapi/turbovision_xl": {
        "scheduler": "DPMSolverMultistepScheduler",
        "num_inference_steps": 5,
        "guidance_scale": 2.0,
        "use_refiner": False,
        "supports_negative_prompt": False,
    },
    "stabilityai/stable-diffusion-xl-base-1.0": {
        "scheduler": "EulerDiscreteScheduler",
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
        "use_refiner": True,
        "supports_negative_prompt": True,
    },
    "stabilityai/sdxl-turbo": {
        "scheduler": "EulerAncestralDiscreteScheduler",
        "num_inference_steps": 1,
        "guidance_scale": 0.0,
        "use_refiner": False,
        "supports_negative_prompt": False,
    }
}

# ControlNet models
CONTROLNET_MODELS = {
    "depth": "diffusers/controlnet-depth-sdxl-1.0",
    "openpose": "thibaud/controlnet-openpose-sdxl-1.0",
    "canny": "diffusers/controlnet-canny-sdxl-1.0"
}

# InsightFace configuration
FACE_SWAP_MODEL = "checkpoints/inswapper_128.onnx"
FACE_ANALYSER_MODEL = "buffalo_l"
FACE_ANALYSER_ROOT = "./checkpoints"

# Path configurations
TMP_PATH = "/tmp/photobooth"
CACHE_DIR = os.getenv("HF_HOME", "/models")

# Default generation parameters
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_SEED = None
DEFAULT_NUM_IMAGES = 1

# Face swapping defaults
DEFAULT_FACE_RESTORE = True
DEFAULT_FACE_UPSAMPLE = True
DEFAULT_BACKGROUND_ENHANCE = True
DEFAULT_UPSCALE = 1
DEFAULT_CODEFORMER_FIDELITY = 0.5

# Output format
DEFAULT_OUTPUT_FORMAT = "JPEG"

def get_model_config(model_id: str = None) -> Dict[str, Any]:
    """Get configuration for the specified model or current MODEL_ID."""
    model = model_id or MODEL_ID

    # Return custom config if exists, otherwise use SDXL base config
    return MODEL_CONFIGS.get(model, MODEL_CONFIGS["stabilityai/stable-diffusion-xl-base-1.0"])

def get_scheduler_class(scheduler_name: str):
    """Get the scheduler class from string name."""
    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        DPMSolverSinglestepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        LMSDiscreteScheduler,
        PNDMScheduler,
    )

    scheduler_map = {
        "DDIM": DDIMScheduler,
        "DDIMScheduler": DDIMScheduler,
        "DPMSolverMultistep": DPMSolverMultistepScheduler,
        "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
        "DPMSolverSinglestep": DPMSolverSinglestepScheduler,
        "DPMSolverSinglestepScheduler": DPMSolverSinglestepScheduler,
        "EulerAncestral": EulerAncestralDiscreteScheduler,
        "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
        "Euler": EulerDiscreteScheduler,
        "EulerDiscreteScheduler": EulerDiscreteScheduler,
        "LMS": LMSDiscreteScheduler,
        "LMSDiscreteScheduler": LMSDiscreteScheduler,
        "PNDM": PNDMScheduler,
        "PNDMScheduler": PNDMScheduler,
    }

    return scheduler_map.get(scheduler_name, EulerDiscreteScheduler)