from config import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DEFAULT_SEED,
    DEFAULT_NUM_IMAGES,
    DEFAULT_FACE_RESTORE,
    DEFAULT_FACE_UPSAMPLE,
    DEFAULT_BACKGROUND_ENHANCE,
    DEFAULT_UPSCALE,
    DEFAULT_CODEFORMER_FIDELITY,
    DEFAULT_OUTPUT_FORMAT,
    get_model_config,
    MODEL_ID
)

# Get model-specific defaults
model_config = get_model_config(MODEL_ID)

INPUT_SCHEMA = {
    # Required inputs
    'image': {
        'type': str,
        'required': True,
        'description': 'Base64 encoded source image'
    },
    'prompt': {
        'type': str,
        'required': True,
        'description': 'Text prompt for image generation'
    },

    # Model selection (optional)
    'model_id': {
        'type': str,
        'required': False,
        'default': MODEL_ID,
        'description': 'HuggingFace model ID to use for generation'
    },

    # Generation parameters
    'negative_prompt': {
        'type': str,
        'required': False,
        'default': None,
        'description': 'Negative prompt (not supported by TurboVisionXL/SDXL-Turbo)'
    },
    'height': {
        'type': int,
        'required': False,
        'default': DEFAULT_HEIGHT,
        'constraints': lambda x: x in [512, 768, 1024, 1536, 2048]
    },
    'width': {
        'type': int,
        'required': False,
        'default': DEFAULT_WIDTH,
        'constraints': lambda x: x in [512, 768, 1024, 1536, 2048]
    },
    'seed': {
        'type': int,
        'required': False,
        'default': DEFAULT_SEED
    },
    'scheduler': {
        'type': str,
        'required': False,
        'default': model_config['scheduler'],
        'constraints': lambda x: x in [
            'DDIM',
            'DPMSolverMultistep',
            'DPMSolverSinglestep',
            'EulerAncestral',
            'Euler',
            'LMS',
            'PNDM'
        ]
    },
    'num_inference_steps': {
        'type': int,
        'required': False,
        'default': model_config['num_inference_steps'],
        'constraints': lambda x: 1 <= x <= 100
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': model_config['guidance_scale'],
        'constraints': lambda x: 0.0 <= x <= 20.0
    },
    'num_images': {
        'type': int,
        'required': False,
        'default': DEFAULT_NUM_IMAGES,
        'constraints': lambda x: 1 <= x <= 4
    },

    # ControlNet parameters
    'use_controlnet': {
        'type': bool,
        'required': False,
        'default': True,
        'description': 'Use ControlNet for guided generation'
    },
    'controlnet_types': {
        'type': list,
        'required': False,
        'default': ['depth', 'openpose'],
        'description': 'Which ControlNet models to use',
        'constraints': lambda x: all(t in ['depth', 'openpose', 'canny'] for t in x)
    },
    'controlnet_conditioning_scale': {
        'type': float,
        'required': False,
        'default': 0.5,
        'description': 'ControlNet conditioning strength',
        'constraints': lambda x: 0.0 <= x <= 1.0
    },

    # Face swapping parameters
    'swap_faces': {
        'type': bool,
        'required': False,
        'default': True,
        'description': 'Whether to swap faces from source to generated image'
    },
    'source_indexes': {
        'type': str,
        'required': False,
        'default': "-1",
        'description': 'Which faces to use from source image (-1 for all)'
    },
    'target_indexes': {
        'type': str,
        'required': False,
        'default': "-1",
        'description': 'Which faces to replace in generated image (-1 for all)'
    },

    # Face restoration parameters
    'face_restore': {
        'type': bool,
        'required': False,
        'default': DEFAULT_FACE_RESTORE
    },
    'face_upsample': {
        'type': bool,
        'required': False,
        'default': DEFAULT_FACE_UPSAMPLE
    },
    'background_enhance': {
        'type': bool,
        'required': False,
        'default': DEFAULT_BACKGROUND_ENHANCE
    },
    'upscale': {
        'type': int,
        'required': False,
        'default': DEFAULT_UPSCALE,
        'constraints': lambda x: x in [1, 2, 4]
    },
    'codeformer_fidelity': {
        'type': float,
        'required': False,
        'default': DEFAULT_CODEFORMER_FIDELITY,
        'constraints': lambda x: 0.0 <= x <= 1.0
    },

    # Output parameters
    'output_format': {
        'type': str,
        'required': False,
        'default': DEFAULT_OUTPUT_FORMAT,
        'constraints': lambda x: x in ['JPEG', 'PNG']
    }
}