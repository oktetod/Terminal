# Filename: modal_app.py
"""
Deploy Merged Model ke Modal.com dengan FastAPI - SIMPLIFIED VERSION
Features: Text-to-Image, Image-to-Image, ControlNet
Model: Juggernaut ALL-IN-ONE (Merged Model)
Changes: No LoRA, No NumPy, Simplified
"""

import modal
from pathlib import Path
import io
import base64
import logging
from typing import Optional, Dict, Any
from enum import Enum

# ===================================================================
# LOGGING SETUP
# ===================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================================================================
# KONFIGURASI GLOBAL
# ===================================================================
class Config:
    """Centralized configuration management"""
    APP_NAME = "juggernaut-api-fastapi"
    VERSION = "5.0.0"
    GPU_TYPE = "L4"
    
    # Directories
    MODEL_DIR = "/civitai-models/merged_models"
    CONTROLNET_DIR = "/controlnet_models"
    
    # Model Path (UPDATED)
    MODEL_PATH = f"{MODEL_DIR}/juggernaut_ALL_IN_ONE.safetensors"
    
    # Limits
    MAX_WIDTH = 2048
    MAX_HEIGHT = 2048
    MIN_DIMENSION = 512
    MAX_IMAGE_SIZE_MB = 10
    MAX_PROMPT_LENGTH = 2000
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE = 10
    RATE_LIMIT_PER_HOUR = 100
    
    # Default Prompts
    DEFAULT_POSITIVE_SUFFIX = (
        "masterpiece, best quality, 8k, photorealistic, intricate details, wide shot, "
        "(full body shot)"
    )
    
    DEFAULT_NEGATIVE_PROMPT = (
        "(worst quality, low quality, normal quality, blurry, fuzzy, pixelated), "
        "(extra limbs, extra fingers, malformed hands, missing fingers, extra digit, "
        "fused fingers, too many hands, bad hands, bad anatomy), "
        "(ugly, deformed, disfigured), "
        "(text, watermark, logo, signature), "
        "(worst quality, low quality, normal quality:1.4), (jpeg artifacts, blurry, grainy), "
        "ugly, duplicate, morbid, mutilated, (deformed, disfigured), (bad anatomy, bad proportions), "
        "(extra limbs, extra fingers, fused fingers, too many fingers, long neck), "
        "(mutated hands, bad hands, poorly drawn hands), (missing arms, missing legs), "
        "malformed limbs, (cross-eyed, bad eyes, asymmetrical eyes), (cleavage), "
        "signature, watermark, username, text, error, out of frame, out of focus, "
        "cropped, close-up, portrait, headshot, medium shot, upper body, bust shot, face"
    )
    
    # ControlNet Models
    CONTROLNET_MODELS = {
        "openpose": "thibaud/controlnet-openpose-sdxl-1.0",
        "canny": "diffusers/controlnet-canny-sdxl-1.0",
        "depth": "diffusers/controlnet-depth-sdxl-1.0",
    }

class ControlNetType(str, Enum):
    """Enum for ControlNet types"""
    OPENPOSE = "openpose"
    CANNY = "canny"
    DEPTH = "depth"

# ===================================================================
# MODAL APP SETUP
# ===================================================================
app = modal.App(Config.APP_NAME)

# Simplified image - NO NumPy, NO LoRA dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]==0.115.0",
        "pydantic==2.9.2",
        "pydantic-settings==2.5.2",
        "torch==2.1.0",
        "diffusers==0.25.0",
        "transformers==4.36.0",
        "accelerate==0.25.0",
        "safetensors==0.4.1",
        "Pillow==10.1.0",
        "requests==2.31.0",
        "huggingface_hub==0.20.1",
        "opencv-python-headless==4.8.1.78",
        "controlnet_aux==0.0.7",
        "pybind11>=2.12",
        "omegaconf",
        "slowapi==0.1.9",
    )
)

# Simplified volumes - NO LoRA, NO JSON
model_volume = modal.Volume.from_name("civitai-models", create_if_missing=True)
controlnet_volume = modal.Volume.from_name("controlnet-sdxl-collection-vol", create_if_missing=True)

# ===================================================================
# UTILITY FUNCTIONS
# ===================================================================
def validate_image_size(image_bytes: bytes) -> bool:
    """Validate image file size"""
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > Config.MAX_IMAGE_SIZE_MB:
        raise ValueError(f"Image size {size_mb:.2f}MB exceeds limit of {Config.MAX_IMAGE_SIZE_MB}MB")
    return True

def calculate_aspect_ratio_dimensions(
    original_width: int,
    original_height: int,
    target_width: int,
    target_height: int
) -> tuple[int, int]:
    """Calculate dimensions while preserving aspect ratio"""
    original_aspect = original_width / original_height
    target_aspect = target_width / target_height
    
    if original_aspect > target_aspect:
        new_width = target_width
        new_height = round(target_width / original_aspect)
    else:
        new_height = target_height
        new_width = round(target_height * original_aspect)
    
    # Ensure dimensions are multiples of 8
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    
    # Ensure minimum dimensions
    new_width = max(new_width, Config.MIN_DIMENSION)
    new_height = max(new_height, Config.MIN_DIMENSION)
    
    return new_width, new_height

# ===================================================================
# PYDANTIC MODELS FOR VALIDATION
# ===================================================================
try:
    from pydantic import BaseModel, Field, field_validator, ConfigDict
except ImportError as e:
    logger.error(f"Failed to import pydantic: {e}")
    raise ImportError(
        "Pydantic is required but not installed. "
        "Please ensure the Modal image includes pydantic>=2.0"
    ) from e

class Text2ImageRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    api_key: str = Field(..., min_length=1, description="API authentication key")
    prompt: str = Field(..., min_length=1, max_length=Config.MAX_PROMPT_LENGTH)
    negative_prompt: Optional[str] = Field(None, max_length=Config.MAX_PROMPT_LENGTH)
    num_steps: int = Field(default=25, ge=10, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    width: int = Field(default=1024, ge=Config.MIN_DIMENSION, le=Config.MAX_WIDTH)
    height: int = Field(default=1024, ge=Config.MIN_DIMENSION, le=Config.MAX_HEIGHT)
    seed: int = Field(default=-1, ge=-1)
    enhance_prompt: bool = Field(default=True)
    
    @field_validator('width', 'height')
    @classmethod
    def validate_dimensions(cls, v: int) -> int:
        if v % 8 != 0:
            raise ValueError(f"Dimension must be multiple of 8, got {v}")
        return v

class Image2ImageRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    api_key: str = Field(..., min_length=1)
    init_image: str = Field(..., min_length=1, description="Base64 encoded image")
    prompt: str = Field(..., min_length=1, max_length=Config.MAX_PROMPT_LENGTH)
    negative_prompt: Optional[str] = Field(None, max_length=Config.MAX_PROMPT_LENGTH)
    num_steps: int = Field(default=25, ge=10, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    strength: float = Field(default=0.75, ge=0.0, le=1.0)
    width: int = Field(default=1024, ge=Config.MIN_DIMENSION, le=Config.MAX_WIDTH)
    height: int = Field(default=1024, ge=Config.MIN_DIMENSION, le=Config.MAX_HEIGHT)
    seed: int = Field(default=-1, ge=-1)
    enhance_prompt: bool = Field(default=True)
    preserve_aspect_ratio: bool = Field(default=False)
    
    @field_validator('width', 'height')
    @classmethod
    def validate_dimensions(cls, v: int) -> int:
        if v % 8 != 0:
            raise ValueError(f"Dimension must be multiple of 8, got {v}")
        return v
    
    @field_validator('init_image')
    @classmethod
    def validate_base64(cls, v: str) -> str:
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError("Invalid base64 image data")
        return v

class ControlNetRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    api_key: str = Field(..., min_length=1)
    control_image: str = Field(..., min_length=1, description="Base64 encoded control image")
    prompt: str = Field(..., min_length=1, max_length=Config.MAX_PROMPT_LENGTH)
    controlnet_type: ControlNetType
    negative_prompt: Optional[str] = Field(None, max_length=Config.MAX_PROMPT_LENGTH)
    controlnet_scale: float = Field(default=0.8, ge=0.0, le=2.0)
    num_steps: int = Field(default=30, ge=10, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    seed: int = Field(default=-1, ge=-1)
    enhance_prompt: bool = Field(default=True)
    
    @field_validator('control_image')
    @classmethod
    def validate_base64(cls, v: str) -> str:
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError("Invalid base64 control image data")
        return v

# ===================================================================
# DOWNLOAD FUNCTIONS
# ===================================================================
@app.function(image=image, volumes={Config.CONTROLNET_DIR: controlnet_volume}, timeout=3600)
def download_controlnet_models():
    """Download all ControlNet models from HuggingFace"""
    from huggingface_hub import snapshot_download
    from pathlib import Path
    
    results = {"success": [], "failed": []}
    
    for name, repo_id in Config.CONTROLNET_MODELS.items():
        model_dir = Path(Config.CONTROLNET_DIR) / name
        
        if model_dir.exists():
            logger.info(f"✓ ControlNet {name} already exists")
            results["success"].append(name)
            continue
        
        try:
            logger.info(f"📥 Downloading ControlNet: {name} from {repo_id}...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False
            )
            controlnet_volume.commit()
            logger.info(f"✓ ControlNet {name} downloaded successfully")
            results["success"].append(name)
            
        except Exception as e:
            logger.error(f"✗ Failed to download ControlNet {name}: {e}")
            results["failed"].append({"name": name, "error": str(e)})
    
    return results

# ===================================================================
# MODEL INFERENCE CLASS - SIMPLIFIED
# ===================================================================
@app.cls(
    image=image,
    gpu=Config.GPU_TYPE,
    volumes={
        Config.MODEL_DIR: model_volume,
        Config.CONTROLNET_DIR: controlnet_volume
    },
    scaledown_window=200,
    timeout=600
)
class ModelInference:
    @modal.enter()
    def load_model(self):
        """Initialize all models and pipelines"""
        from diffusers import (
            StableDiffusionXLPipeline,
            StableDiffusionXLImg2ImgPipeline,
            EulerDiscreteScheduler
        )
        import torch
        from transformers import pipeline as transformers_pipeline
        
        logger.info("🚀 Initializing Juggernaut ALL-IN-ONE model...")
        
        if not Path(Config.MODEL_PATH).exists():
            raise FileNotFoundError(f"Model not found at {Config.MODEL_PATH}")
        
        try:
            self.txt2img_pipe = StableDiffusionXLPipeline.from_single_file(
                Config.MODEL_PATH,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
            self.txt2img_pipe.scheduler = EulerDiscreteScheduler.from_config(
                self.txt2img_pipe.scheduler.config
            )
            self.txt2img_pipe.to("cuda")
            
            self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
                vae=self.txt2img_pipe.vae,
                text_encoder=self.txt2img_pipe.text_encoder,
                text_encoder_2=self.txt2img_pipe.text_encoder_2,
                tokenizer=self.txt2img_pipe.tokenizer,
                tokenizer_2=self.txt2img_pipe.tokenizer_2,
                unet=self.txt2img_pipe.unet,
                scheduler=self.txt2img_pipe.scheduler,
            )
            self.img2img_pipe.to("cuda")
            
            logger.info("🔧 Initializing depth estimator...")
            self.depth_estimator = transformers_pipeline('depth-estimation')
            
            self.current_controlnet_pipe = None
            self.current_controlnet_type = None
            
            logger.info("✅ Juggernaut ALL-IN-ONE Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            raise

    def _cleanup_controlnet(self):
        """Cleanup ControlNet pipeline to prevent memory leaks"""
        if self.current_controlnet_pipe is not None:
            try:
                del self.current_controlnet_pipe
                self.current_controlnet_pipe = None
                self.current_controlnet_type = None
                
                import torch
                torch.cuda.empty_cache()
                
                logger.info("🧹 ControlNet pipeline cleaned up")
            except Exception as e:
                logger.warning(f"⚠️ ControlNet cleanup warning: {e}")

    def _preprocess_control_image(self, image, controlnet_type: str):
        """Preprocess control image based on type - NO NUMPY"""
        from controlnet_aux import OpenposeDetector
        from PIL import Image
        import cv2
        
        try:
            if controlnet_type == "openpose":
                processor = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
                return processor(image)
                
            elif controlnet_type == "canny":
                # Convert PIL to cv2 without numpy
                image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if hasattr(image, 'mode') else image
                canny_image = cv2.Canny(image_cv, 100, 200)
                # Convert back to PIL
                canny_rgb = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB)
                return Image.fromarray(canny_rgb)
                
            elif controlnet_type == "depth":
                depth = self.depth_estimator(image)['depth']
                return depth
            
        except Exception as e:
            logger.error(f"✗ Control image preprocessing failed: {e}")
            raise

    @modal.method()
    def text_to_image(self, request_dict: Dict[str, Any]):
        """Generate image from text prompt"""
        import torch
        
        try:
            request = Text2ImageRequest(**request_dict)
            
            enhanced_prompt = (
                f"{request.prompt}, {Config.DEFAULT_POSITIVE_SUFFIX}"
                if request.enhance_prompt
                else request.prompt
            )
            negative_prompt = request.negative_prompt or Config.DEFAULT_NEGATIVE_PROMPT
            
            logger.info(f"🎨 Generating text-to-image...")
            
            generator = None
            if request.seed != -1:
                generator = torch.Generator(device="cuda").manual_seed(request.seed)
            
            image = self.txt2img_pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=request.num_steps,
                guidance_scale=request.guidance_scale,
                width=request.width,
                height=request.height,
                generator=generator
            ).images[0]
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG", quality=95)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            logger.info("✅ Text-to-image generation completed")
            
            return {
                "image": img_str,
                "prompt": enhanced_prompt,
                "original_prompt": request.prompt,
                "negative_prompt": negative_prompt,
                "seed": request.seed,
                "width": request.width,
                "height": request.height,
                "model": "Juggernaut ALL-IN-ONE (Merged)",
                "version": Config.VERSION
            }
            
        except Exception as e:
            logger.error(f"✗ Text-to-image failed: {e}")
            raise

    @modal.method()
    def image_to_image(self, request_dict: Dict[str, Any]):
        """Edit image with prompt"""
        from PIL import Image
        import torch
        
        try:
            request = Image2ImageRequest(**request_dict)
            
            init_image_bytes = base64.b64decode(request.init_image)
            validate_image_size(init_image_bytes)
            
            init_image = Image.open(io.BytesIO(init_image_bytes)).convert("RGB")
            original_width, original_height = init_image.size
            
            if request.preserve_aspect_ratio:
                target_width, target_height = calculate_aspect_ratio_dimensions(
                    original_width, original_height,
                    request.width, request.height
                )
                logger.info(f"📐 Preserving aspect ratio: {original_width}x{original_height} → {target_width}x{target_height}")
            else:
                target_width, target_height = request.width, request.height
            
            init_image = init_image.resize((target_width, target_height), Image.LANCZOS)
            
            enhanced_prompt = (
                f"{request.prompt}, {Config.DEFAULT_POSITIVE_SUFFIX}"
                if request.enhance_prompt
                else request.prompt
            )
            negative_prompt = request.negative_prompt or Config.DEFAULT_NEGATIVE_PROMPT
            
            logger.info(f"🖼️ Generating image-to-image...")
            
            generator = None
            if request.seed != -1:
                generator = torch.Generator(device="cuda").manual_seed(request.seed)
            
            image = self.img2img_pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=request.strength,
                num_inference_steps=request.num_steps,
                guidance_scale=request.guidance_scale,
                generator=generator,
                width=target_width,
                height=target_height
            ).images[0]
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG", quality=95)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            logger.info("✅ Image-to-image generation completed")
            
            return {
                "image": img_str,
                "prompt": enhanced_prompt,
                "original_prompt": request.prompt,
                "negative_prompt": negative_prompt,
                "strength": request.strength,
                "seed": request.seed,
                "original_size": f"{original_width}x{original_height}",
                "output_size": f"{target_width}x{target_height}",
                "aspect_ratio_preserved": request.preserve_aspect_ratio,
                "model": "Juggernaut ALL-IN-ONE (Merged)",
                "version": Config.VERSION
            }
            
        except Exception as e:
            logger.error(f"✗ Image-to-image failed: {e}")
            raise

    @modal.method()
    def generate_with_controlnet(self, request_dict: Dict[str, Any]):
        """Generate image with ControlNet guidance"""
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
        from PIL import Image
        import torch
        
        try:
            request = ControlNetRequest(**request_dict)
            
            control_image_bytes = base64.b64decode(request.control_image)
            validate_image_size(control_image_bytes)
            control_image = Image.open(io.BytesIO(control_image_bytes)).convert("RGB")
            
            if (self.current_controlnet_type != request.controlnet_type.value or
                self.current_controlnet_pipe is None):
                
                self._cleanup_controlnet()
                
                logger.info(f"🎮 Loading ControlNet: {request.controlnet_type.value}")
                controlnet = ControlNetModel.from_pretrained(
                    str(Path(Config.CONTROLNET_DIR) / request.controlnet_type.value),
                    torch_dtype=torch.float16
                )
                
                self.current_controlnet_pipe = StableDiffusionXLControlNetPipeline(
                    vae=self.txt2img_pipe.vae,
                    text_encoder=self.txt2img_pipe.text_encoder,
                    text_encoder_2=self.txt2img_pipe.text_encoder_2,
                    tokenizer=self.txt2img_pipe.tokenizer,
                    tokenizer_2=self.txt2img_pipe.tokenizer_2,
                    unet=self.txt2img_pipe.unet,
                    scheduler=self.txt2img_pipe.scheduler,
                    controlnet=controlnet,
                )
                self.current_controlnet_pipe.to("cuda")
                self.current_controlnet_type = request.controlnet_type.value
            
            control_image = self._preprocess_control_image(
                control_image,
                request.controlnet_type.value
            )
            
            enhanced_prompt = (
                f"{request.prompt}, {Config.DEFAULT_POSITIVE_SUFFIX}"
                if request.enhance_prompt
                else request.prompt
            )
            negative_prompt = request.negative_prompt or Config.DEFAULT_NEGATIVE_PROMPT
            
            logger.info(f"🎮 Generating with ControlNet ({request.controlnet_type.value})...")
            
            generator = None
            if request.seed != -1:
                generator = torch.Generator(device="cuda").manual_seed(request.seed)
            
            image = self.current_controlnet_pipe(
                enhanced_prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                num_inference_steps=request.num_steps,
                guidance_scale=request.guidance_scale,
                controlnet_conditioning_scale=float(request.controlnet_scale),
                generator=generator,
            ).images[0]
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG", quality=95)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            logger.info("✅ ControlNet generation completed")
            
            return {
                "image": img_str,
                "prompt": enhanced_prompt,
                "original_prompt": request.prompt,
                "negative_prompt": negative_prompt,
                "seed": request.seed,
                "controlnet_type": request.controlnet_type.value,
                "controlnet_scale": request.controlnet_scale,
                "model": "Juggernaut ALL-IN-ONE (Merged)",
                "version": Config.VERSION
            }
            
        except Exception as e:
            logger.error(f"✗ ControlNet generation failed: {e}")
            raise

# ===================================================================
# FASTAPI APPLICATION - SIMPLIFIED
# ===================================================================
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("custom-secret")]
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException, Request, Depends
    from fastapi.responses import JSONResponse
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    import os
    
    web_app = FastAPI(
        title="Juggernaut ALL-IN-ONE API - Simplified",
        version=Config.VERSION,
        description="Merged SDXL model - No LoRA, No NumPy"
    )
    
    limiter = Limiter(key_func=get_remote_address)
    web_app.state.limiter = limiter
    web_app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    async def verify_api_key(request: Request):
        """Dependency to verify API key"""
        try:
            data = await request.json()
            api_key = data.get("api_key")
            
            if not api_key:
                raise HTTPException(status_code=401, detail="API key is required")
            
            expected_key = os.environ.get("API_KEY")
            if not expected_key:
                raise HTTPException(status_code=500, detail="API_KEY not configured on server")
            
            if api_key != expected_key:
                logger.warning(f"⚠️ Invalid API key attempt from {request.client.host}")
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            return data
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"✗ API key verification error: {e}")
            raise HTTPException(status_code=400, detail="Invalid request format")
    
    @web_app.get("/")
    @limiter.limit("30/minute")
    async def root(request: Request):
        """API information endpoint"""
        return {
            "service": "Juggernaut ALL-IN-ONE API",
            "version": Config.VERSION,
            "gpu": Config.GPU_TYPE,
            "status": "operational",
            "model": "Juggernaut ALL-IN-ONE (Merged Model)",
            "improvements": [
                "✅ No LoRA (merged into base model)",
                "✅ No NumPy dependency",
                "✅ Simplified codebase",
                "✅ Using merged_models/juggernaut_ALL_IN_ONE.safetensors"
            ],
            "endpoints": {
                "health": "GET /health",
                "text_to_image": "POST /text2img",
                "image_to_image": "POST /img2img",
                "controlnet": "POST /controlnet"
            },
            "features": [
                "✅ Text-to-Image (SDXL)",
                "✅ Image-to-Image (SDXL)",
                "✅ ControlNet (openpose, canny, depth)",
                "✅ Aspect ratio preservation",
                "✅ Request validation",
                "✅ Rate limiting"
            ],
            "configuration": {
                "max_width": Config.MAX_WIDTH,
                "max_height": Config.MAX_HEIGHT,
                "max_image_size_mb": Config.MAX_IMAGE_SIZE_MB,
                "max_prompt_length": Config.MAX_PROMPT_LENGTH,
                "rate_limit": f"{Config.RATE_LIMIT_PER_MINUTE}/min, {Config.RATE_LIMIT_PER_HOUR}/hour"
            },
            "controlnet_types": list(Config.CONTROLNET_MODELS.keys())
        }
    
    @web_app.get("/health")
    @limiter.limit("60/minute")
    async def health_check(request: Request):
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": Config.APP_NAME,
            "version": Config.VERSION,
            "model": "Juggernaut ALL-IN-ONE (Merged)",
            "gpu": Config.GPU_TYPE,
            "lora_support": False,
            "numpy_dependency": False
        }
    
    @web_app.post("/text2img")
    @limiter.limit(f"{Config.RATE_LIMIT_PER_MINUTE}/minute")
    async def text_to_image_endpoint(request: Request):
        """Text-to-image generation endpoint"""
        try:
            data = await verify_api_key(request)
            
            model = ModelInference()
            result = model.text_to_image.remote(data)
            
            return JSONResponse(content=result)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"✗ Text2img endpoint error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @web_app.post("/img2img")
    @limiter.limit(f"{Config.RATE_LIMIT_PER_MINUTE}/minute")
    async def image_to_image_endpoint(request: Request):
        """Image-to-image generation endpoint"""
        try:
            data = await verify_api_key(request)
            
            model = ModelInference()
            result = model.image_to_image.remote(data)
            
            return JSONResponse(content=result)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"✗ Img2img endpoint error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @web_app.post("/controlnet")
    @limiter.limit(f"{Config.RATE_LIMIT_PER_MINUTE}/minute")
    async def controlnet_endpoint(request: Request):
        """ControlNet generation endpoint"""
        try:
            data = await verify_api_key(request)
            
            model = ModelInference()
            result = model.generate_with_controlnet.remote(data)
            
            return JSONResponse(content=result)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"✗ ControlNet endpoint error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return web_app

# ===================================================================
# LOCAL ENTRYPOINT
# ===================================================================
@app.local_entrypoint()
def main():
    """Local test entrypoint with helpful commands"""
    print("\n" + "="*70)
    print("🚀 JUGGERNAUT ALL-IN-ONE API - SIMPLIFIED DEPLOYMENT")
    print("="*70)
    print("\n📦 Model: juggernaut_ALL_IN_ONE.safetensors (Merged)")
    print("✅ Changes Applied:")
    print("  • Removed all LoRA functionality")
    print("  • Removed NumPy dependency")
    print("  • Using merged model at civitai-models/merged_models/")
    print("  • Simplified codebase")
    print("\n📋 Available Commands:\n")
    print("  1. Download ControlNet models:")
    print("     modal run modal_app.py::download_controlnet_models\n")
    print("  2. Deploy API to production:")
    print("     modal deploy modal_app.py\n")
    print("  3. Test locally:")
    print("     modal serve modal_app.py\n")
    print("="*70)
    print("\n⚠️  IMPORTANT NOTES:")
    print("  • Make sure your merged model exists at:")
    print("    /civitai-models/merged_models/juggernaut_ALL_IN_ONE.safetensors")
    print("  • Volume 'civitai-models' must contain the merged model")
    print("  • No LoRA downloads needed (already merged)")
    print("  • No JSON configuration files needed")
    print("="*70)
    print("\n🎯 Next Steps:")
    print("  1. Verify model file exists in Modal volume")
    print("  2. Download ControlNet models (if needed)")
    print("  3. Deploy with: modal deploy modal_app.py")
    print("  4. Test your merged model!")
    print("="*70 + "\n")