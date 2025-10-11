# Filename: modal_app.py
"""
Deploy Juggernaut ALL-IN-ONE Model ke Modal.com dengan FastAPI - FIXED VERSION
Features: Text-to-Image, Image-to-Image, ControlNet, Multi-LoRA
Model: Juggernaut ALL-IN-ONE (Merged Model)
Version: 5.2.1 - All Bugs Fixed
"""

# ===================================================================
# TOP-LEVEL IMPORTS - ONLY STANDARD LIBRARY & MODAL
# ===================================================================
import modal
from pathlib import Path
import io
import base64
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
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
    APP_NAME = "civitai-api-fastapi"
    VERSION = "5.2.1"  # Fixed version
    GPU_TYPE = "L4"
    
    # Directories
    MODEL_DIR = "/models"
    LORA_DIR = "/loras"
    CONTROLNET_DIR = "/controlnet_models"
    JSON_DATA_DIR = "/json_data"
    
    # Model Path - Juggernaut ALL-IN-ONE
    MODEL_PATH = f"{MODEL_DIR}/model.safetensors"
    
    # Model URL - Juggernaut from CivitAI
    MODEL_URL = "https://civitai.com/api/download/models/348913?type=Model&format=SafeTensor&size=full&fp=fp16"
    
    # GitHub JSON URLs for LoRAs
    JSON_URLS = {
        "01.json": "https://raw.githubusercontent.com/oktetod/Terminal/main/01.json",
        "02.json": "https://raw.githubusercontent.com/oktetod/Terminal/main/02.json",
        "03.json": "https://raw.githubusercontent.com/oktetod/Terminal/main/03.json",
        "04.json": "https://raw.githubusercontent.com/oktetod/Terminal/main/04.json",
        "05.json": "https://raw.githubusercontent.com/oktetod/Terminal/main/05.json",
        "06.json": "https://raw.githubusercontent.com/oktetod/Terminal/main/06.json",
        "07.json": "https://raw.githubusercontent.com/oktetod/Terminal/main/07.json",
    }
    
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

# Modal Image - Build all dependencies (FIXED: removed pybind11 from pip)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        # Core dependencies
        "pydantic==2.9.2",
        "pydantic-settings==2.5.2",
        "fastapi[standard]==0.115.0",
        "slowapi==0.1.9",
        # ML/AI dependencies
        "numpy<2.0.0",
        "torch==2.1.0",
        "diffusers==0.25.0",
        "transformers==4.36.0",
        "accelerate==0.25.0",
        "safetensors==0.4.1",
        # Image processing
        "Pillow==10.1.0",
        "opencv-python-headless==4.8.1.78",
        "controlnet_aux==0.0.7",
        # Utilities
        "requests==2.31.0",
        "huggingface_hub==0.20.1",
        "omegaconf",
    )
    .run_commands(
        "python -c 'import pydantic; print(f\"[OK] Pydantic {pydantic.__version__}\")'",
        "python -c 'import fastapi; print(f\"[OK] FastAPI {fastapi.__version__}\")'",
        "python -c 'import torch; print(f\"[OK] PyTorch {torch.__version__}\")'",
        "echo '[OK] Image build completed!'",
    )
)

# Volumes
model_volume = modal.Volume.from_name("civitai-models", create_if_missing=True)
lora_volume = modal.Volume.from_name("civitai-loras-collection-vol", create_if_missing=True)
controlnet_volume = modal.Volume.from_name("controlnet-sdxl-collection-vol", create_if_missing=True)
json_volume = modal.Volume.from_name("json-config-vol", create_if_missing=True)

# ===================================================================
# UTILITY FUNCTIONS - IMPORTED INSIDE FUNCTIONS
# ===================================================================
def get_pydantic_models():
    """Return Pydantic models - imported inside function"""
    from pydantic import BaseModel, Field, field_validator, ConfigDict
    
    class Text2ImageRequest(BaseModel):
        model_config = ConfigDict(extra='forbid')
        
        api_key: str = Field(..., min_length=1, description="API authentication key")
        prompt: str = Field(..., min_length=1, max_length=Config.MAX_PROMPT_LENGTH)
        negative_prompt: Optional[str] = Field(None, max_length=Config.MAX_PROMPT_LENGTH)
        lora_name: Optional[str] = None
        lora_scale: float = Field(default=0.8, ge=0.0, le=2.0)
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
        lora_name: Optional[str] = None
        lora_scale: float = Field(default=0.8, ge=0.0, le=2.0)
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
        lora_name: Optional[str] = None
        lora_scale: float = Field(default=0.8, ge=0.0, le=2.0)
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
    
    return Text2ImageRequest, Image2ImageRequest, ControlNetRequest

def load_lora_models_from_json(json_dir: str = Config.JSON_DATA_DIR) -> Dict[str, Any]:
    """Load ALL LoRA models from ALL JSON files"""
    lora_models = {}
    json_files = list(Config.JSON_URLS.keys())
    
    for json_file in json_files:
        json_path = Path(json_dir) / json_file
        try:
            if not json_path.exists():
                logger.warning(f"[WARN] {json_path} not found, skipping...")
                continue
                
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Filter out comment keys
                filtered_data = {k: v for k, v in data.items() if not k.startswith("//")}
                lora_models.update(filtered_data)
                logger.info(f"[OK] Loaded {len(filtered_data)} LoRAs from {json_file}")
                
        except json.JSONDecodeError as e:
            logger.error(f"[ERROR] Error parsing {json_file}: {e}")
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error loading {json_file}: {e}")
    
    logger.info(f"[INFO] Total LoRAs loaded: {len(lora_models)}")
    return lora_models

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
) -> Tuple[int, int]:
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
# DOWNLOAD FUNCTIONS
# ===================================================================
@app.function(image=image, volumes={Config.JSON_DATA_DIR: json_volume}, timeout=600)
def download_json_files():
    """Download ALL JSON configuration files from GitHub"""
    import requests
    from pathlib import Path
    import time
    
    json_path = Path(Config.JSON_DATA_DIR)
    json_path.mkdir(parents=True, exist_ok=True)
    
    max_retries = 3
    results = {"success": [], "failed": []}
    
    for filename, url in Config.JSON_URLS.items():
        dest = json_path / filename
        
        for attempt in range(max_retries):
            try:
                logger.info(f"[DOWNLOAD] {filename} (attempt {attempt + 1}/{max_retries})...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                with open(dest, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"[OK] Successfully downloaded {filename}")
                results["success"].append(filename)
                break
                
            except Exception as e:
                logger.error(f"[ERROR] Attempt {attempt + 1} failed for {filename}: {e}")
                if attempt == max_retries - 1:
                    results["failed"].append({"file": filename, "error": str(e)})
                else:
                    time.sleep(2 ** attempt)
    
    json_volume.commit()
    logger.info(f"[COMPLETE] Download: {len(results['success'])} success, {len(results['failed'])} failed")
    return results

@app.function(image=image, volumes={Config.MODEL_DIR: model_volume}, timeout=3600)
def download_model():
    """Download Juggernaut ALL-IN-ONE model from CivitAI with progress tracking"""
    import requests
    from pathlib import Path
    
    model_path = Path(Config.MODEL_PATH)
    
    if model_path.exists():
        logger.info(f"[OK] Model already exists at {model_path}")
        file_size = model_path.stat().st_size / (1024 * 1024)
        return {
            "status": "exists", 
            "path": str(model_path),
            "size_mb": f"{file_size:.2f}"
        }
    
    try:
        logger.info("[DOWNLOAD] Juggernaut ALL-IN-ONE model from CivitAI...")
        response = requests.get(Config.MODEL_URL, stream=True, timeout=120)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (50 * 1024 * 1024) == 0:  # Log every 50MB
                        progress = (downloaded / total_size) * 100
                        logger.info(f"Progress: {progress:.1f}% ({downloaded/(1024*1024):.0f}MB)")
        
        model_volume.commit()
        size_mb = downloaded / (1024 * 1024)
        logger.info(f"[SUCCESS] Model downloaded: {size_mb:.2f}MB")
        return {
            "status": "success", 
            "path": str(model_path), 
            "size_mb": f"{size_mb:.2f}"
        }
        
    except Exception as e:
        logger.error(f"[ERROR] Model download failed: {e}")
        if model_path.exists():
            model_path.unlink()  # Clean up partial download
        raise

@app.function(
    image=image,
    volumes={Config.LORA_DIR: lora_volume, Config.JSON_DATA_DIR: json_volume},
    timeout=7200
)
def download_loras():
    """Download ALL LoRA models from ALL JSON configuration files"""
    import requests
    from pathlib import Path
    import time
    
    logger.info("="*70)
    logger.info("[INFO] LOADING LORA CONFIGURATIONS FROM ALL JSON FILES")
    logger.info("="*70)
    
    lora_models = load_lora_models_from_json(Config.JSON_DATA_DIR)
    
    if not lora_models:
        logger.error("[ERROR] No LoRA models found in JSON files!")
        return {"status": "error", "message": "No LoRA models found"}
    
    logger.info(f"\n[INFO] Total LoRAs to process: {len(lora_models)}")
    logger.info("="*70)
    
    failed_downloads = []
    successful_downloads = []
    skipped_existing = []
    
    total_loras = len(lora_models)
    current_index = 0
    max_retries = 2
    
    for name, data in lora_models.items():
        current_index += 1
        
        if not isinstance(data, dict) or 'url' not in data or 'filename' not in data:
            logger.warning(f"[{current_index}/{total_loras}] [WARN] Invalid data format for: {name}")
            continue
        
        lora_path = Path(Config.LORA_DIR) / data["filename"]
        
        if lora_path.exists():
            logger.info(f"[{current_index}/{total_loras}] [OK] LoRA already exists: {name}")
            skipped_existing.append(name)
            continue
        
        for attempt in range(max_retries):
            try:
                logger.info(f"[{current_index}/{total_loras}] [DOWNLOAD] {name} (attempt {attempt+1})...")
                
                response = requests.get(data["url"], stream=True, timeout=180)
                response.raise_for_status()
                
                lora_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(lora_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                lora_volume.commit()
                logger.info(f"[{current_index}/{total_loras}] [SUCCESS] {name}")
                successful_downloads.append(name)
                break
                
            except Exception as e:
                logger.error(f"[{current_index}/{total_loras}] [ERROR] Attempt {attempt+1} failed: {e}")
                
                if attempt == max_retries - 1:
                    failed_downloads.append({
                        "index": current_index,
                        "name": name,
                        "filename": data["filename"],
                        "error": str(e)
                    })
                    if lora_path.exists():
                        lora_path.unlink()  # Clean up partial download
                else:
                    time.sleep(2 ** attempt)
    
    logger.info("\n" + "="*70)
    logger.info("[REPORT] DOWNLOAD SUMMARY")
    logger.info("="*70)
    logger.info(f"[SUCCESS] Downloaded: {len(successful_downloads)}")
    logger.info(f"[SKIPPED] Already existed: {len(skipped_existing)}")
    logger.info(f"[FAILED] Failed: {len(failed_downloads)}")
    logger.info(f"[TOTAL] Total LoRAs: {total_loras}")
    
    if failed_downloads:
        logger.warning("\n[WARNING] FAILED DOWNLOADS:")
        for fail in failed_downloads:
            logger.warning(f"  #{fail['index']} - {fail['name']}: {fail['error']}")
    
    return {
        "status": "complete",
        "total": total_loras,
        "success": len(successful_downloads),
        "skipped": len(skipped_existing),
        "failed": len(failed_downloads),
        "failed_list": failed_downloads
    }

@app.function(image=image, volumes={Config.CONTROLNET_DIR: controlnet_volume}, timeout=3600)
def download_controlnet_models():
    """Download all ControlNet models from HuggingFace"""
    from huggingface_hub import snapshot_download
    from pathlib import Path
    
    results = {"success": [], "failed": []}
    
    for name, repo_id in Config.CONTROLNET_MODELS.items():
        model_dir = Path(Config.CONTROLNET_DIR) / name
        
        if model_dir.exists():
            logger.info(f"[OK] ControlNet {name} already exists")
            results["success"].append(name)
            continue
        
        try:
            logger.info(f"[DOWNLOAD] ControlNet: {name} from {repo_id}...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False
            )
            controlnet_volume.commit()
            logger.info(f"[SUCCESS] ControlNet {name} downloaded")
            results["success"].append(name)
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to download ControlNet {name}: {e}")
            results["failed"].append({"name": name, "error": str(e)})
    
    return results

# ===================================================================
# MODEL INFERENCE CLASS
# ===================================================================
@app.cls(
    image=image,
    gpu=Config.GPU_TYPE,
    volumes={
        Config.MODEL_DIR: model_volume,
        Config.LORA_DIR: lora_volume,
        Config.CONTROLNET_DIR: controlnet_volume,
        Config.JSON_DATA_DIR: json_volume
    },
    container_idle_timeout=300,
    timeout=600
)
class ModelInference:
    @modal.enter()
    def load_model(self):
        """Initialize all models and pipelines - imports inside"""
        from diffusers import (
            StableDiffusionXLPipeline,
            StableDiffusionXLImg2ImgPipeline,
            EulerDiscreteScheduler
        )
        import torch
        from transformers import pipeline as transformers_pipeline
        
        logger.info("[INIT] Initializing Juggernaut ALL-IN-ONE model...")
        
        if not Path(Config.MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Model not found at {Config.MODEL_PATH}\n"
                f"Please run: modal run modal_app.py::download_model"
            )
        
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
            
            logger.info("[INIT] Initializing depth estimator...")
            self.depth_estimator = transformers_pipeline('depth-estimation')
            
            # Load LoRA configurations
            self.lora_models = load_lora_models_from_json(Config.JSON_DATA_DIR)
            logger.info(f"[OK] {len(self.lora_models)} LoRAs available")
            
            self.current_controlnet_pipe = None
            self.current_controlnet_type = None
            
            logger.info("[SUCCESS] Juggernaut ALL-IN-ONE Model loaded!")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to load model: {e}")
            raise

    def _cleanup_controlnet(self):
        """Cleanup ControlNet pipeline to prevent memory leaks"""
        if self.current_controlnet_pipe is not None:
            try:
                import torch
                import gc
                
                # Properly cleanup pipeline components
                if hasattr(self.current_controlnet_pipe, 'controlnet'):
                    del self.current_controlnet_pipe.controlnet
                
                del self.current_controlnet_pipe
                self.current_controlnet_pipe = None
                self.current_controlnet_type = None
                
                # Force garbage collection
                gc.collect()
                torch.cuda.empty_cache()
                
                logger.info("[CLEANUP] ControlNet pipeline cleaned up")
            except Exception as e:
                logger.warning(f"[WARNING] ControlNet cleanup warning: {e}")

    def _apply_lora(self, pipe, lora_name: Optional[str], lora_scale: float):
        """Apply LoRA weights to pipeline"""
        try:
            # Unload previous LoRA
            if hasattr(pipe, 'unload_lora_weights'):
                pipe.unload_lora_weights()
                logger.info("[INFO] Previous LoRA unloaded")
        except Exception as e:
            logger.debug(f"[DEBUG] LoRA unload: {e}")
        
        if lora_name and lora_name in self.lora_models:
            try:
                logger.info(f"[LORA] Applying LoRA: {lora_name}")
                lora_info = self.lora_models[lora_name]
                lora_path = f"{Config.LORA_DIR}/{lora_info['filename']}"
                
                if not Path(lora_path).exists():
                    raise FileNotFoundError(f"LoRA file not found: {lora_path}")
                
                pipe.load_lora_weights(lora_path, adapter_name=lora_name)
                pipe.fuse_lora(lora_scale=lora_scale, adapter_names=[lora_name])
                logger.info(f"[OK] LoRA {lora_name} applied with scale {lora_scale}")
                
            except Exception as e:
                logger.error(f"[ERROR] Failed to apply LoRA: {e}")
                raise

    def _preprocess_control_image(self, image, controlnet_type: str):
        """Preprocess control image based on type"""
        from controlnet_aux import OpenposeDetector
        from PIL import Image
        import cv2
        import numpy as np
        
        try:
            if controlnet_type == "openpose":
                processor = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
                return processor(image)
                
            elif controlnet_type == "canny":
                image_array = np.array(image)
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                canny_edges = cv2.Canny(image_bgr, 100, 200)
                canny_rgb = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2RGB)
                return Image.fromarray(canny_rgb)
                
            elif controlnet_type == "depth":
                depth = self.depth_estimator(image)['depth']
                return depth
            else:
                raise ValueError(f"Unknown controlnet_type: {controlnet_type}")
            
        except Exception as e:
            logger.error(f"[ERROR] Control image preprocessing failed: {e}")
            raise

    @modal.method()
    def text_to_image(self, request_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image from text prompt"""
        import torch
        
        try:
            # Get Pydantic models
            Text2ImageRequest, _, _ = get_pydantic_models()
            request = Text2ImageRequest(**request_dict)
            
            # Apply LoRA if specified
            self._apply_lora(self.txt2img_pipe, request.lora_name, request.lora_scale)
            
            enhanced_prompt = (
                f"{request.prompt}, {Config.DEFAULT_POSITIVE_SUFFIX}"
                if request.enhance_prompt
                else request.prompt
            )
            negative_prompt = request.negative_prompt or Config.DEFAULT_NEGATIVE_PROMPT
            
            logger.info(f"[GENERATE] Text-to-image...")
            
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
            
            logger.info("[SUCCESS] Text-to-image generation completed")
            
            return {
                "image": img_str,
                "prompt": enhanced_prompt,
                "original_prompt": request.prompt,
                "negative_prompt": negative_prompt,
                "seed": request.seed if request.seed != -1 else None,
                "width": request.width,
                "height": request.height,
                "lora": request.lora_name,
                "model": "Juggernaut ALL-IN-ONE",
                "version": Config.VERSION
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Text-to-image failed: {e}")
            raise

    @modal.method()
    def image_to_image(self, request_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Edit image with prompt"""
        from PIL import Image
        import torch
        
        try:
            # Get Pydantic models
            _, Image2ImageRequest, _ = get_pydantic_models()
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
                logger.info(f"[INFO] Preserving aspect ratio: {original_width}x{original_height} -> {target_width}x{target_height}")
            else:
                target_width, target_height = request.width, request.height
            
            init_image = init_image.resize(
                (target_width, target_height), 
                Image.Resampling.LANCZOS
            )
            
            # Apply LoRA if specified
            self._apply_lora(self.img2img_pipe, request.lora_name, request.lora_scale)
            
            enhanced_prompt = (
                f"{request.prompt}, {Config.DEFAULT_POSITIVE_SUFFIX}"
                if request.enhance_prompt
                else request.prompt
            )
            negative_prompt = request.negative_prompt or Config.DEFAULT_NEGATIVE_PROMPT
            
            logger.info(f"[GENERATE] Image-to-image...")
            
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
            
            logger.info("[SUCCESS] Image-to-image generation completed")
            
            return {
                "image": img_str,
                "prompt": enhanced_prompt,
                "original_prompt": request.prompt,
                "negative_prompt": negative_prompt,
                "strength": request.strength,
                "seed": request.seed if request.seed != -1 else None,
                "original_size": f"{original_width}x{original_height}",
                "output_size": f"{target_width}x{target_height}",
                "aspect_ratio_preserved": request.preserve_aspect_ratio,
                "lora": request.lora_name,
                "model": "Juggernaut ALL-IN-ONE",
                "version": Config.VERSION
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Image-to-image failed: {e}")
            raise

    @modal.method()
    def generate_with_controlnet(self, request_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image with ControlNet guidance"""
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
        from PIL import Image
        import torch
        
        try:
            # Get Pydantic models
            _, _, ControlNetRequest = get_pydantic_models()
            request = ControlNetRequest(**request_dict)
            
            control_image_bytes = base64.b64decode(request.control_image)
            validate_image_size(control_image_bytes)
            control_image = Image.open(io.BytesIO(control_image_bytes)).convert("RGB")
            
            if (self.current_controlnet_type != request.controlnet_type.value or
                self.current_controlnet_pipe is None):
                
                self._cleanup_controlnet()
                
                logger.info(f"[LOAD] ControlNet: {request.controlnet_type.value}")
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
            
            # Apply LoRA if specified
            self._apply_lora(self.current_controlnet_pipe, request.lora_name, request.lora_scale)
            
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
            
            logger.info(f"[GENERATE] ControlNet ({request.controlnet_type.value})...")
            
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
            
            logger.info("[SUCCESS] ControlNet generation completed")
            
            return {
                "image": img_str,
                "prompt": enhanced_prompt,
                "original_prompt": request.prompt,
                "negative_prompt": negative_prompt,
                "seed": request.seed if request.seed != -1 else None,
                "controlnet_type": request.controlnet_type.value,
                "controlnet_scale": request.controlnet_scale,
                "lora": request.lora_name,
                "model": "Juggernaut ALL-IN-ONE",
                "version": Config.VERSION
            }
            
        except Exception as e:
            logger.error(f"[ERROR] ControlNet generation failed: {e}")
            raise

# ===================================================================
# FASTAPI APPLICATION
# ===================================================================
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("custom-secret")],
    volumes={Config.JSON_DATA_DIR: json_volume}
)
@modal.asgi_app()
def fastapi_app():
    """FastAPI application - imports inside function"""
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    import os
    
    web_app = FastAPI(
        title="Juggernaut ALL-IN-ONE API (Fixed)",
        version=Config.VERSION,
        description="SDXL image generation with LoRA and ControlNet support - All Bugs Fixed"
    )
    
    limiter = Limiter(key_func=get_remote_address)
    web_app.state.limiter = limiter
    web_app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # Load LoRA configurations
    lora_models = load_lora_models_from_json(Config.JSON_DATA_DIR)
    
    async def verify_api_key(request: Request) -> Dict[str, Any]:
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
                logger.warning(f"[WARN] Invalid API key attempt from {request.client.host}")
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            return data
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[ERROR] API key verification error: {e}")
            raise HTTPException(status_code=400, detail="Invalid request format")
    
    @web_app.get("/")
    @limiter.limit("30/minute")
    async def root(request: Request):
        """API information endpoint"""
        return {
            "service": "Juggernaut ALL-IN-ONE API (Fixed)",
            "version": Config.VERSION,
            "gpu": Config.GPU_TYPE,
            "status": "operational",
            "model": "Juggernaut ALL-IN-ONE (CivitAI)",
            "improvements": [
                "[OK] Fixed all emoji encoding issues",
                "[OK] Removed invalid pybind11 from pip_install",
                "[OK] Fixed async/await for Modal remote methods",
                "[OK] Improved error handling and logging",
                "[OK] Fixed memory leak in ControlNet cleanup",
                "[OK] Added proper type hints",
                "[OK] Fixed partial download cleanup"
            ],
            "endpoints": {
                "health": "GET /health",
                "list_loras": "GET /loras?page=1&limit=50",
                "text_to_image": "POST /text2img",
                "image_to_image": "POST /img2img",
                "controlnet": "POST /controlnet"
            },
            "features": [
                "Text-to-Image (SDXL)",
                "Image-to-Image (SDXL)",
                "ControlNet (openpose, canny, depth)",
                f"{len(lora_models)} LoRA models available",
                "Aspect ratio preservation",
                "Request validation",
                "Rate limiting"
            ],
            "configuration": {
                "max_width": Config.MAX_WIDTH,
                "max_height": Config.MAX_HEIGHT,
                "max_image_size_mb": Config.MAX_IMAGE_SIZE_MB,
                "max_prompt_length": Config.MAX_PROMPT_LENGTH,
                "rate_limit": f"{Config.RATE_LIMIT_PER_MINUTE}/min"
            },
            "total_loras": len(lora_models),
            "controlnet_types": list(Config.CONTROLNET_MODELS.keys())
        }
    
    @web_app.get("/health")
    @limiter.limit("60/minute")
    async def health_check(request: Request):
        """Health check endpoint"""
        import sys
        
        return {
            "status": "healthy",
            "service": Config.APP_NAME,
            "version": Config.VERSION,
            "model": "Juggernaut ALL-IN-ONE",
            "gpu": Config.GPU_TYPE,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "total_loras": len(lora_models),
            "features": "all_bugs_fixed"
        }
    
    @web_app.get("/loras")
    @limiter.limit("30/minute")
    async def list_loras(
        request: Request,
        page: int = 1,
        limit: int = 50
    ):
        """Get paginated list of available LoRA models"""
        try:
            if page < 1:
                raise HTTPException(status_code=400, detail="Page must be >= 1")
            if limit < 1 or limit > 100:
                raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")
            
            lora_list = list(lora_models.keys())
            total = len(lora_list)
            
            start_idx = (page - 1) * limit
            end_idx = start_idx + limit
            
            paginated_loras = lora_list[start_idx:end_idx]
            
            return {
                "total": total,
                "page": page,
                "limit": limit,
                "total_pages": (total + limit - 1) // limit,
                "loras": paginated_loras,
                "has_next": end_idx < total,
                "has_prev": page > 1
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[ERROR] List LoRAs error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @web_app.post("/text2img")
    @limiter.limit(f"{Config.RATE_LIMIT_PER_MINUTE}/minute")
    async def text_to_image_endpoint(request: Request):
        """Text-to-image generation endpoint"""
        try:
            data = await verify_api_key(request)
            
            # Create ModelInference instance and call remote method
            # FIXED: Modal remote methods should be called synchronously in FastAPI endpoints
            model = ModelInference()
            result = model.text_to_image.remote(data)
            
            return JSONResponse(content=result)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[ERROR] Text2img endpoint error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @web_app.post("/img2img")
    @limiter.limit(f"{Config.RATE_LIMIT_PER_MINUTE}/minute")
    async def image_to_image_endpoint(request: Request):
        """Image-to-image generation endpoint"""
        try:
            data = await verify_api_key(request)
            
            # Create ModelInference instance and call remote method
            # FIXED: Modal remote methods should be called synchronously in FastAPI endpoints
            model = ModelInference()
            result = model.image_to_image.remote(data)
            
            return JSONResponse(content=result)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[ERROR] Img2img endpoint error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @web_app.post("/controlnet")
    @limiter.limit(f"{Config.RATE_LIMIT_PER_MINUTE}/minute")
    async def controlnet_endpoint(request: Request):
        """ControlNet generation endpoint"""
        try:
            data = await verify_api_key(request)
            
            # Create ModelInference instance and call remote method
            # FIXED: Modal remote methods should be called synchronously in FastAPI endpoints
            model = ModelInference()
            result = model.generate_with_controlnet.remote(data)
            
            return JSONResponse(content=result)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[ERROR] ControlNet endpoint error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return web_app

# ===================================================================
# LOCAL ENTRYPOINT
# ===================================================================
@app.local_entrypoint()
def main():
    """Local test entrypoint"""
    print("\n" + "="*70)
    print("[INIT] JUGGERNAUT ALL-IN-ONE API - FIXED VERSION v5.2.1")
    print("="*70)
    print("\n[OK] WHAT'S FIXED:")
    print("  [1] Removed invalid pybind11 from pip_install")
    print("  [2] Fixed all emoji encoding issues")
    print("  [3] Fixed async/await for Modal remote methods")
    print("  [4] Improved ControlNet memory cleanup with gc")
    print("  [5] Added proper type hints (Tuple)")
    print("  [6] Fixed partial download cleanup")
    print("  [7] Improved error handling in all functions")
    print("  [8] Fixed container_idle_timeout parameter")
    print("\n[INFO] STEP-BY-STEP DEPLOYMENT:\n")
    print("  [1] Download JSON configs (7 files):")
    print("      modal run modal_app.py::download_json_files\n")
    print("  [2] Download Juggernaut model (~6.5GB):")
    print("      modal run modal_app.py::download_model\n")
    print("  [3] Download ALL LoRAs from JSON:")
    print("      modal run modal_app.py::download_loras\n")
    print("  [4] Download ControlNet models:")
    print("      modal run modal_app.py::download_controlnet_models\n")
    print("  [5] Deploy to production:")
    print("      modal deploy modal_app.py\n")
    print("="*70)
    print("\n[INFO] DIRECTORY STRUCTURE:")
    print(f"  • Models: {Config.MODEL_DIR}")
    print(f"  • LoRAs: {Config.LORA_DIR}")
    print(f"  • ControlNet: {Config.CONTROLNET_DIR}")
    print(f"  • JSON configs: {Config.JSON_DATA_DIR}")
    print("\n[INFO] MODEL INFO:")
    print("  • Base: Juggernaut ALL-IN-ONE (SDXL)")
    print("  • Source: CivitAI")
    print("  • LoRAs: From 7 JSON files (GitHub)")
    print("  • ControlNet: OpenPose, Canny, Depth")
    print("="*70 + "\n")