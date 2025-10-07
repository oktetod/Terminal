"""
Deploy Model CivitAI ke Modal.com dengan FastAPI
Features: Text-to-Image, Image-to-Image, Uncensored
"""

import modal
from pathlib import Path

# Inisialisasi Modal app
app = modal.App("civitai-api-fastapi")

# Default prompts untuk quality control
DEFAULT_NEGATIVE_PROMPT = (
    "(worst quality, low quality, normal quality, blurry, fuzzy, pixelated), "
    "(extra limbs, extra fingers, malformed hands, missing fingers, extra digit, "
    "fused fingers, too many hands, bad hands, bad anatomy), "
    "(ugly, deformed, disfigured), "
    "(text, watermark, logo, signature), "
    "(dark skin, black hair, ugly face), "
    "out of frame, out of focus"
)

DEFAULT_POSITIVE_PROMPT_SUFFIX = (
    "masterpiece, best quality, 8k, photorealistic, intricate details, "
    "finely detailed skin, realistic texture, perfect anatomy"
)

# Definisikan image dengan dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]",
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "Pillow",
        "requests",
    )
)

# Volume untuk menyimpan model
model_volume = modal.Volume.from_name("civitai-models", create_if_missing=True)
MODEL_DIR = "/models"

# Download model (jalankan sekali)
@app.function(
    image=image,
    volumes={MODEL_DIR: model_volume},
    timeout=3600
)
def download_model():
    """Download model dari CivitAI"""
    import requests
    from pathlib import Path
    
    model_url = "https://civitai.com/api/download/models/1759168?type=Model&format=SafeTensor&size=full&fp=fp16"
    model_path = Path(MODEL_DIR) / "model.safetensors"
    
    print("Downloading model...")
    response = requests.get(model_url, stream=True)
    response.raise_for_status()
    
    with open(model_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    model_volume.commit()
    print(f"Model downloaded to {model_path}")
    return str(model_path)


# Class untuk inference
@app.cls(
    image=image,
    gpu="T4",
    volumes={MODEL_DIR: model_volume},
    container_idle_timeout=300  # Auto-shutdown setelah 5 menit idle
)
class ModelInference:
    @modal.enter()
    def load_model(self):
        """Load model saat container start"""
        from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
        import torch
        
        print("Loading model...")
        model_path = f"{MODEL_DIR}/model.safetensors"
        
        # Load Text-to-Image pipeline - NO SAFETY CHECKER
        self.txt2img_pipe = StableDiffusionPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None,  # Completely disabled
            requires_safety_checker=False
        )
        self.txt2img_pipe.to("cuda")
        
        # Load Image-to-Image pipeline - NO SAFETY CHECKER
        self.img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=None,  # Completely disabled
            feature_extractor=None,
            requires_safety_checker=False
        )
        self.img2img_pipe.to("cuda")
        
        print("✓ Model loaded successfully! Uncensored mode active.")
    
    @modal.method()
    def text_to_image(
        self, 
        prompt: str, 
        negative_prompt: str = "", 
        num_steps: int = 20, 
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: int = -1,
        enhance_prompt: bool = True
    ):
        """Generate image dari text prompt"""
        import io
        import base64
        import torch
        
        # Enhance prompt dengan default suffix jika diminta
        if enhance_prompt:
            enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}"
        else:
            enhanced_prompt = prompt
        
        # ROBUST: Handle None, empty string, or whitespace
        if negative_prompt is None or not str(negative_prompt).strip():
            negative_prompt = DEFAULT_NEGATIVE_PROMPT
        else:
            negative_prompt = str(negative_prompt).strip()
        
        print(f"Text-to-Image: {enhanced_prompt[:100]}...")
        
        # Set seed untuk reproducibility
        generator = None
        if seed != -1:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        
        image = self.txt2img_pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator
        ).images[0]
        
        # Convert ke base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "image": img_str, 
            "prompt": enhanced_prompt,
            "original_prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed if seed != -1 else "random",
            "uncensored": True
        }
    
    @modal.method()
    def image_to_image(
        self,
        init_image_b64: str,
        prompt: str,
        negative_prompt: str = "",
        num_steps: int = 20,
        guidance_scale: float = 7.5,
        strength: float = 0.75,
        seed: int = -1,
        enhance_prompt: bool = True
    ):
        """Edit image dengan prompt"""
        import io
        import base64
        from PIL import Image
        import torch
        
        # Enhance prompt dengan default suffix jika diminta
        if enhance_prompt:
            enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}"
        else:
            enhanced_prompt = prompt
        
        # ROBUST: Handle None, empty string, or whitespace
        if negative_prompt is None or not str(negative_prompt).strip():
            negative_prompt = DEFAULT_NEGATIVE_PROMPT
        else:
            negative_prompt = str(negative_prompt).strip()
        
        print(f"Image-to-Image: {enhanced_prompt[:100]}...")
        
        # Decode base64 image
        init_image_bytes = base64.b64decode(init_image_b64)
        init_image = Image.open(io.BytesIO(init_image_bytes)).convert("RGB")
        
        # Set seed
        generator = None
        if seed != -1:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # Generate
        image = self.img2img_pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=strength,  # 0.0 = no change, 1.0 = full change
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        # Convert ke base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "image": img_str,
            "prompt": enhanced_prompt,
            "original_prompt": prompt,
            "negative_prompt": negative_prompt,
            "strength": strength,
            "seed": seed if seed != -1 else "random",
            "uncensored": True
        }


# Mount FastAPI app ke Modal
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("api-secret")]
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    import os
    
    web_app = FastAPI()

    @web_app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "service": "CivitAI Model API - Uncensored",
            "version": "2.0",
            "endpoints": {
                "health": "GET /health",
                "text-to-image": "POST /text2img",
                "image-to-image": "POST /img2img"
            },
            "features": [
                "✓ No NSFW filter", 
                "✓ Uncensored generation",
                "✓ Text-to-Image", 
                "✓ Image-to-Image",
                "✓ Auto quality enhancement",
                "✓ Default negative prompts for best results"
            ],
            "default_prompts": {
                "positive_suffix": DEFAULT_POSITIVE_PROMPT_SUFFIX,
                "negative": DEFAULT_NEGATIVE_PROMPT
            }
        }

    @web_app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy", 
            "service": "civitai-model-api",
            "mode": "uncensored"
        }

    @web_app.post("/text2img")
    async def text_to_image_endpoint(request: Request):
        """
        Text-to-Image endpoint (Uncensored)
        
        Body:
        {
            "prompt": "your prompt here",
            "negative_prompt": "optional (uses default if not provided)",
            "num_steps": 20,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 512,
            "seed": -1,
            "enhance_prompt": true,
            "api_key": "your-api-key"
        }
        
        If enhance_prompt=true, adds quality enhancement suffix automatically.
        If negative_prompt not provided, uses default quality control negative prompt.
        """
        try:
            data = await request.json()
            
            # Validasi API key
            api_key = data.get("api_key")
            expected_key = os.environ.get("API_KEY")
            
            if api_key != expected_key:
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            prompt = data.get("prompt")
            if not prompt:
                raise HTTPException(status_code=400, detail="Prompt is required")
            
            # JANGAN kirim negative_prompt jika None/empty - biar method pakai default
            kwargs = {
                "prompt": prompt,
                "num_steps": data.get("num_steps", 20),
                "guidance_scale": data.get("guidance_scale", 7.5),
                "width": data.get("width", 512),
                "height": data.get("height", 512),
                "seed": data.get("seed", -1),
                "enhance_prompt": data.get("enhance_prompt", True)
            }
            
            # Hanya tambahkan negative_prompt jika ada dan tidak kosong
            neg = data.get("negative_prompt")
            if neg and str(neg).strip():
                kwargs["negative_prompt"] = str(neg).strip()
            
            # Generate image
            model = ModelInference()
            result = model.text_to_image.remote(**kwargs)
            
            return JSONResponse(content=result)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @web_app.post("/img2img")
    async def image_to_image_endpoint(request: Request):
        """
        Image-to-Image endpoint (Uncensored)
        
        Body:
        {
            "init_image": "base64_string_of_image",
            "prompt": "your prompt",
            "negative_prompt": "optional (uses default if not provided)",
            "num_steps": 20,
            "guidance_scale": 7.5,
            "strength": 0.75,
            "seed": -1,
            "enhance_prompt": true,
            "api_key": "your-api-key"
        }
        
        strength: 0.0-1.0 (0=no change, 1=full change)
        If enhance_prompt=true, adds quality enhancement suffix automatically.
        If negative_prompt not provided, uses default quality control negative prompt.
        """
        try:
            data = await request.json()
            
            # Validasi API key
            api_key = data.get("api_key")
            expected_key = os.environ.get("API_KEY")
            
            if api_key != expected_key:
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            init_image = data.get("init_image")
            if not init_image:
                raise HTTPException(status_code=400, detail="init_image is required")
            
            prompt = data.get("prompt")
            if not prompt:
                raise HTTPException(status_code=400, detail="prompt is required")
            
            # JANGAN kirim negative_prompt jika None/empty - biar method pakai default
            kwargs = {
                "init_image_b64": init_image,
                "prompt": prompt,
                "num_steps": data.get("num_steps", 20),
                "guidance_scale": data.get("guidance_scale", 7.5),
                "strength": data.get("strength", 0.75),
                "seed": data.get("seed", -1),
                "enhance_prompt": data.get("enhance_prompt", True)
            }
            
            # Hanya tambahkan negative_prompt jika ada dan tidak kosong
            neg = data.get("negative_prompt")
            if neg and str(neg).strip():
                kwargs["negative_prompt"] = str(neg).strip()
            
            # Generate image
            model = ModelInference()
            result = model.image_to_image.remote(**kwargs)
            
            return JSONResponse(content=result)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return web_app


# Local entry untuk testing
@app.local_entrypoint()
def main():
    """Test locally"""
    model = ModelInference()
    
    # Test Text-to-Image dengan enhancement
    result = model.text_to_image.remote(
        prompt="beautiful woman portrait",
        num_steps=25,
        enhance_prompt=True
    )
    print("✓ Text-to-Image Success!")
    print(f"  Original Prompt: {result['original_prompt']}")
    print(f"  Enhanced Prompt: {result['prompt'][:100]}...")
    print(f"  Uncensored: {result['uncensored']}")
