# Filename: modal_app_rollback.py
"""
Deploy Model CivitAI ke Modal.com dengan FastAPI
Features: Text-to-Image, Image-to-Image, Uncensored
MODIFIED: Ditambahkan fungsi LoRA dan ControlNet, dan Image-to-Image disempurnakan.
"""

import modal
from pathlib import Path
import io
import base64

# ===================================================================
# KONFIGURASI ASLI ANDA (TIDAK DIUBAH)
# ===================================================================
app = modal.App("civitai-api-fastapi")

DEFAULT_NEGATIVE_PROMPT = (
    "(worst quality, low quality, normal quality, blurry, fuzzy, pixelated), "
    "(extra limbs, extra fingers, malformed hands, missing fingers, extra digit, "
    "fused fingers, too many hands, bad hands, bad anatomy), "
    "(ugly, deformed, disfigured), "
    "(text, watermark, logo, signature), "
    "(3D, CGI, render, rendering, video game, Unreal Engine, Blender, ZBrush, painting, drawing, sketch, illustration, digital art, concept art, artwork, style, stylized, cartoon, manga, comic, 2D, flat), "
    "out of frame, out of focus, "
    "cropped, close-up, portrait, headshot, medium shot, upper body, bust shot, face, out of frame"
)

DEFAULT_POSITIVE_PROMPT_SUFFIX = (
    "masterpiece, best quality, 8k, photorealistic, intricate details, wide shot, "
    "(full body shot)"
)

# ===================================================================
# PENAMBAHAN KONFIGURASI LORA & CONTROLNET
# ===================================================================
LORA_DIR = "/loras"
LORA_MODELS = {
    "add_detail": {"url": "https://civitai.com/api/download/models/223332", "filename": "add_detail.safetensors"},
    "epi_noiseoffset": {"url": "https://civitai.com/api/download/models/10643", "filename": "epi_noiseoffset.safetensors"},
    "detail_tweaker_xl": {"url": "https://civitai.com/api/download/models/122359", "filename": "detail_tweaker_xl.safetensors"},
    "jk_skirt": {"url": "https://civitai.com/api/download/models/120250", "filename": "jk_skirt.safetensors"},
    "mecha_angel": {"url": "https://civitai.com/api/download/models/116297", "filename": "mecha_angel.safetensors"},
    "makima_csm": {"url": "https://civitai.com/api/download/models/30141", "filename": "makima_csm.safetensors"},
    "russian_beauty": {"url": "https://civitai.com/api/download/models/1880455", "filename": "russian_beauty.safetensors"},
    "kpop_aesthetics": {"url": "https://civitai.com/api/download/models/2281898", "filename": "kpop_aesthetics.safetensors"},
    "oversized_concept": {"url": "https://civitai.com/api/download/models/255091", "filename": "oversized_concept.safetensors"},
    "chubby_concept": {"url": "https://civitai.com/api/download/models/191838", "filename": "chubby_concept.safetensors"},
    "stacia_asuna": {"url": "https://civitai.com/api/download/models/15754", "filename": "stacia_asuna.safetensors"},
    "asian_male": {"url": "https://civitai.com/api/download/models/20348", "filename": "asian_male.safetensors"},
    "asian_girls_face": {"url": "https://civitai.com/api/download/models/86339", "filename": "asian_girls_face.safetensors"},
    "style_asian_less": {"url": "https://civitai.com/api/download/models/55490", "filename": "style_asian_less.safetensors"},
    "retro_1990s_kpop": {"url": "https://civitai.com/api/download/models/2275608", "filename": "retro_1990s_kpop.safetensors"},
    "kpop_concept_photos": {"url": "https://civitai.com/api/download/models/1655610", "filename": "kpop_concept_photos.safetensors"}
}

CONTROLNET_DIR = "/controlnet_models"
CONTROLNET_MODELS = {
    "openpose": "thibaud/controlnet-openpose-sdxl-1.0",
    "canny": "diffusers/controlnet-canny-sdxl-1.0",
    "depth": "diffusers/controlnet-depth-sdxl-1.0",
}

# ===================================================================
# DEFINISI IMAGE & VOLUME
# ===================================================================
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]", "torch", "diffusers", "transformers",
        "accelerate", "safetensors", "Pillow", "requests", "huggingface_hub", "opencv-python-headless"
    )
)

model_volume = modal.Volume.from_name("civitai-models", create_if_missing=True)
lora_volume = modal.Volume.from_name("civitai-loras-collection-vol", create_if_missing=True)
controlnet_volume = modal.Volume.from_name("controlnet-sdxl-collection-vol", create_if_missing=True)

# ===================================================================
# FUNGSI DOWNLOAD
# ===================================================================
@app.function(image=image, volumes={MODEL_DIR: model_volume}, timeout=3600)
def download_model():
    import requests
    from pathlib import Path
    
    model_url = "https://civitai.com/api/download/models/348913?type=Model&format=SafeTensor&size=full&fp=fp16"
    model_path = Path(MODEL_DIR) / "model.safetensors"
    
    if not model_path.exists():
        print("Downloading model...")
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
        model_volume.commit()
        print(f"Model downloaded to {model_path}")
    else:
        print(f"Model already exists at {model_path}")

@app.function(image=image, volumes={LORA_DIR: lora_volume}, timeout=3600)
def download_loras():
    import requests
    for name, data in LORA_MODELS.items():
        lora_path = Path(LORA_DIR) / data["filename"]
        if not lora_path.exists():
            print(f"Downloading LoRA: {name}...")
            response = requests.get(data["url"], stream=True)
            response.raise_for_status()
            with open(lora_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
            lora_volume.commit()
    print("All LoRAs download check complete.")

@app.function(image=image, volumes={CONTROLNET_DIR: controlnet_volume}, timeout=3600)
def download_controlnet_models():
    from huggingface_hub import snapshot_download
    for name, repo_id in CONTROLNET_MODELS.items():
        model_dir = Path(CONTROLNET_DIR) / name
        if not model_dir.exists():
            print(f"Downloading ControlNet model: {name} from {repo_id}...")
            snapshot_download(repo_id=repo_id, local_dir=model_dir, local_dir_use_symlinks=False)
            controlnet_volume.commit()
    print("All ControlNet models download check complete.")


# ===================================================================
# KELAS INFERENCE
# ===================================================================
@app.cls(
    image=image,
    gpu="A10G",
    volumes={MODEL_DIR: model_volume, LORA_DIR: lora_volume, CONTROLNET_DIR: controlnet_volume},
    container_idle_timeout=200
)
class ModelInference:
    @modal.enter()
    def load_model(self):
        from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
        from diffusers import EulerDiscreteScheduler # Import scheduler
        import torch
        
        print("Loading SDXL model...")
        model_path = f"{MODEL_DIR}/model.safetensors"
        
        self.txt2img_pipe = StableDiffusionXLPipeline.from_single_file(
            model_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )
        # PENAMBAHAN: Mengatur scheduler untuk txt2img
        self.txt2img_pipe.scheduler = EulerDiscreteScheduler.from_config(self.txt2img_pipe.scheduler.config)
        self.txt2img_pipe.to("cuda")
        
        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
            vae=self.txt2img_pipe.vae, text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2, tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2, unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler, # Menggunakan scheduler yang sama
        )
        self.img2img_pipe.to("cuda")
        print("✓ SDXL Model loaded successfully! Uncensored mode active.")

    def _apply_lora(self, pipe, lora_name: str, lora_scale: float):
        pipe.unload_lora_weights()
        if lora_name and lora_name in LORA_MODELS:
            print(f"Applying LoRA: {lora_name} with scale {lora_scale}")
            lora_info = LORA_MODELS[lora_name]
            lora_path = f"{LORA_DIR}/{lora_info['filename']}"
            pipe.load_lora_weights(lora_path, adapter_name=lora_name)
            pipe.fuse_lora(lora_scale=lora_scale, adapter_names=[lora_name])

    @modal.method()
    def text_to_image(self, prompt: str, lora_name: str = None, lora_scale: float = 0.8, **kwargs):
        self._apply_lora(self.txt2img_pipe, lora_name, lora_scale)
        
        enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}" if kwargs.get("enhance_prompt", True) else prompt
        negative_prompt = kwargs.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT
        
        import torch
        generator = None
        if kwargs.get("seed", -1) != -1:
            generator = torch.Generator(device="cuda").manual_seed(kwargs.get("seed", -1))
        
        image = self.txt2img_pipe(
            prompt=enhanced_prompt, negative_prompt=negative_prompt,
            num_inference_steps=kwargs.get("num_steps", 25),
            guidance_scale=kwargs.get("guidance_scale", 7.5),
            width=kwargs.get("width", 1024), height=kwargs.get("height", 1024),
            generator=generator
        ).images[0]
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "image": img_str, "prompt": enhanced_prompt, "original_prompt": prompt,
            "negative_prompt": negative_prompt, "seed": kwargs.get("seed", -1), "uncensored": True
        }
    
    # === PERUBAHAN UTAMA: image_to_image YANG DISEMPURNAKAN ===
    @modal.method()
    def image_to_image(
        self,
        init_image_b64: str,
        prompt: str,
        negative_prompt: str = "",
        num_steps: int = 25,
        guidance_scale: float = 7.5,
        strength: float = 0.75,
        seed: int = -1,
        enhance_prompt: bool = True,
        lora_name: str = None, 
        lora_scale: float = 0.8,
        # PENAMBAHAN PARAMETER BARU UNTUK KONTROL UKURAN
        width: int = 1024,
        height: int = 1024
    ):
        """Edit image dengan prompt (disempurnakan untuk SDXL)"""
        import io
        import base64
        from PIL import Image
        import torch
        
        self._apply_lora(self.img2img_pipe, lora_name, lora_scale)
        
        if enhance_prompt:
            enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}"
        else:
            enhanced_prompt = prompt
        
        # Menggunakan DEFAULT_NEGATIVE_PROMPT jika tidak disediakan atau kosong
        if negative_prompt is None or not str(negative_prompt).strip():
            negative_prompt = DEFAULT_NEGATIVE_PROMPT
        else:
            negative_prompt = str(negative_prompt).strip()
        
        print(f"Image-to-Image (SDXL): {enhanced_prompt[:100]}...")
        
        init_image_bytes = base64.b64decode(init_image_b64)
        init_image = Image.open(io.BytesIO(init_image_bytes)).convert("RGB")
        
        # === PENAMBAHAN: Resizing init_image ke ukuran yang diminta ===
        # Memastikan init_image memiliki dimensi yang sama dengan target output
        init_image = init_image.resize((width, height), Image.LANCZOS)
        
        generator = None
        if seed != -1:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        
        image = self.img2img_pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            # PENAMBAHAN: Menentukan ukuran output secara eksplisit
            width=width,
            height=height
        ).images[0]
        
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
            "uncensored": True,
            "output_width": width, # Menambahkan info ukuran output
            "output_height": height # Menambahkan info ukuran output
        }

    @modal.method()
    def generate_with_controlnet(self, prompt: str, control_image_b64: str, controlnet_type: str, lora_name: str = None, lora_scale: float = 0.8, **kwargs):
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
        from PIL import Image
        import torch
        
        if controlnet_type not in CONTROLNET_MODELS:
            raise ValueError(f"Invalid controlnet_type. Supported: {list(CONTROLNET_MODELS.keys())}")

        controlnet = ControlNetModel.from_pretrained(str(Path(CONTROLNET_DIR) / controlnet_type), torch_dtype=torch.float16)
        
        control_pipe = StableDiffusionXLControlNetPipeline(
            vae=self.txt2img_pipe.vae, text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2, tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2, unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler, controlnet=controlnet,
        )
        control_pipe.to("cuda")

        self._apply_lora(control_pipe, lora_name, lora_scale)

        control_image_bytes = base64.b64decode(control_image_b64)
        control_image = Image.open(io.BytesIO(control_image_bytes)).convert("RGB")
        
        enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}"
        negative_prompt = kwargs.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT
        generator = None
        if kwargs.get("seed", -1) != -1:
            generator = torch.Generator(device="cuda").manual_seed(kwargs.get("seed", -1))

        image = control_pipe(
            enhanced_prompt, negative_prompt=negative_prompt, image=control_image,
            num_inference_steps=kwargs.get("num_steps", 30),
            guidance_scale=kwargs.get("guidance_scale", 7.5),
            controlnet_conditioning_scale=float(kwargs.get("controlnet_scale", 0.8)),
            generator=generator,
        ).images[0]
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "image": img_str, "prompt": enhanced_prompt, "original_prompt": prompt,
            "negative_prompt": negative_prompt, "seed": kwargs.get("seed", -1), "uncensored": True
        }


# ===================================================================
# ENDPOINT API
# ===================================================================
@app.function(image=image, secrets=[modal.Secret.from_name("custom-secret")])
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    import os
    
    web_app = FastAPI()
    
    @web_app.get("/")
    async def root():
        return {
            "service": "CivitAI Model API - Uncensored (SDXL)",
            "version": "3.1", # Versi diperbarui untuk mencerminkan penyempurnaan
            "endpoints": {
                "health": "GET /health",
                "text-to-image": "POST /text2img",
                "image-to-image": "POST /img2img",
                "controlnet": "POST /controlnet"
            },
            "features": [
                "✓ No NSFW filter", 
                "✓ Uncensored generation",
                "✓ Text-to-Image (SDXL)", 
                "✓ Image-to-Image (SDXL) - **Disempurnakan**",
                "✓ ControlNet (SDXL)", # Ditambahkan
                "✓ Multi-LoRA support", # Ditambahkan
                "✓ Auto quality enhancement",
                "✓ Default negative prompts for best results"
            ],
            "default_prompts": {
                "positive_suffix": DEFAULT_POSITIVE_PROMPT_SUFFIX,
                "negative": DEFAULT_NEGATIVE_PROMPT
            },
            "available_loras": list(LORA_MODELS.keys()), # Ditambahkan
            "available_controlnets": list(CONTROLNET_MODELS.keys()) # Ditambahkan
        }

    @web_app.get("/health")
    async def health_check():
        return {
            "status": "healthy", 
            "service": "civitai-model-api",
            "mode": "uncensored-sdxl"
        }

    @web_app.post("/text2img")
    async def text_to_image_endpoint(request: Request):
        try:
            data = await request.json()
            api_key = data.get("api_key")
            if api_key != os.environ.get("API_KEY"):
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            prompt = data.get("prompt")
            if not prompt: raise HTTPException(status_code=400, detail="Prompt is required")
            
            lora_name = data.get("lora_name")
            lora_scale = data.get("lora_scale", 0.8)
            
            model = ModelInference()
            result = model.text_to_image.remote(prompt=prompt, lora_name=lora_name, lora_scale=lora_scale, **data)
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.post("/img2img")
    async def image_to_image_endpoint(request: Request):
        try:
            data = await request.json()
            api_key = data.get("api_key")
            if api_key != os.environ.get("API_KEY"):
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            init_image = data.get("init_image")
            prompt = data.get("prompt")
            if not init_image or not prompt:
                raise HTTPException(status_code=400, detail="init_image and prompt are required")
            
            lora_name = data.get("lora_name")
            lora_scale = data.get("lora_scale", 0.8)
            
            model = ModelInference()
            result = model.image_to_image.remote(
                init_image_b64=init_image, prompt=prompt, lora_name=lora_name, lora_scale=lora_scale,
                width=data.get("width", 1024), # Menambahkan width
                height=data.get("height", 1024), # Menambahkan height
                **data
            )
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.post("/controlnet")
    async def controlnet_endpoint(request: Request):
        try:
            data = await request.json()
            api_key = data.get("api_key")
            if api_key != os.environ.get("API_KEY"):
                raise HTTPException(status_code=401, detail="Invalid API key")

            prompt = data.get("prompt")
            control_image = data.get("control_image")
            controlnet_type = data.get("controlnet_type")
            if not all([prompt, control_image, controlnet_type]):
                raise HTTPException(400, "prompt, control_image (base64), and controlnet_type are required.")
            
            lora_name = data.get("lora_name")
            lora_scale = data.get("lora_scale", 0.8)

            model = ModelInference()
            result = model.generate_with_controlnet.remote(
                prompt=prompt, control_image_b64=control_image, controlnet_type=controlnet_type,
                lora_name=lora_name, lora_scale=lora_scale, **data
            )
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return web_app

@app.local_entrypoint()
def main():
    """Test locally"""
    pass
