# Filename: modal_app.py
"""
Deploy Model CivitAI ke Modal.com dengan FastAPI
Fitur: Text-to-Image dengan dukungan Multi-LoRA Dinamis
"""

import modal
from pathlib import Path
import io
import base64

# ===================================================================
# KONFIGURASI APLIKASI
# ===================================================================

app = modal.App("civitai-api-with-loras-final")

# Konfigurasi Model Utama (Juggernaut XL v9)
MODEL_DIR = "/models"
BASE_MODEL_URL = "https://civitai.com/api/download/models/333322"
BASE_MODEL_FILENAME = "juggernaut_v9.safetensors"

# Konfigurasi LoRA
LORA_DIR = "/loras"
LORA_MODELS = {
    # LoRA Awal
    "add_detail": {"url": "https://civitai.com/api/download/models/223332", "filename": "add_detail.safetensors"},
    "epi_noiseoffset": {"url": "https://civitai.com/api/download/models/10643", "filename": "epi_noiseoffset.safetensors"},
    
    # LoRA Tambahan
    "lora_slider": {"url": "https://civitai.com/api/download/models/309826", "filename": "lora_slider.safetensors"},
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

# Konfigurasi Prompt Asli Anda
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
DEFAULT_POSITIVE_PROMPT_SUFFIX = "masterpiece, best quality, 8k, photorealistic, intricate details, wide shot, (full body shot)"


# ===================================================================
# DEFINISI CONTAINER IMAGE & VOLUME
# ===================================================================

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]", "torch", "diffusers", "transformers",
        "accelerate", "safetensors", "Pillow", "requests",
    )
)

model_volume = modal.Volume.from_name("civitai-juggernaut-v9-vol", create_if_missing=True)
lora_volume = modal.Volume.from_name("civitai-loras-collection-vol", create_if_missing=True)


# ===================================================================
# FUNGSI DOWNLOAD (JALANKAN SEKALI SECARA MANUAL)
# ===================================================================

@app.function(image=image, volumes={MODEL_DIR: model_volume}, timeout=3600)
def download_main_model():
    import requests
    model_path = Path(MODEL_DIR) / BASE_MODEL_FILENAME
    if not model_path.exists():
        print(f"Downloading main model from {BASE_MODEL_URL}...")
        response = requests.get(BASE_MODEL_URL, stream=True)
        response.raise_for_status()
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
        model_volume.commit()
    print("Main model download check complete.")

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


# ===================================================================
# KELAS INFERENCE UTAMA DENGAN LOGIKA LORA
# ===================================================================

@app.cls(
    image=image,
    gpu="T4",
    volumes={MODEL_DIR: model_volume, LORA_DIR: lora_volume},
    container_idle_timeout=300,
    keep_warm=1
)
class ModelInference:
    @modal.enter()
    def load_model(self):
        from diffusers import StableDiffusionXLPipeline
        import torch
        
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            f"{MODEL_DIR}/{BASE_MODEL_FILENAME}",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        self.pipe.to("cuda")
        print("✓ SDXL Model loaded successfully!")

    def _apply_lora(self, lora_name: str, lora_scale: float):
        self.pipe.unload_lora_weights()
        if lora_name and lora_name in LORA_MODELS:
            print(f"Applying LoRA: {lora_name} with scale {lora_scale}")
            lora_info = LORA_MODELS[lora_name]
            lora_path = f"{LORA_DIR}/{lora_info['filename']}"
            self.pipe.load_lora_weights(lora_path, adapter_name=lora_name)
            self.pipe.fuse_lora(lora_scale=lora_scale, adapter_names=[lora_name])

    @modal.method()
    def text_to_image(self, prompt: str, **kwargs):
        import io, base64, torch
        
        lora_name = kwargs.get("lora_name")
        lora_scale = float(kwargs.get("lora_scale", 0.8))
        self._apply_lora(lora_name, lora_scale)
        
        enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}" if kwargs.get("enhance_prompt", True) else prompt
        negative_prompt = kwargs.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT
        
        generator = torch.Generator(device="cuda").manual_seed(kwargs.get("seed", -1)) if kwargs.get("seed", -1) != -1 else None
        
        image = self.pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=kwargs.get("num_steps", 30),
            guidance_scale=kwargs.get("guidance_scale", 7.5),
            width=kwargs.get("width", 1024),
            height=kwargs.get("height", 1024),
            generator=generator
        ).images[0]
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {"image": img_str}

# ===================================================================
# ENDPOINT API DENGAN FASTAPI
# ===================================================================

@app.function(
    secrets=[modal.Secret.from_name("custom-secret")],
    keep_warm=1
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    import os
    
    web_app = FastAPI()

    @web_app.get("/")
    def root():
        return {
            "service": "CivitAI API with LoRA support",
            "available_loras": list(LORA_MODELS.keys())
        }
        
    @web_app.post("/text2img")
    async def text_to_image_endpoint(request: Request):
        try:
            data = await request.json()
            api_key = data.get("api_key")
            if api_key != os.environ.get("API_KEY"):
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            prompt = data.get("prompt")
            if not prompt:
                raise HTTPException(status_code=400, detail="Prompt is required")
            
            model = ModelInference()
            result = model.text_to_image.remote(prompt, **data)
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return web_app
