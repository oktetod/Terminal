# Filename: modal_app.py
"""
Deploy Model CivitAI ke Modal.com dengan FastAPI
Versi Final: GPU L4, Filter NSFW, LoRA, ControlNet
"""

import modal
from pathlib import Path
import io
import base64

# ===================================================================
# KONFIGURASI APLIKASI
# ===================================================================

app = modal.App("civitai-api-ultimate-l4")

# Konfigurasi Model Utama (Juggernaut XL v9)
MODEL_DIR = "/models"
BASE_MODEL_URL = "https://civitai.com/api/download/models/333322"
BASE_MODEL_FILENAME = "juggernaut_v9.safetensors"

# Konfigurasi LoRA
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

# Konfigurasi ControlNet
CONTROLNET_DIR = "/controlnet_models"
CONTROLNET_MODELS = {
    "openpose": "thibaud/controlnet-openpose-sdxl-1.0",
    "canny": "diffusers/controlnet-canny-sdxl-1.0",
    "depth": "diffusers/controlnet-depth-sdxl-1.0",
}

# === PERUBAHAN: NEGATIVE PROMPT DENGAN FILTER NSFW ===
DEFAULT_NEGATIVE_PROMPT = (
    "nsfw, nude, naked, porn, sex, sexual, explicit, uncensored, "
    "ass, breasts, nipple, pussy, genitalia, "
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
        "fastapi==0.110.0", "uvicorn", "pydantic", "starlette",
        "torch==2.1.2", "diffusers==0.24.0", "transformers==4.35.2",
        "accelerate", "safetensors", "Pillow", "requests", "huggingface_hub", "opencv-python-headless"
    )
)

model_volume = modal.Volume.from_name("civitai-juggernaut-v9-vol", create_if_missing=True)
lora_volume = modal.Volume.from_name("civitai-loras-collection-vol", create_if_missing=True)
controlnet_volume = modal.Volume.from_name("controlnet-sdxl-collection-vol", create_if_missing=True)


# ===================================================================
# FUNGSI DOWNLOAD
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
# KELAS INFERENCE UTAMA
# ===================================================================

@app.cls(
    # === PERUBAHAN: MENGGUNAKAN GPU L4 ===
    gpu="L4",
    volumes={MODEL_DIR: model_volume, LORA_DIR: lora_volume, CONTROLNET_DIR: controlnet_volume},
    container_idle_timeout=300, 
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
        print("✓ SDXL Base Model (Juggernaut) loaded successfully!")

    def _apply_lora(self, pipe, lora_name: str, lora_scale: float):
        pipe.unload_lora_weights()
        if lora_name and lora_name in LORA_MODELS:
            print(f"Applying LoRA: {lora_name} with scale {lora_scale}")
            lora_info = LORA_MODELS[lora_name]
            lora_path = f"{LORA_DIR}/{lora_info['filename']}"
            pipe.load_lora_weights(lora_path, adapter_name=lora_name)
            pipe.fuse_lora(lora_scale=lora_scale, adapter_names=[lora_name])

    @modal.method()
    def text_to_image(self, prompt: str, **kwargs):
        import io, base64, torch
        
        self._apply_lora(self.pipe, kwargs.get("lora_name"), float(kwargs.get("lora_scale", 0.8)))
        
        enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}"
        negative_prompt = kwargs.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT
        
        generator = torch.Generator(device="cuda").manual_seed(kwargs.get("seed", -1)) if kwargs.get("seed", -1) != -1 else None
        
        image = self.pipe(
            prompt=enhanced_prompt, negative_prompt=negative_prompt,
            num_inference_steps=kwargs.get("num_steps", 30),
            guidance_scale=kwargs.get("guidance_scale", 7.5),
            width=kwargs.get("width", 1024), height=kwargs.get("height", 1024),
            generator=generator
        ).images[0]
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return {"image": base64.b64encode(buffered.getvalue()).decode()}

    @modal.method()
    def generate_with_controlnet(self, prompt: str, control_image_b64: str, controlnet_type: str, **kwargs):
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
        from PIL import Image
        import torch
        
        if controlnet_type not in CONTROLNET_MODELS:
            raise ValueError(f"Invalid controlnet_type. Supported: {list(CONTROLNET_MODELS.keys())}")

        controlnet = ControlNetModel.from_pretrained(str(Path(CONTROLNET_DIR) / controlnet_type), torch_dtype=torch.float16)
        
        control_pipe = StableDiffusionXLControlNetPipeline(
            vae=self.pipe.vae, text_encoder=self.pipe.text_encoder, text_encoder_2=self.pipe.text_encoder_2,
            tokenizer=self.pipe.tokenizer, tokenizer_2=self.pipe.tokenizer_2,
            unet=self.pipe.unet, scheduler=self.pipe.scheduler, controlnet=controlnet,
        )
        control_pipe.to("cuda")

        self._apply_lora(control_pipe, kwargs.get("lora_name"), float(kwargs.get("lora_scale", 0.8)))

        control_image_bytes = base64.b64decode(control_image_b64)
        control_image = Image.open(io.BytesIO(control_image_bytes)).convert("RGB")
        
        enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}"
        negative_prompt = kwargs.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT
        
        generator = torch.Generator(device="cuda").manual_seed(kwargs.get("seed", -1)) if kwargs.get("seed", -1) != -1 else None

        image = control_pipe(
            enhanced_prompt, negative_prompt=negative_prompt, image=control_image,
            num_inference_steps=kwargs.get("num_steps", 30),
            guidance_scale=kwargs.get("guidance_scale", 7.5),
            controlnet_conditioning_scale=float(kwargs.get("controlnet_scale", 0.8)),
            generator=generator,
        ).images[0]
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return {"image": base64.b64encode(buffered.getvalue()).decode()}

# ===================================================================
# ENDPOINT API DENGAN FASTAPI
# ===================================================================

@app.function(
    secrets=[modal.Secret.from_name("custom-secret")],
    min_containers=1,
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
            "service": "CivitAI Ultimate API (T2I, LoRA, ControlNet)",
            "endpoints": ["/text2img", "/controlnet"],
            "available_loras": list(LORA_MODELS.keys()),
            "available_controlnets": list(CONTROLNET_MODELS.keys())
        }
        
    @web_app.post("/{endpoint:path}")
    async def handle_generation(endpoint: str, request: Request):
        if endpoint not in ["text2img", "controlnet"]:
            raise HTTPException(status_code=404, detail="Endpoint not found")

        try:
            data = await request.json()
            api_key = data.get("api_key")
            if api_key != os.environ.get("API_KEY"):
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            prompt = data.get("prompt")
            if not prompt:
                raise HTTPException(status_code=400, detail="Prompt is required")
            
            model = ModelInference()
            result = None

            if endpoint == "text2img":
                result = model.text_to_image.remote(prompt, **data)
            elif endpoint == "controlnet":
                control_image = data.get("control_image")
                controlnet_type = data.get("controlnet_type")
                if not all([control_image, controlnet_type]):
                    raise HTTPException(400, "control_image (base64) and controlnet_type are required.")
                result = model.generate_with_controlnet.remote(prompt, control_image, controlnet_type, **data)

            return JSONResponse(content=result)
        except Exception as e:
            print(f"Internal Server Error: {e}") 
            raise HTTPException(status_code=500, detail="An internal server error occurred.")
    
    return web_app
