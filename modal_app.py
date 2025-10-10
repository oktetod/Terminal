# Filename: modal_app.py
"""
Deploy Model CivitAI ke Modal.com dengan FastAPI
Features: Text-to-Image, Image-to-Image, ControlNet, Multi-LoRA
Fixed: Proper JSON file mounting
"""

import modal
from pathlib import Path
import io
import base64
import json

# ===================================================================
# KONFIGURASI
# ===================================================================
app = modal.App("civitai-api-fastapi")

DEFAULT_NEGATIVE_PROMPT = (
    "(worst quality, low quality, normal quality, blurry, fuzzy, pixelated), "
    "(extra limbs, extra fingers, malformed hands, missing fingers, extra digit, "
    "fused fingers, too many hands, bad hands, bad anatomy), "
    "(ugly, deformed, disfigured), "
    "(text, watermark, logo, signature), "
    "(worst quality, low quality, normal quality:1.4), (jpeg artifacts, blurry, grainy), ugly, duplicate, morbid, mutilated, (deformed, disfigured), (bad anatomy, bad proportions), (extra limbs, extra fingers, fused fingers, too many fingers, long neck), (mutated hands, bad hands, poorly drawn hands), (missing arms, missing legs), malformed limbs, (cross-eyed, bad eyes, asymmetrical eyes), (cleavage), signature, watermark, username, text, error, "
    "out of frame, out of focus, "
    "cropped, close-up, portrait, headshot, medium shot, upper body, bust shot, face, out of frame"
)

DEFAULT_POSITIVE_PROMPT_SUFFIX = (
    "masterpiece, best quality, 8k, photorealistic, intricate details, wide shot, "
    "(full body shot)"
)

# ===================================================================
# KONFIGURASI LORA & CONTROLNET
# ===================================================================
LORA_DIR = "/loras"
CONTROLNET_DIR = "/controlnet_models"
CONTROLNET_MODELS = {
    "openpose": "thibaud/controlnet-openpose-sdxl-1.0",
    "canny": "diffusers/controlnet-canny-sdxl-1.0",
    "depth": "diffusers/controlnet-depth-sdxl-1.0",
}
MODEL_DIR = "/models"

# JSON files will be mounted as a volume at runtime
JSON_DATA_DIR = "/json_data"

# Function to load all LoRA models from JSON files
def load_lora_models_from_json(json_dir=JSON_DATA_DIR):
    """
    Load all LoRA models from JSON files (01.json to 07.json)
    Returns a dictionary with all LoRA configurations
    """
    lora_models = {}
    json_files = ["01.json", "03.json", "04.json", "05.json", "07.json"]
    
    for json_file in json_files:
        json_path = Path(json_dir) / json_file
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Filter out comment keys (keys starting with "//")
                filtered_data = {k: v for k, v in data.items() if not k.startswith("//")}
                lora_models.update(filtered_data)
                print(f"✓ Loaded {len(filtered_data)} LoRAs from {json_file}")
        except FileNotFoundError:
            print(f"⚠ Warning: {json_path} not found, skipping...")
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing {json_file}: {e}")
        except Exception as e:
            print(f"❌ Unexpected error loading {json_file}: {e}")
    
    print(f"📦 Total LoRAs loaded: {len(lora_models)}")
    return lora_models

# ===================================================================
# DEFINISI IMAGE & VOLUME
# ===================================================================
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]", "torch", "diffusers", "transformers",
        "accelerate", "safetensors", "Pillow", "requests", "huggingface_hub", 
        "opencv-python-headless", "controlnet_aux"
    )
)

model_volume = modal.Volume.from_name("civitai-models", create_if_missing=True)
lora_volume = modal.Volume.from_name("civitai-loras-collection-vol", create_if_missing=True)
controlnet_volume = modal.Volume.from_name("controlnet-sdxl-collection-vol", create_if_missing=True)
json_volume = modal.Volume.from_name("json-config-vol", create_if_missing=True)

# ===================================================================
# FUNGSI DOWNLOAD
# ===================================================================
@app.function(image=image, volumes={JSON_DATA_DIR: json_volume}, timeout=600)
def download_json_files():
    """Download JSON configuration files from GitHub to volume"""
    import requests
    from pathlib import Path
    
    github_urls = {
        "01.json": "https://raw.githubusercontent.com/oktetod/Terminal/main/01.json",
        "02.json": "https://raw.githubusercontent.com/oktetod/Terminal/main/02.json",
        "03.json": "https://raw.githubusercontent.com/oktetod/Terminal/main/03.json",
        "04.json": "https://raw.githubusercontent.com/oktetod/Terminal/main/04.json",
        "05.json": "https://raw.githubusercontent.com/oktetod/Terminal/main/05.json",
        "06.json": "https://raw.githubusercontent.com/oktetod/Terminal/main/06.json",
        "07.json": "https://raw.githubusercontent.com/oktetod/Terminal/main/07.json",
    }
    
    Path(JSON_DATA_DIR).mkdir(parents=True, exist_ok=True)
    
    for filename, url in github_urls.items():
        dest = Path(JSON_DATA_DIR) / filename
        try:
            print(f"📥 Downloading {filename}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(dest, 'wb') as f:
                f.write(response.content)
            
            print(f"✓ Downloaded {filename}")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
    
    json_volume.commit()
    print("✓ All JSON files downloaded to volume")

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
            for chunk in response.iter_content(chunk_size=8192): 
                f.write(chunk)
        model_volume.commit()
        print(f"Model downloaded to {model_path}")
    else:
        print(f"Model already exists at {model_path}")

@app.function(
    image=image, 
    volumes={LORA_DIR: lora_volume, JSON_DATA_DIR: json_volume}, 
    timeout=7200
)
def download_loras():
    """
    Download all LoRA models from JSON configuration files
    Supports batch downloading with progress tracking and error handling
    """
    import requests
    from pathlib import Path
    import json
    
    # Load LoRA configurations from JSON files
    print("="*70)
    print("📋 LOADING LORA CONFIGURATIONS FROM JSON FILES")
    print("="*70)
    
    lora_models = load_lora_models_from_json(JSON_DATA_DIR)
    
    if not lora_models:
        print("❌ No LoRA models found in JSON files!")
        print("⚠️  Make sure JSON files are in the same directory as modal_app.py")
        return
    
    print(f"\n📦 Total LoRAs to process: {len(lora_models)}")
    print("="*70)
    
    # Download LoRAs
    failed_downloads = []
    successful_downloads = []
    skipped_existing = []
    
    total_loras = len(lora_models)
    current_index = 0
    
    for name, data in lora_models.items():
        current_index += 1
        
        # Skip if no URL or filename
        if not isinstance(data, dict) or 'url' not in data or 'filename' not in data:
            print(f"[{current_index}/{total_loras}] ⚠ Invalid data format for: {name}")
            continue
            
        lora_path = Path(LORA_DIR) / data["filename"]
        
        if lora_path.exists():
            print(f"[{current_index}/{total_loras}] ✓ LoRA sudah ada: {name}")
            skipped_existing.append(name)
            continue
        
        try:
            print(f"[{current_index}/{total_loras}] 📥 Downloading LoRA: {name}...")
            print(f"    URL: {data['url'][:60]}...")
            
            response = requests.get(data["url"], stream=True, timeout=120)
            response.raise_for_status()
            
            # Create directory if not exists
            lora_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(lora_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192): 
                    f.write(chunk)
            
            lora_volume.commit()
            print(f"[{current_index}/{total_loras}] ✅ LoRA berhasil diunduh: {name}")
            successful_downloads.append(name)
            
        except requests.exceptions.RequestException as e:
            print(f"[{current_index}/{total_loras}] ❌ GAGAL download LoRA: {name}")
            print(f"    Error: {str(e)}")
            failed_downloads.append({
                "index": current_index,
                "name": name,
                "filename": data["filename"],
                "error": str(e)
            })
            continue
        except Exception as e:
            print(f"[{current_index}/{total_loras}] ❌ ERROR tidak terduga: {name}")
            print(f"    Error: {str(e)}")
            failed_downloads.append({
                "index": current_index,
                "name": name,
                "filename": data["filename"],
                "error": str(e)
            })
            continue
    
    # Laporan akhir
    print("\n" + "="*70)
    print("📊 LAPORAN DOWNLOAD LORA")
    print("="*70)
    print(f"✅ Berhasil diunduh: {len(successful_downloads)}")
    print(f"⏭️  Sudah ada (diskip): {len(skipped_existing)}")
    print(f"❌ Gagal diunduh: {len(failed_downloads)}")
    print(f"📦 Total LoRA: {total_loras}")
    
    if failed_downloads:
        print("\n" + "⚠️ " * 35)
        print("DAFTAR LORA YANG GAGAL DIUNDUH:")
        print("="*70)
        for fail in failed_downloads:
            print(f"\n#{fail['index']} - {fail['name']}")
            print(f"   File: {fail['filename']}")
            print(f"   Error: {fail['error']}")
        print("\n" + "⚠️ " * 35)
    else:
        print("\n🎉 Semua LoRA berhasil diproses!")
    
    print("="*70)

@app.function(image=image, volumes={CONTROLNET_DIR: controlnet_volume}, timeout=3600)
def download_controlnet_models():
    from huggingface_hub import snapshot_download
    from pathlib import Path
    
    for name, repo_id in CONTROLNET_MODELS.items():
        model_dir = Path(CONTROLNET_DIR) / name
        if not model_dir.exists():
            print(f"Downloading ControlNet model: {name} from {repo_id}...")
            snapshot_download(repo_id=repo_id, local_dir=str(model_dir), local_dir_use_symlinks=False)
            controlnet_volume.commit()
            print(f"ControlNet {name} downloaded.")
    print("All ControlNet models download check complete.")


# ===================================================================
# KELAS INFERENCE
# ===================================================================
@app.cls(
    image=image,
    gpu="L4",
    volumes={MODEL_DIR: model_volume, LORA_DIR: lora_volume, CONTROLNET_DIR: controlnet_volume, JSON_DATA_DIR: json_volume},
    scaledown_window=200
)
class ModelInference:
    @modal.enter()
    def load_model(self):
        from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, EulerDiscreteScheduler
        import torch
        
        print("Loading SDXL model...")
        model_path = f"{MODEL_DIR}/model.safetensors"
        
        self.txt2img_pipe = StableDiffusionXLPipeline.from_single_file(
            model_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )
        self.txt2img_pipe.scheduler = EulerDiscreteScheduler.from_config(self.txt2img_pipe.scheduler.config)
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
        
        # Load LoRA models list from JSON
        self.lora_models = load_lora_models_from_json(JSON_DATA_DIR)
        
        print("✓ SDXL Model loaded successfully! Uncensored mode active.")
        print(f"✓ {len(self.lora_models)} LoRAs available")

    def _apply_lora(self, pipe, lora_name: str, lora_scale: float):
        """Apply LoRA weights to pipeline"""
        try:
            pipe.unload_lora_weights()
        except:
            pass
            
        if lora_name and lora_name in self.lora_models:
            print(f"Applying LoRA: {lora_name} with scale {lora_scale}")
            lora_info = self.lora_models[lora_name]
            lora_path = f"{LORA_DIR}/{lora_info['filename']}"
            pipe.load_lora_weights(lora_path, adapter_name=lora_name)
            pipe.fuse_lora(lora_scale=lora_scale, adapter_names=[lora_name])

    def _preprocess_control_image(self, image, controlnet_type: str):
        """Preprocess control image based on type"""
        from controlnet_aux import OpenposeDetector
        from PIL import Image
        import numpy as np
        import cv2
        
        if controlnet_type == "openpose":
            processor = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            return processor(image)
        elif controlnet_type == "canny":
            image_np = np.array(image)
            low_threshold = 100
            high_threshold = 200
            canny_image = cv2.Canny(image_np, low_threshold, high_threshold)
            canny_image = canny_image[:, :, None]
            canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
            return Image.fromarray(canny_image)
        elif controlnet_type == "depth":
            from transformers import pipeline
            depth_estimator = pipeline('depth-estimation')
            depth = depth_estimator(image)['depth']
            return depth
        else:
            return image

    @modal.method()
    def text_to_image(self, prompt: str, lora_name: str = None, lora_scale: float = 0.8, **kwargs):
        """Generate image from text prompt"""
        import torch
        
        self._apply_lora(self.txt2img_pipe, lora_name, lora_scale)
        
        enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}" if kwargs.get("enhance_prompt", True) else prompt
        negative_prompt = kwargs.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT
        
        generator = None
        if kwargs.get("seed", -1) != -1:
            generator = torch.Generator(device="cuda").manual_seed(kwargs.get("seed", -1))
        
        image = self.txt2img_pipe(
            prompt=enhanced_prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=kwargs.get("num_steps", 25),
            guidance_scale=kwargs.get("guidance_scale", 7.5),
            width=kwargs.get("width", 1024), 
            height=kwargs.get("height", 1024),
            generator=generator
        ).images[0]
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "image": img_str, 
            "prompt": enhanced_prompt, 
            "original_prompt": prompt,
            "negative_prompt": negative_prompt, 
            "seed": kwargs.get("seed", -1), 
            "uncensored": True
        }

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
        width: int = 1024,
        height: int = 1024
    ):
        """Edit image with prompt"""
        from PIL import Image
        import torch
        
        self._apply_lora(self.img2img_pipe, lora_name, lora_scale)
        
        enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}" if enhance_prompt else prompt
        
        if not negative_prompt or not str(negative_prompt).strip():
            negative_prompt = DEFAULT_NEGATIVE_PROMPT
        
        print(f"Image-to-Image: {enhanced_prompt[:100]}...")
        
        init_image_bytes = base64.b64decode(init_image_b64)
        init_image = Image.open(io.BytesIO(init_image_bytes)).convert("RGB")
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
            "output_width": width,
            "output_height": height
        }

    @modal.method()
    def generate_with_controlnet(
        self, 
        prompt: str, 
        control_image_b64: str, 
        controlnet_type: str, 
        lora_name: str = None, 
        lora_scale: float = 0.8, 
        **kwargs
    ):
        """Generate image with ControlNet guidance"""
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
        from PIL import Image
        import torch
        
        if controlnet_type not in CONTROLNET_MODELS:
            raise ValueError(f"Invalid controlnet_type. Supported: {list(CONTROLNET_MODELS.keys())}")

        controlnet = ControlNetModel.from_pretrained(
            str(Path(CONTROLNET_DIR) / controlnet_type), 
            torch_dtype=torch.float16
        )
        
        control_pipe = StableDiffusionXLControlNetPipeline(
            vae=self.txt2img_pipe.vae, 
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2, 
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2, 
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler, 
            controlnet=controlnet,
        )
        control_pipe.to("cuda")

        self._apply_lora(control_pipe, lora_name, lora_scale)

        control_image_bytes = base64.b64decode(control_image_b64)
        control_image = Image.open(io.BytesIO(control_image_bytes)).convert("RGB")
        
        # Preprocess control image based on type
        control_image = self._preprocess_control_image(control_image, controlnet_type)
        
        enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}"
        negative_prompt = kwargs.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT
        
        generator = None
        if kwargs.get("seed", -1) != -1:
            generator = torch.Generator(device="cuda").manual_seed(kwargs.get("seed", -1))

        image = control_pipe(
            enhanced_prompt, 
            negative_prompt=negative_prompt, 
            image=control_image,
            num_inference_steps=kwargs.get("num_steps", 30),
            guidance_scale=kwargs.get("guidance_scale", 7.5),
            controlnet_conditioning_scale=float(kwargs.get("controlnet_scale", 0.8)),
            generator=generator,
        ).images[0]
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "image": img_str, 
            "prompt": enhanced_prompt, 
            "original_prompt": prompt,
            "negative_prompt": negative_prompt, 
            "seed": kwargs.get("seed", -1), 
            "controlnet_type": controlnet_type,
            "uncensored": True
        }


# ===================================================================
# ENDPOINT API
# ===================================================================
@app.function(
    image=image, 
    secrets=[modal.Secret.from_name("custom-secret")],
    volumes={JSON_DATA_DIR: json_volume}
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    import os
    
    web_app = FastAPI()
    
    # Load LoRA models list for API response
    lora_models = load_lora_models_from_json(JSON_DATA_DIR)
    
    @web_app.get("/")
    async def root():
        return {
            "service": "CivitAI Model API - Uncensored (SDXL)",
            "version": "3.3",
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
                "✓ Image-to-Image (SDXL)",
                "✓ ControlNet (SDXL)",
                "✓ Multi-LoRA support",
                "✓ GPU L4",
                "✓ Auto quality enhancement",
                "✓ Default negative prompts for best results"
            ],
            "default_prompts": {
                "positive_suffix": DEFAULT_POSITIVE_PROMPT_SUFFIX,
                "negative": DEFAULT_NEGATIVE_PROMPT
            },
            "total_loras": len(lora_models),
            "available_loras": list(lora_models.keys())[:20],
            "available_controlnets": list(CONTROLNET_MODELS.keys())
        }

    @web_app.get("/health")
    async def health_check():
        return {
            "status": "healthy", 
            "service": "civitai-api-fastapi",
            "mode": "uncensored-sdxl",
            "gpu": "L4",
            "total_loras": len(lora_models)
        }
    
    @web_app.get("/loras")
    async def list_loras():
        """Get list of all available LoRA models"""
        return {
            "total": len(lora_models),
            "loras": list(lora_models.keys())
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
            
            lora_name = data.get("lora_name")
            lora_scale = data.get("lora_scale", 0.8)
            
            model = ModelInference()
            result = model.text_to_image.remote(
                prompt=prompt, 
                lora_name=lora_name, 
                lora_scale=lora_scale, 
                **data
            )
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
                init_image_b64=init_image, 
                prompt=prompt, 
                lora_name=lora_name, 
                lora_scale=lora_scale,
                negative_prompt=data.get("negative_prompt", ""),
                num_steps=data.get("num_steps", 25),
                guidance_scale=data.get("guidance_scale", 7.5),
                strength=data.get("strength", 0.75),
                seed=data.get("seed", -1),
                enhance_prompt=data.get("enhance_prompt", True),
                width=data.get("width", 1024),
                height=data.get("height", 1024)
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
                prompt=prompt, 
                control_image_b64=control_image, 
                controlnet_type=controlnet_type,
                lora_name=lora_name, 
                lora_scale=lora_scale, 
                **data
            )
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return web_app

@app.local_entrypoint()
def main():
    """Local test entrypoint"""
    print("Available commands:")
    print("  modal run modal_app.py::download_json_files")
    print("  modal run modal_app.py::download_model")
    print("  modal run modal_app.py::download_loras")
    print("  modal run modal_app.py::download_controlnet_models")
    print("  modal deploy modal_app.py")
