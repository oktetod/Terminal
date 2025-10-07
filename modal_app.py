"""
Deploy Model CivitAI ke Modal.com dengan FastAPI
Format baru - compatible dengan Modal terbaru
"""

import modal
from pathlib import Path

# Inisialisasi Modal app
app = modal.App("civitai-api-fastapi")

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
    container_idle_timeout=300
)
class ModelInference:
    @modal.enter()
    def load_model(self):
        """Load model saat container start"""
        from diffusers import StableDiffusionPipeline
        import torch
        
        print("Loading model...")
        model_path = f"{MODEL_DIR}/model.safetensors"
        
        self.pipe = StableDiffusionPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        self.pipe.to("cuda")
        print("Model loaded successfully!")
    
    @modal.method()
    def generate(self, prompt: str, negative_prompt: str = "", num_steps: int = 20, guidance_scale: float = 7.5):
        """Generate image dari prompt"""
        import io
        import base64
        from PIL import Image
        
        print(f"Generating image for prompt: {prompt}")
        
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale
        ).images[0]
        
        # Convert ke base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {"image": img_str, "prompt": prompt}


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

    @web_app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "service": "civitai-model-api"}

    @web_app.post("/generate")
    async def generate_image_endpoint(request: Request):
        """
        Generate image endpoint
        
        Body:
        {
            "prompt": "your prompt",
            "negative_prompt": "optional",
            "num_steps": 20,
            "guidance_scale": 7.5,
            "api_key": "your-api-key"
        }
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
            
            # Generate image
            model = ModelInference()
            result = model.generate.remote(
                prompt=prompt,
                negative_prompt=data.get("negative_prompt", ""),
                num_steps=data.get("num_steps", 20),
                guidance_scale=data.get("guidance_scale", 7.5)
            )
            
            return JSONResponse(content=result)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return web_app


# Local entry untuk testing
@app.local_entrypoint()
def main():
    """Test locally"""
    model = ModelInference()
    result = model.generate.remote(
        prompt="beautiful landscape with mountains",
        num_steps=20
    )
    print("Success!")
    print(f"Prompt: {result['prompt']}")
