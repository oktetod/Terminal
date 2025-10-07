"""
Deploy Model CivitAI ke Modal.com dengan API Endpoint
Untuk diintegrasikan dengan Telegram Bot
"""

import modal
from pathlib import Path

# Inisialisasi Modal app
app = modal.App("civitai-model-api")

# Definisikan image dengan dependencies yang diperlukan
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
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
    gpu="T4",  # GPU T4 lebih murah (~$0.60/hour)
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
        
        # Sesuaikan dengan tipe model Anda (Stable Diffusion, LoRA, dll)
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
        
        # Convert ke base64 untuk transfer
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {"image": img_str, "prompt": prompt}


# Web endpoint untuk API
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("api-secret")]  # Secret untuk API key
)
@modal.web_endpoint(method="POST")
def generate_image(data: dict):
    """
    API endpoint untuk generate image
    
    Request body:
    {
        "prompt": "your prompt here",
        "negative_prompt": "optional negative prompt",
        "num_steps": 20,
        "guidance_scale": 7.5,
        "api_key": "your-secret-api-key"
    }
    """
    import os
    
    # Validasi API key
    api_key = data.get("api_key")
    if api_key != os.environ.get("API_KEY"):
        return {"error": "Invalid API key"}, 401
    
    prompt = data.get("prompt")
    if not prompt:
        return {"error": "Prompt is required"}, 400
    
    # Generate image
    model = ModelInference()
    result = model.generate.remote(
        prompt=prompt,
        negative_prompt=data.get("negative_prompt", ""),
        num_steps=data.get("num_steps", 20),
        guidance_scale=data.get("guidance_scale", 7.5)
    )
    
    return result


# Endpoint health check
@app.function()
@modal.web_endpoint(method="GET")
def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "civitai-model-api"}


# Local entry untuk testing
@app.local_entrypoint()
def main():
    """Test the deployment locally"""
    # Download model (uncomment jika belum download)
    # download_model.remote()
    
    # Test inference
    model = ModelInference()
    result = model.generate.remote(
        prompt="a beautiful landscape with mountains and lake, sunset",
        num_steps=20
    )
    print("Generation successful!")
    print(f"Prompt: {result['prompt']}")
