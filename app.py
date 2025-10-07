# app.py

import io
from pathlib import Path

from modal import Image, Stub, asgi_app

# --- KONFIGURASI ---
MODEL_URL = "https://civitai.com/api/download/models/1759168"
MODEL_FILENAME = "juggernautXL_v8Rundiffusion.safetensors"
MODEL_DIR = "/model_storage"

# --- FUNGSI UNTUK MENGUNDUH MODEL (HANYA DIJALANKAN SEKALI SAAT SETUP) ---
def download_model():
    import requests
    model_path = Path(f"{MODEL_DIR}/{MODEL_FILENAME}")
    if not model_path.exists():
        print(f"Model '{MODEL_FILENAME}' tidak ditemukan, mengunduh...")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192 * 4):
                    f.write(chunk)
        print("Unduhan model selesai.")

# --- DEFINISI LINGKUNGAN APLIKASI ---
# Di sinilah kita menginstal semua library yang dibutuhkan
# dan menjalankan fungsi download_model saat pertama kali image dibuat.
stub_image = (
    Image.debian_slim()
    .pip_install(
        "torch", "diffusers[torch]", "transformers",
        "accelerate", "safetensors", "requests", "fastapi"
    )
    .run_function(download_model, timeout=1800) # Beri waktu 30 menit untuk download
)

# --- DEFINISI APLIKASI MODAL ---
stub = Stub("juggernaut-telegram-api", image=stub_image)

# Volume penyimpanan persisten agar model tidak hilang
volume = stub.NetworkFileSystem.persisted("juggernaut-model-volume")

# --- KELAS UNTUK MENJALANKAN MODEL AI ---
@stub.cls(
    gpu="T4",
    network_file_systems={MODEL_DIR: volume},
    container_idle_timeout=300, # Jaga GPU tetap "panas" selama 5 menit
)
class JuggernautXL:
    def __enter__(self):
        import torch
        from diffusers import StableDiffusionXLPipeline

        print("Memuat model ke VRAM...")
        model_path = f"{MODEL_DIR}/{MODEL_FILENAME}"

        self.pipe = StableDiffusionXLPipeline.from_single_file(
            model_path, torch_dtype=torch.float16
        ).to("cuda")
        print("Model berhasil dimuat.")

    @stub.method()
    def generate(self, prompt: str):
        image_result = self.pipe(prompt=prompt, num_inference_steps=25).images[0]

        buffer = io.BytesIO()
        image_result.save(buffer, format="PNG")
        return buffer.getvalue()

# --- API ENDPOINT YANG AKAN DIAKSES PUBLIK ---
@stub.function()
@asgi_app()
def api_endpoint():
    from fastapi import FastAPI, Request
    from fastapi.responses import Response

    app = FastAPI()

    @app.post("/generate")
    async def generate_image(request: Request):
        data = await request.json()
        prompt = data.get("prompt")
        if not prompt:
            return Response(content='{"error": "Prompt tidak ditemukan"}', status_code=400)

        # Panggil fungsi generate dari kelas JuggernautXL
        image_bytes = JuggernautXL().generate.remote(prompt)
        return Response(content=image_bytes, media_type="image/png")

    return app
