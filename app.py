# app.py

import io
from pathlib import Path

from modal import App, Image, Volume, asgi_app

# --- Konfigurasi ---
MODEL_URL = "https://civitai.com/api/download/models/1759168"
MODEL_FILENAME = "juggernautXL_v8Rundiffusion.safensors"
MODEL_DIR = "/model_storage"

# --- Definisi Lingkungan ---
# Mendefinisikan semua library yang dibutuhkan.
# Model akan diunduh hanya sekali saat lingkungan ini dibuat pertama kali.
image = (
    Image.debian_slim()
    .pip_install(
        "torch",
        "diffusers[torch]",
        "transformers",
        "accelerate",
        "safetokensors",
        "fastapi",
        "requests", # Menambahkan 'requests' yang lupa diinstal
    )
    .run_function(
        # Argumen 'secret=None' yang menyebabkan error telah dihapus dari sini
        lambda: Path(f"{MODEL_DIR}/{MODEL_FILENAME}").parent.mkdir(parents=True, exist_ok=True)
    )
)

# --- Inisialisasi Aplikasi Modal ---
app = App("juggernaut-xl-api", image=image)

# --- Penyimpanan Persisten ---
# Membuat 'hard disk' di cloud untuk menyimpan file model yang besar
# agar tidak perlu diunduh setiap kali aplikasi dijalankan.
volume = Volume.persisted("juggernaut-model-volume")

# --- Kelas untuk Menjalankan Model AI ---
# Menggunakan @app.cls memastikan model tetap dimuat di VRAM GPU
# untuk respons yang cepat pada panggilan API berikutnya.
@app.cls(gpu="T4", volume={MODEL_DIR: volume}, container_idle_timeout=300)
class JuggernautXL:
    def __init__(self):
        import requests
        import torch
        from diffusers import StableDiffusionXLPipeline

        # Cek dan unduh model jika tidak ada di volume
        model_path = Path(f"{MODEL_DIR}/{MODEL_FILENAME}")
        if not model_path.exists():
            print(f"Mengunduh model ke volume... (ini akan memakan waktu)")
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status()
                with open(model_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192 * 4):
                        f.write(chunk)
            print("Model berhasil diunduh.")

        # Muat model ke VRAM GPU
        print("Memuat pipeline model ke GPU...")
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            str(model_path), torch_dtype=torch.float16
        ).to("cuda")
        print("Model berhasil dimuat.")

    @app.method()
    def generate(self, prompt: str):
        # Jalankan proses generasi gambar
        image_result = self.pipe(prompt=prompt, num_inference_steps=28).images[0]

        # Simpan gambar ke format PNG di memori
        buffer = io.BytesIO()
        image_result.save(buffer, format="PNG")
        return buffer.getvalue()

# --- Endpoint API Publik ---
# Fungsi ini akan menjadi URL publik yang bisa diakses oleh bot Telegram Anda.
@app.function()
@asgi_app()
def api_endpoint():
    from fastapi import FastAPI, Request
    from fastapi.responses import Response

    app_fastapi = FastAPI()

    @app_fastapi.post("/generate")
    async def generate_image(request: Request):
        data = await request.json()
        prompt = data.get("prompt")
        if not prompt:
            return Response(content='{"error": "Prompt tidak ditemukan"}', status_code=400)

        # Panggil fungsi generate dari kelas JuggernautXL
        image_bytes = JuggernautXL().generate.remote(prompt)
        return Response(content=image_bytes, media_type="image/png")

    return app_fastapi
