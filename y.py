import modal
import subprocess
from pathlib import Path

# ==============================================================================
# BAGIAN 1: PENGATURAN LINGKungan
# ==============================================================================
image = modal.Image.debian_slim(python_version="3.10").apt_install(
    "git"
).pip_install(
    "torch", "torchvision", "torchaudio",
    extra_options="--extra-index-url https://download.pytorch.org/whl/cu118"
).pip_install(
    "git+https://github.com/kohya-ss/sd-scripts.git"
)

base_model_storage = modal.Volume.from_name("civitai-models")
loras_storage = modal.Volume.from_name("civitai-loras-collection-vol")
app = modal.App("sdxl-lora-merge-debug-v2", image=image) # Nama diubah untuk hindari cache
BASE_MODEL_DIR = Path("/base_model")
LORAS_DIR = Path("/loras")

# ==============================================================================
# BAGIAN 2: FUNGSI-FUNGSI MODAL
# ==============================================================================

@app.function(
    volumes={LORAS_DIR: loras_storage},
    timeout=300,
)
def get_all_lora_filenames():
    """Fungsi ini membaca dan mengembalikan semua nama file dari Volume LoRA."""
    print(f"🔍 Membaca semua file dari Volume '{loras_storage.object_id}'...")
    lora_files = [f.name for f in LORAS_DIR.iterdir() if f.is_file() and f.name.endswith(('.safetensors', '.ckpt'))]
    print(f"✅ Ditemukan {len(lora_files)} file LoRA.")
    return lora_files

@app.function(
    volumes={BASE_MODEL_DIR: base_model_storage, LORAS_DIR: loras_storage},
    timeout=1200,
    gpu="any"
)
def merge_loras_on_modal(base_model_path: str, output_model_path: str, lora_files: list[str], lora_ratios: list[float]):
    """
    FUNGSI DEBUG: Fungsi ini hanya untuk menemukan path file yang benar.
    """
    import subprocess
    print("--- 🕵️  Memulai mode debug: Mencari file 'sdxl_merge_lora.py'... ---")
    
    # Perintah untuk mencari file di seluruh sistem di dalam container
    cmd = ["find", "/", "-name", "sdxl_merge_lora.py", "-type", "f"]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("--- 📜 Hasil Pencarian ---")
    if result.stdout:
        print("✅ File ditemukan di path berikut:")
        # Output ini adalah yang kita butuhkan!
        print(result.stdout)
    else:
        print("❌ File 'sdxl_merge_lora.py' tidak ditemukan di mana pun.")
    
    if result.stderr:
        print("--- ⚠️ Pesan Error dari Perintah 'find' ---")
        print(result.stderr)
        
    print("--- 🛑 Selesai debug. Proses dihentikan dengan sengaja. ---")
    print("--- 👉 Salin path yang muncul di atas (jika ada) dan berikan ke saya. ---")
    
    # Menghentikan proses dengan sengaja setelah debug selesai
    raise RuntimeError("DEBUG SELESAI: Ini bukan error, hanya cara untuk menghentikan eksekusi.")

# ==============================================================================
# BAGIAN 3: FUNGSI UTAMA
# ==============================================================================
@app.local_entrypoint()
def main():
    print("🚀 Memulai skrip 'Gabungkan Semua'...")
    
    loras_to_merge = get_all_lora_filenames.remote()

    if not loras_to_merge:
        print("Tidak ada file LoRA yang ditemukan. Proses dihentikan.")
        return

    uniform_ratio = 0.1
    ratios_for_loras = [uniform_ratio] * len(loras_to_merge)
    
    base_model = "model.safetensors"
    output_model = "merged_models/juggernaut_ALL_IN_ONE.safetensors"
    
    print(f"Akan menggabungkan {len(loras_to_merge)} LoRA ke model '{base_model}'...")
    
    merge_loras_on_modal.remote(
        base_model_path=base_model,
        output_model_path=output_model,
        lora_files=loras_to_merge,
        # --- PERBAIKAN TYPO DI SINI ---
        lora_ratios=ratios_for_loras,
    )
