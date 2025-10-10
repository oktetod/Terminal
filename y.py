import modal
import subprocess
from pathlib import Path

# ==============================================================================
# BAGIAN 1: PENGATURAN LINGKUNGAN (Tidak perlu diubah)
# ==============================================================================
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch", "torchvision", "torchaudio", "--extra-index-url", "https://download.pytorch.org/whl/cu118"
).pip_install(
    "git+https://github.com/kohya-ss/sd-scripts.git"
)
base_model_storage = modal.Volume.from_name("civitai-model")
loras_storage = modal.Volume.from_name("civitai-loras-collection-vol")
app = modal.App("sdxl-lora-merge-all", image=image)
BASE_MODEL_DIR = Path("/base_model")
LORAS_DIR = Path("/loras")

# ==============================================================================
# BAGIAN 2: FUNGSI-FUNGSI MODAL (Tidak perlu diubah)
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
    timeout=1200, # Waktu ditambah karena prosesnya berat
)
def merge_loras_on_modal(base_model_path: str, output_model_path: str, lora_files: list[str], lora_ratios: list[float]):
    """Fungsi ini menjalankan proses merge untuk semua LoRA yang ditemukan."""
    base_model_full_path = BASE_MODEL_DIR / base_model_path
    output_model_full_path = BASE_MODEL_DIR / output_model_path
    output_model_full_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "sdxl_merge_lora.py", "--save_precision", "fp16",
        "--sd_model", str(base_model_full_path), "--save_to", str(output_model_full_path),
    ]
    lora_full_paths = [str(LORAS_DIR / lora) for lora in lora_files]
    cmd.extend(["--models", *lora_full_paths])
    ratio_strs = [str(r) for r in lora_ratios]
    cmd.extend(["--ratios", *ratio_strs])
    print(f"--- 🌀 Memulai proses merge untuk {len(lora_files)} LoRA... ---")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("--- ❌ Proses merge gagal! ---"); print("STDERR:", result.stderr); raise RuntimeError("Proses merge gagal.")
    else:
        print(f"--- ✅ Proses merge selesai! Model 'super-merge' disimpan di Volume 'civitai-model' pada path: {output_model_path}")

# ==============================================================================
# BAGIAN 3: FUNGSI UTAMA (Tidak perlu diubah)
# ==============================================================================
@app.local_entrypoint()
def main():
    print("🚀 Memulai skrip 'Gabungkan Semua'...")
    
    # 1. Mengambil semua nama file LoRA secara otomatis
    loras_to_merge = get_all_lora_filenames.remote()

    if not loras_to_merge:
        print("Tidak ada file LoRA yang ditemukan. Proses dihentikan.")
        return

    # 2. Menetapkan bobot (ratio) yang sama untuk semua LoRA
    # Nilai kecil untuk mencoba mengurangi "kerusakan" pada model
    uniform_ratio = 0.1
    ratios_for_loras = [uniform_ratio] * len(loras_to_merge)
    
    # 3. Menjalankan proses merge
    base_model = "model.safetensors"
    output_model = "merged_models/juggernaut_ALL_IN_ONE.safetensors"
    
    merge_loras_on_modal.remote(
        base_model_path=base_model,
        output_model_path=output_model,
        lora_files=loras_to_merge,
        lora_ratios=ratios_for_loras,
    )
