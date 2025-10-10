import modal
import subprocess
from pathlib import Path

# ==============================================================================
# BAGIAN 1: PENGATURAN LINGKUNGAN
# ==============================================================================
image = modal.Image.debian_slim(python_version="3.10").apt_install(
    "git",
    "libgl1-mesa-glx",
    "libglib2.0-0"
).pip_install(
    "numpy<2.0.0"
).pip_install(
    "torch", "torchvision", "torchaudio",
    extra_options="--extra-index-url https://download.pytorch.org/whl/cu118"
).pip_install(
    "accelerate",
    "safetensors",
    "transformers"
).run_commands(
    "cd /root && git clone https://github.com/kohya-ss/sd-scripts.git",
    "cd /root/sd-scripts && pip install -r requirements.txt"
)

base_model_storage = modal.Volume.from_name("civitai-models")
loras_storage = modal.Volume.from_name("civitai-loras-collection-vol")
app = modal.App("sdxl-lora-merge-v4", image=image)
BASE_MODEL_DIR = Path("/base_model")
LORAS_DIR = Path("/loras")
SD_SCRIPTS_DIR = Path("/root/sd-scripts")

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
    Menggabungkan multiple LoRA ke base model menggunakan sd-scripts
    """
    import os
    
    print(f"🔧 Memulai proses merge {len(lora_files)} LoRA...")
    
    # Path lengkap untuk base model dan output
    full_base_path = BASE_MODEL_DIR / base_model_path
    full_output_path = BASE_MODEL_DIR / output_model_path
    
    # Buat direktori output jika belum ada
    full_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Verifikasi base model ada
    if not full_base_path.exists():
        raise FileNotFoundError(f"Base model tidak ditemukan: {full_base_path}")
    
    print(f"✅ Base model ditemukan: {full_base_path}")
    
    # Bangun command untuk merge
    # Path ke script merge_lora.py
    merge_script = SD_SCRIPTS_DIR / "networks" / "sdxl_merge_lora.py"
    
    if not merge_script.exists():
        raise FileNotFoundError(f"Script merge tidak ditemukan: {merge_script}")
    
    cmd = [
        "python",
        str(merge_script),
        "--sd_model", str(full_base_path),
        "--save_to", str(full_output_path),
        "--precision", "fp16",
        "--save_precision", "fp16"
    ]
    
    # Tambahkan semua LoRA dengan ratio-nya
    for lora_file, ratio in zip(lora_files, lora_ratios):
        lora_path = LORAS_DIR / lora_file
        if not lora_path.exists():
            print(f"⚠️ WARNING: LoRA tidak ditemukan: {lora_path}, dilewati.")
            continue
        cmd.extend(["--models", str(lora_path)])
        cmd.extend(["--ratios", str(ratio)])
    
    print(f"🎯 Command yang akan dijalankan:")
    print(" ".join(cmd))
    
    # Jalankan proses merge
    print("⏳ Memulai proses merge... (ini mungkin memakan waktu)")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Tampilkan output
    if result.stdout:
        print("📤 STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("📥 STDERR:")
        print(result.stderr)
    
    # Cek apakah berhasil
    if result.returncode != 0:
        raise RuntimeError(f"❌ Proses merge gagal dengan return code {result.returncode}")
    
    # Verifikasi output file ada
    if not full_output_path.exists():
        raise RuntimeError(f"❌ File output tidak ditemukan: {full_output_path}")
    
    file_size_mb = full_output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Merge berhasil! File output: {full_output_path}")
    print(f"📦 Ukuran file: {file_size_mb:.2f} MB")
    
    # Commit volume untuk menyimpan perubahan
    base_model_storage.commit()
    
    return str(full_output_path)

# ==============================================================================
# BAGIAN 3: FUNGSI UTAMA
# ==============================================================================
@app.local_entrypoint()
def main():
    print("🚀 Memulai skrip 'Gabungkan Semua LoRA'...")
    
    # Dapatkan semua file LoRA
    loras_to_merge = get_all_lora_filenames.remote()

    if not loras_to_merge:
        print("❌ Tidak ada file LoRA yang ditemukan. Proses dihentikan.")
        return

    print(f"📋 Total LoRA yang akan digabungkan: {len(loras_to_merge)}")
    
    # Set uniform ratio untuk semua LoRA
    uniform_ratio = 0.1
    ratios_for_loras = [uniform_ratio] * len(loras_to_merge)
    
    # Config
    base_model = "model.safetensors"
    output_model = "merged_models/juggernaut_ALL_IN_ONE.safetensors"
    
    print(f"🎨 Base model: {base_model}")
    print(f"💾 Output akan disimpan ke: {output_model}")
    print(f"⚖️ Ratio untuk setiap LoRA: {uniform_ratio}")
    
    # Jalankan merge
    result_path = merge_loras_on_modal.remote(
        base_model_path=base_model,
        output_model_path=output_model,
        lora_files=loras_to_merge,
        lora_ratios=ratios_for_loras,
    )
    
    print(f"🎉 Selesai! Model gabungan tersimpan di: {result_path}")
