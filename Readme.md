# Panduan Deploy & Penggunaan API Model Juggernaut XL di Modal.com

Panduan ini mencakup semua langkah yang diperlukan untuk men-deploy model AI dari CivitAI ke Modal.com dan cara memanggil API-nya menggunakan `curl` dan Python.

## 1. Setup Awal

### Struktur File Proyek
Pastikan proyek Anda memiliki struktur file berikut:
```
your-repo/
├── modal_app.py      # File utama untuk deploy ke Modal
├── .env              # File untuk menyimpan environment variables
└── README.md
```

### Environment Variables (`.env`)
Buat file `.env` di direktori proyek Anda. File ini **tidak boleh** di-commit ke Git.
```
# Kunci API ini bebas Anda tentukan, gunakan untuk mengamankan endpoint Anda.
# Harus sama dengan yang Anda simpan di Modal Secrets.
API_KEY="your-super-secret-key-12345"
```

## 2. Setup Modal.com

### a. Install Modal CLI & Login
Jika belum terinstall, buka terminal dan jalankan:
```bash
pip install modal
```
Selanjutnya, hubungkan CLI dengan akun Modal Anda:
```bash
modal token new
```
Ini akan membuka browser untuk proses autentikasi.

### b. Buat Secret di Modal
Aplikasi kita memerlukan sebuah *secret* untuk menyimpan `API_KEY`.

1.  Buka halaman Secrets di dashboard Modal: [https://modal.com/secrets](https://modal.com/secrets)
2.  Klik **"Create"**.
3.  **Secret name**: `custom-secret` (Nama ini harus sama persis dengan yang ada di `modal_app.py`).
4.  Tambahkan satu variabel:
    * **Key**: `API_KEY`
    * **Value**: `your-super-secret-key-12345` (Isi dengan kunci rahasia yang sama dengan di file `.env` Anda).
5.  Klik **"Create"**.

## 3. Deploy Aplikasi ke Modal

### a. Download Model (Hanya Sekali)
Perintah ini akan men-download model Juggernaut XL dan menyimpannya di Modal Volume agar tidak perlu di-download ulang setiap kali aplikasi berjalan.
```bash
modal run modal_app.py::download_model
```

### b. Deploy Aplikasi Utama
Perintah ini akan men-deploy kode `modal_app.py` Anda dan memberikan sebuah URL publik.
```bash
modal deploy modal_app.py
```
Setelah selesai, Anda akan mendapatkan URL endpoint. Catat URL ini. Contoh:
`https://your-workspace--civitai-api-fastapi-fastapi-app.modal.run`

## 4. Contoh Pemanggilan API

Ganti `YOUR_MODAL_URL` dengan URL yang Anda dapatkan setelah deploy dan `YOUR_API_KEY` dengan kunci rahasia Anda.

### a. Text-to-Image (`curl`)
Endpoint ini menggunakan kunci JSON bernama `"api_key"`.
```bash
curl -X POST https://YOUR_MODAL_URL/text2img \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a majestic lion with a crown of stars, cinematic lighting",
    "api_key": "YOUR_API_KEY",
    "num_steps": 25,
    "width": 1024,
    "height": 1024,
    "negative_prompt": "cartoon, drawing, ugly"
  }'
```

### b. Image-to-Image (`curl`)
**Perhatian:** Endpoint ini, sesuai kode `modal_app.py` Anda, menggunakan kunci JSON bernama `"custom-secret"` untuk otentikasi.
```bash
curl -X POST https://YOUR_MODAL_URL/img2img \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "make this person a cyborg, futuristic, neon lights",
    "custom-secret": "YOUR_API_KEY",
    "init_image": "iVBORw0KGgoAAAANSUhEUgA...",
    "strength": 0.8
  }'
```
> **Catatan:** `iVBORw0KGgoAAAANSUhEUgA...` adalah contoh string base64 yang sangat panjang dari sebuah gambar.

## 5. Contoh Pemanggilan API dengan Python

Ini adalah cara yang lebih praktis untuk memanggil API Anda dari aplikasi lain.

### Helper Function (untuk Image-to-Image)
Simpan fungsi ini untuk mengubah file gambar menjadi string base64.
```python
import base64

def encode_image_to_base64(filepath):
    """Membaca file gambar dan meng-encode-nya ke base64."""
    with open(filepath, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
```

### a. Contoh Kode: Text-to-Image
```python
import requests
import base64

MODAL_URL = "https://YOUR_MODAL_URL/text2img"  # Ganti dengan URL Anda
API_KEY = "YOUR_API_KEY"                      # Ganti dengan kunci Anda

payload = {
    "prompt": "a majestic lion with a crown of stars, cinematic lighting",
    "api_key": API_KEY,
    "num_steps": 25,
    "negative_prompt": "cartoon, ugly, drawing"
}

try:
    response = requests.post(MODAL_URL, json=payload, timeout=180)
    response.raise_for_status()

    result = response.json()
    image_b64 = result["image"]

    # Simpan gambar yang diterima
    image_data = base64.b64decode(image_b64)
    with open("output_image.png", "wb") as f:
        f.write(image_data)
    
    print("Gambar berhasil dibuat dan disimpan sebagai output_image.png")

except requests.exceptions.RequestException as e:
    print(f"Error saat memanggil API: {e}")
```

### b. Contoh Kode: Image-to-Image
```python
import requests
import base64

# (Pastikan fungsi encode_image_to_base64 sudah ada di sini)

MODAL_URL = "https://YOUR_MODAL_URL/img2img"  # Ganti dengan URL Anda
API_KEY = "YOUR_API_KEY"                      # Ganti dengan kunci Anda

# Path ke gambar yang ingin Anda edit
input_image_path = "path/to/your/image.png"

payload = {
    "prompt": "make this person a cyborg, futuristic, neon lights",
    "custom-secret": API_KEY, # Perhatikan: key di sini adalah 'custom-secret'
    "init_image": encode_image_to_base64(input_image_path),
    "strength": 0.75, # Seberapa kuat efek prompt (0.0 - 1.0)
    "num_steps": 30
}

try:
    response = requests.post(MODAL_URL, json=payload, timeout=180)
    response.raise_for_status()

    result = response.json()
    image_b64 = result["image"]

    # Simpan gambar hasil editan
    image_data = base64.b64decode(image_b64)
    with open("edited_image.png", "wb") as f:
        f.write(image_data)
    
    print("Gambar berhasil diedit dan disimpan sebagai edited_image.png")

except requests.exceptions.RequestException as e:
    print(f"Error saat memanggil API: {e}")
```

## 6. Monitoring dan Tips

-   **Lihat Logs Aplikasi:**
    ```bash
    modal app logs civitai-api-fastapi
    ```
-   **Optimasi Biaya:** Gunakan `container_idle_timeout` (sudah diatur di `modal_app.py`) untuk mematikan container secara otomatis saat tidak ada permintaan.
-   **GPU:** `T4` adalah pilihan yang seimbang antara biaya dan kecepatan. Ganti ke `A10G` atau `A100` di `modal_app.py` jika butuh kecepatan lebih tinggi dengan biaya lebih mahal.
