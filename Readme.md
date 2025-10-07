# Panduan Deploy Model CivitAI ke Modal.com

## 1. Setup GitHub Codespaces

### Buat Repository Baru
```bash
# Di GitHub, buat repository baru untuk project ini
# Clone ke Codespaces atau buat Codespace langsung dari repository
```

### File Structure
```
your-repo/
├── modal_app.py          # File utama (dari artifact sebelumnya)
├── requirements.txt      # Dependencies
├── .env                  # Environment variables (jangan commit!)
└── README.md
```

### requirements.txt
```
modal
torch
diffusers
transformers
accelerate
safetensors
Pillow
requests
python-telegram-bot
```

## 2. Setup Modal.com

### Install Modal CLI
```bash
pip install modal
```

### Login ke Modal
```bash
modal token new
```
Akan membuka browser untuk authentication.

### Buat Secrets di Modal Dashboard

1. Buka https://modal.com/secrets
2. Buat secret bernama `api-secret`:
   ```
   API_KEY=your-super-secret-key-here
   ```
3. (Opsional) Buat `huggingface-secret` jika perlu HF token:
   ```
   HUGGINGFACE_TOKEN=your-hf-token
   ```

## 3. Deploy ke Modal

### Download Model (Sekali Saja)
```bash
modal run modal_app.py::download_model
```

### Deploy App
```bash
modal deploy modal_app.py
```

Setelah deploy, Anda akan mendapat URL endpoint seperti:
```
https://your-workspace--civitai-model-api-generate-image.modal.run
```

### Test API Endpoint
```bash
curl -X POST https://your-workspace--civitai-model-api-generate-image.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "beautiful landscape, mountains, sunset",
    "api_key": "your-super-secret-key-here",
    "num_steps": 20
  }'
```

## 4. Integrasi dengan Telegram Bot

### Install Telegram Bot Library
```bash
pip install python-telegram-bot
```

### Kode Telegram Bot
```python
import asyncio
import base64
import io
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Konfigurasi
TELEGRAM_BOT_TOKEN = "your-telegram-bot-token"
MODAL_API_URL = "https://your-workspace--civitai-model-api-generate-image.modal.run"
MODAL_API_KEY = "your-super-secret-key-here"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Halo! Kirim prompt untuk generate gambar.\n"
        "Contoh: /generate beautiful landscape with mountains"
    )

async def generate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Gunakan: /generate [prompt]")
        return
    
    prompt = " ".join(context.args)
    await update.message.reply_text("⏳ Generating image...")
    
    try:
        # Call Modal API
        response = requests.post(
            MODAL_API_URL,
            json={
                "prompt": prompt,
                "api_key": MODAL_API_KEY,
                "num_steps": 20,
                "guidance_scale": 7.5
            },
            timeout=120
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Decode base64 image
        image_data = base64.b64decode(result["image"])
        image_file = io.BytesIO(image_data)
        
        # Send image
        await update.message.reply_photo(
            photo=image_file,
            caption=f"Prompt: {prompt}"
        )
        
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("generate", generate_command))
    
    print("Bot started!")
    app.run_polling()

if __name__ == "__main__":
    main()
```

### Jalankan Bot
```bash
python telegram_bot.py
```

## 5. Tips & Troubleshooting

### Monitoring Modal
```bash
# Lihat logs
modal app logs civitai-model-api

# Lihat volumes
modal volume list
```

### Optimasi Biaya
- Gunakan `container_idle_timeout` untuk auto-shutdown container
- Pilih GPU sesuai kebutuhan (T4 lebih murah, A100 lebih cepat)
- Cache model di Volume untuk menghindari re-download

### Error Handling
- Jika timeout saat download: tingkatkan `timeout` parameter
- Jika OOM (Out of Memory): gunakan GPU lebih besar atau kurangi batch size
- Jika model tidak load: pastikan format model sesuai dengan loader

## 6. Deploy Bot ke Production

### Opsi 1: Render.com atau Railway.app
1. Push kode bot ke GitHub
2. Deploy sebagai web service
3. Pastikan service always running

### Opsi 2: Modal.com untuk Bot juga
```python
# Deploy bot di Modal juga
@app.function(schedule=modal.Period(seconds=1))
def run_telegram_bot():
    # Kode bot di sini
    pass
```

### Opsi 3: VPS/Cloud VM
```bash
# Install supervisor untuk keep-alive
sudo apt-get install supervisor
```

## Environment Variables
Buat file `.env`:
```
TELEGRAM_BOT_TOKEN=your-token
MODAL_API_URL=your-modal-url
MODAL_API_KEY=your-api-key
```

Load dengan:
```python
from dotenv import load_dotenv
load_dotenv()
```
