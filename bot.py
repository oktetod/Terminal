# Filename: bot.py
import os
import io
import json
import base64
import logging
import asyncio
import aiohttp
from typing import Dict, Any, List

from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, ContextTypes, filters

# Impor fungsi-fungsi dari file database.py
import database

# ===================================================================
# LOGGING & CONFIGURATION
# ===================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    TELEGRAM_TOKEN = os.environ.get("8222362928:AAG85K4WRPmf2yBPb_6j3uJiMHDYgscgolc")
    CEREBRAS_API_KEY = os.environ.get("csk-j439vyke89px4we44r29wcvetwcfm6mjmp5xwmxx4m2mpmcn")
    MODAL_API_URL = os.environ.get("https://oktetod--civitai-api-fastapi-fastapi-app.modal.run")
    MODAL_API_KEY = os.environ.get("gilpad008")
    ADMIN_USER_ID = int(os.environ.get("8484686373", 0)) # ID Telegram Anda untuk akses perintah admin
    REQUEST_TIMEOUT = 180

# ===================================================================
# CEREBRAS AI CLIENT
# ===================================================================
class CerebrasAI:
    def __init__(self):
        from cerebras.cloud.sdk import Cerebras
        self.client = Cerebras(api_key=Config.CEREBRAS_API_KEY)
        self.prompt_template = """
You are an autonomous image generation expert... Your goal is to analyze a user's prompt and select the MOST appropriate LoRA from the list provided below.

**Available LoRA Models:**
{lora_list_json}

**Decision Rules:**
- Choose a LoRA if the prompt's style description strongly matches its name.
- If the prompt is generic, you can choose null.

**Output Format (JSON ONLY):**
{{
  "intent": "t2i|i2i",
  "enhanced_prompt": "Your enhanced version of the user's prompt.",
  "selected_lora": "lora_name_from_list_or_null",
  "reasoning": "A brief explanation of your choice."
}}
"""
    async def analyze_and_select_tools(self, message: str, has_image: bool = False) -> Dict[str, Any]:
        # Mengambil daftar LoRA TERBARU dari database setiap kali dipanggil
        lora_list = await database.get_loras_from_db()
        lora_list_json = json.dumps(lora_list, indent=2)

        # Membuat system prompt dinamis dengan daftar LoRA dari database
        system_prompt = self.prompt_template.format(lora_list_json=lora_list_json)
        
        user_content = f"User message: '{message}'. Has attached image: {has_image}."
        
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            model="gpt-oss-120b", temperature=0.5, max_completion_tokens=512
        )
        result_text = response.choices[0].message.content
        if "```json" in result_text: result_text = result_text.split("```json")[1].split("```")[0]
        
        analysis = json.loads(result_text.strip())
        logger.info(f"Cerebras Analysis using DB LoRAs: {analysis}")
        return analysis

# ===================================================================
# MODAL API CLIENT
# ===================================================================
class ModalAPIClient:
    def __init__(self):
        self.base_url = Config.MODAL_API_URL.rstrip('/')
        self.api_key = Config.MODAL_API_KEY
        self.session = aiohttp.ClientSession()

    async def get_all_loras(self) -> List[str]:
        """Mengambil SEMUA LoRA dari API Modal, menangani paginasi."""
        all_loras = []
        page = 1
        while True:
            try:
                async with self.session.get(f"{self.base_url}/loras", params={"page": page, "limit": 100}) as response:
                    response.raise_for_status()
                    data = await response.json()
                    loras_on_page = data.get("loras", [])
                    if not loras_on_page:
                        break
                    all_loras.extend(loras_on_page)
                    page += 1
            except Exception as e:
                logger.error(f"Failed to fetch LoRAs on page {page}: {e}")
                break
        return all_loras
    
    # ... (fungsi generate_image dan _make_request tetap sama seperti sebelumnya)
    async def generate_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        endpoint = "text2img"
        if params.get("init_image"): endpoint = "img2img"
        return await self._make_request(endpoint, params)

    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint}"
        params["api_key"] = self.api_key
        async with self.session.post(url, json=params, timeout=aiohttp.ClientTimeout(total=Config.REQUEST_TIMEOUT)) as response:
            response.raise_for_status()
            return await response.json()

    async def close(self):
        await self.session.close()

# ===================================================================
# SMART BOT HANDLER
# ===================================================================
class SmartImageBot:
    def __init__(self):
        self.cerebras = CerebrasAI()
        self.modal = ModalAPIClient()

    async def update_loras_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler untuk perintah /update_loras, hanya untuk admin."""
        user_id = update.effective_user.id
        if user_id != Config.ADMIN_USER_ID:
            await update.message.reply_text("⛔ Anda tidak memiliki izin untuk menjalankan perintah ini.")
            return

        await update.message.reply_text("🔄 Memulai sinkronisasi LoRA... Mohon tunggu.")
        
        # 1. Ambil daftar LoRA terbaru dari API Modal
        api_loras = await self.modal.get_all_loras()
        if not api_loras:
            await update.message.reply_text("❌ Gagal mengambil daftar LoRA dari Modal API.")
            return
            
        # 2. Lakukan sinkronisasi (compare & merge) ke database
        sync_result = await database.sync_loras_to_db(api_loras)
        
        # 3. Laporkan hasilnya
        report = (
            f"✅ Sinkronisasi LoRA selesai!\n\n"
            f"➕ LoRA baru ditambahkan: {sync_result['added']}\n"
            f"➖ LoRA lama dihapus: {sync_result['removed']}\n"
            f"📊 Total LoRA di database sekarang: {len(api_loras)}"
        )
        await update.message.reply_text(report)

    # ... (fungsi handle_message tetap sama seperti sebelumnya)
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            # ... (logika handle_message tidak berubah signifikan)
            await update.message.chat.send_action("typing")
            analysis = await self.cerebras.analyze_and_select_tools(
                message=update.message.text or "",
                has_image=bool(update.message.photo)
            )
            # ... (sisa logika untuk menyiapkan params dan generate image)
            params = {
                "prompt": analysis.get("enhanced_prompt", update.message.text), "num_steps": 35,
                "lora_name": analysis.get("selected_lora"),
                # ... etc
            }
            if update.message.photo:
                # ... (get photo logic)
                pass # placeholder
            
            result = await self.modal.generate_image(params)
            # ... (send photo logic)
            pass # placeholder
        except Exception as e:
            logger.error(f"Handler error: {e}")
            await update.message.reply_text(f"Error: {e}")

# ===================================================================
# MAIN APPLICATION & STARTUP
# ===================================================================
async def post_init(application: Application):
    """Fungsi yang dijalankan setelah bot siap, sebelum polling dimulai."""
    logger.info("Bot initialized. Running post-init setup...")
    # 1. Inisialisasi skema database
    await database.init_db()
    
    # 2. Jalankan sinkronisasi LoRA pertama kali secara otomatis
    bot_instance = application.bot_data["bot_instance"]
    logger.info("Running initial LoRA sync on startup...")
    api_loras = await bot_instance.modal.get_all_loras()
    if api_loras:
        sync_result = await database.sync_loras_to_db(api_loras)
        logger.info(f"Initial sync complete: {sync_result['added']} added, {sync_result['removed']} removed.")
    else:
        logger.warning("Could not fetch LoRAs for initial sync.")

def main():
    bot = SmartImageBot()
    application = (
        Application.builder()
        .token(Config.TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )
    application.bot_data["bot_instance"] = bot

    # Tambahkan command handler untuk /update_loras
    application.add_handler(CommandHandler("update_loras", bot.update_loras_command))
    # Tambahkan message handler utama
    application.add_handler(MessageHandler(filters.TEXT | filters.PHOTO, bot.handle_message))
    
    logger.info("🚀 Starting bot...")
    application.run_polling()
    asyncio.run(bot.modal.close())

if __name__ == "__main__":
    main()
