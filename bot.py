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
    TELEGRAM_TOKEN = "8222362928:AAG85K4WRPmf2yBPb_6j3uJiMHDYgscgolc"
    CEREBRAS_API_KEY = "csk-j439vyke89px4we44r29wcvetwcfm6mjmp5xwmxx4m2mpmcn"
    MODAL_API_URL = "https://oktetod--civitai-api-fastapi-fastapi-app.modal.run"
    MODAL_API_KEY = "gilpad008"
    ADMIN_USER_ID = "8484686373" # Ganti dengan ID Telegram Anda
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
        lora_list = await database.get_loras_from_db()
        lora_list_json = json.dumps(lora_list, indent=2)

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
# MODAL API CLIENT (FIXED for RuntimeError: no running event loop)
# ===================================================================
class ModalAPIClient:
    def __init__(self):
        self.base_url = Config.MODAL_API_URL.rstrip('/')
        self.api_key = Config.MODAL_API_KEY
        self.session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def get_all_loras(self) -> List[str]:
        all_loras = []
        page = 1
        session = await self._get_session()
        while True:
            try:
                async with session.get(f"{self.base_url}/loras", params={"page": page, "limit": 100}) as response:
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
    
    async def generate_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        endpoint = "text2img"
        if params.get("init_image"): endpoint = "img2img"
        return await self._make_request(endpoint, params)

    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint}"
        params["api_key"] = self.api_key
        session = await self._get_session()
        async with session.post(url, json=params, timeout=aiohttp.ClientTimeout(total=Config.REQUEST_TIMEOUT)) as response:
            response.raise_for_status()
            return await response.json()

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("Aiohttp session closed.")

# ===================================================================
# SMART BOT HANDLER
# ===================================================================
class SmartImageBot:
    def __init__(self):
        self.cerebras = CerebrasAI()
        self.modal = ModalAPIClient()

    async def update_loras_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        if user_id != Config.ADMIN_USER_ID:
            await update.message.reply_text("⛔ Anda tidak memiliki izin untuk menjalankan perintah ini.")
            return

        await update.message.reply_text("🔄 Memulai sinkronisasi LoRA... Mohon tunggu.")
        
        try:
            api_loras = await self.modal.get_all_loras()
            if not api_loras:
                await update.message.reply_text("❌ Gagal mengambil daftar LoRA dari Modal API.")
                return
                
            sync_result = await database.sync_loras_to_db(api_loras)
            
            report = (
                f"✅ Sinkronisasi LoRA selesai!\n\n"
                f"➕ LoRA baru ditambahkan: {sync_result['added']}\n"
                f"➖ LoRA lama dihapus: {sync_result['removed']}\n"
                f"📊 Total LoRA di database sekarang: {len(api_loras)}"
            )
            await update.message.reply_text(report)
        except Exception as e:
            logger.error(f"Error during /update_loras: {e}")
            await update.message.reply_text(f"Terjadi kesalahan saat sinkronisasi: {e}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            await update.message.chat.send_action("typing")
            
            prompt_text = update.message.text or update.message.caption or ""
            has_image = bool(update.message.photo)
            
            analysis = await self.cerebras.analyze_and_select_tools(
                message=prompt_text,
                has_image=has_image
            )

            # Siapkan parameter dasar untuk API Modal
            params = {
                "prompt": analysis.get("enhanced_prompt", prompt_text),
                "num_steps": 35,
                "lora_name": analysis.get("selected_lora") if analysis.get("selected_lora") != "null" else None,
                "lora_scale": 0.8,
                "width": 1024,
                "height": 1024
            }

            if has_image:
                file = await context.bot.get_file(update.message.photo[-1].file_id)
                img_bytes = io.BytesIO()
                await file.download_to_memory(img_bytes)
                img_bytes.seek(0)
                params["init_image"] = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
                params["strength"] = 0.75 # Nilai default untuk i2i

            status_message = await update.message.reply_text("🎨 Permintaan Anda sedang diproses, mohon tunggu...")

            result = await self.modal.generate_image(params)
            
            image_data = base64.b64decode(result["image"])
            
            caption = (
                f"✨ **Hasil Gambar**\n\n"
                f"**Prompt:**\n`{result.get('prompt', 'N/A')}`\n\n"
                f"**LoRA:** `{result.get('lora', 'None')}`\n"
                f"**Seed:** `{result.get('seed', 'N/A')}`"
            )

            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=io.BytesIO(image_data),
                caption=caption,
                parse_mode='Markdown'
            )
            await status_message.delete()

        except Exception as e:
            logger.error(f"Handler error: {e}", exc_info=True)
            error_message = f"Terjadi kesalahan: {e}"
            if "status_message" in locals() and status_message:
                await status_message.edit_text(error_message)
            else:
                await update.message.reply_text(error_message)


# ===================================================================
# MAIN APPLICATION & STARTUP
# ===================================================================
async def post_init(application: Application):
    logger.info("Bot initialized. Running post-init setup...")
    await database.init_db()
    
    bot_instance = application.bot_data["bot_instance"]
    logger.info("Running initial LoRA sync on startup...")
    api_loras = await bot_instance.modal.get_all_loras()
    if api_loras:
        sync_result = await database.sync_loras_to_db(api_loras)
        logger.info(f"Initial sync complete: {sync_result['added']} added, {sync_result['removed']} removed.")
    else:
        logger.warning("Could not fetch LoRAs for initial sync.")

async def main():
    """Memulai bot secara asynchronous."""
    bot = SmartImageBot()
    
    application = (
        Application.builder()
        .token(Config.TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )
    application.bot_data["bot_instance"] = bot

    application.add_handler(CommandHandler("update_loras", bot.update_loras_command))
    application.add_handler(MessageHandler(filters.TEXT | filters.PHOTO, bot.handle_message))
    
    logger.info("🚀 Starting bot asynchronously...")

    try:
        await application.initialize()
        await application.start()
        await application.updater.start_polling()
        logger.info("✅ Bot is polling.")
        
        # Jaga agar skrip tetap berjalan selamanya
        await asyncio.Event().wait()
            
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        logger.info("🛑 Shutting down bot...")
        if application.updater and application.updater.running:
            await application.updater.stop()
        if application.running:
            await application.stop()
        await application.shutdown()
        await bot.modal.close()
        logger.info("👋 Bot shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
