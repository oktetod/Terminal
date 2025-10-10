# Filename: bot.py
import os
import io
import json
import base64
import logging
import asyncio
import aiohttp
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, ContextTypes, filters

import database

# ===================================================================
# LOGGING & CONFIGURATION
# ===================================================================
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class Config:
    TELEGRAM_TOKEN = "8222362928:AAG85K4WRPmf2yBPb_6j3uJiMHDYgscgolc"
    CEREBRAS_API_KEY = "csk-j439vyke89px4we44r29wcvetwcfm6mjmp5xwmxx4m2mpmcn"
    MODAL_API_URL = "https://oktetod--civitai-api-fastapi-fastapi-app.modal.run"
    MODAL_API_KEY = "gilpad008"
    ADMIN_USER_ID = "8484686373"
    REQUEST_TIMEOUT = 180
    MAX_RETRIES = 3
    CACHE_DURATION = 300  # 5 menit cache untuk LoRA list

# ===================================================================
# IN-MEMORY CACHE
# ===================================================================
class SimpleCache:
    """Simple in-memory cache dengan TTL"""
    def __init__(self):
        self._cache: Dict[str, tuple] = {}  # {key: (value, expiry_time)}
    
    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, expiry = self._cache[key]
            if datetime.now() < expiry:
                return value
            else:
                del self._cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 300):
        expiry = datetime.now() + timedelta(seconds=ttl_seconds)
        self._cache[key] = (value, expiry)
    
    def clear(self):
        self._cache.clear()

# ===================================================================
# CEREBRAS AI CLIENT
# ===================================================================
class CerebrasAI:
    def __init__(self):
        from cerebras.cloud.sdk import Cerebras
        self.client = Cerebras(api_key=Config.CEREBRAS_API_KEY)
        self.cache = SimpleCache()
        self.prompt_template = """You are an autonomous image generation expert AI assistant. Your goal is to analyze a user's prompt and select the MOST appropriate LoRA (Low-Rank Adaptation) model from the list provided below.

**Your Responsibilities:**
1. Understand the user's creative intent (style, theme, subject matter)
2. Enhance their prompt with better descriptive details for optimal image generation
3. Select the most suitable LoRA from the available list, or choose null if none fit well
4. Determine whether this is a text-to-image (t2i) or image-to-image (i2i) task

**Available LoRA Models:**
{lora_list_json}

**Decision Rules:**
- Choose a LoRA ONLY if the prompt's style, theme, or subject strongly matches the LoRA's name/description
- If the prompt is generic or no LoRA clearly fits, set selected_lora to null
- Consider artistic styles (anime, realistic, cartoon, etc.), themes, and specific visual characteristics
- Enhance the prompt with descriptive details: lighting, composition, quality, mood, colors, etc.
- Be conservative: when in doubt, choose null rather than forcing a poor match

**Output Format (JSON ONLY - NO OTHER TEXT):**
{{
  "intent": "t2i",
  "enhanced_prompt": "Your enhanced version of the user's prompt with rich descriptive details",
  "selected_lora": "exact_lora_name_from_list_or_null",
  "reasoning": "Brief explanation of why you chose this LoRA or null"
}}

**Examples:**
- User: "cute anime girl" → might select anime-style LoRA if available
- User: "photorealistic portrait" → might select realistic/photo LoRA if available
- User: "abstract painting" → likely null unless specific abstract LoRA exists
- User with image: "make it more vibrant" → intent should be "i2i"
"""

    async def analyze_and_select_tools(
        self, 
        message: str, 
        has_image: bool = False
    ) -> Dict[str, Any]:
        """Analyze user message and select appropriate LoRA with caching"""
        try:
            # Get LoRA list with caching
            lora_list = self.cache.get("lora_list")
            if lora_list is None:
                lora_list = await database.get_loras_from_db()
                self.cache.set("lora_list", lora_list, Config.CACHE_DURATION)
            
            if not lora_list:
                logger.warning("No LoRAs available in database")
                lora_list = []
            
            lora_list_json = json.dumps(lora_list, indent=2)
            system_prompt = self.prompt_template.format(lora_list_json=lora_list_json)
            user_content = f"User message: '{message}'. Has attached image: {has_image}."
            
            # Call Cerebras API with timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.chat.completions.create,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    model="gpt-oss-120b",
                    temperature=0.5,
                    max_completion_tokens=512
                ),
                timeout=30
            )
            
            result_text = response.choices[0].message.content
            
            # Clean up JSON from markdown code blocks
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            # Parse and validate JSON
            analysis = json.loads(result_text.strip())
            
            # Validate required fields
            required_keys = ["intent", "enhanced_prompt", "selected_lora", "reasoning"]
            if not all(key in analysis for key in required_keys):
                raise ValueError(f"Missing required keys. Got: {list(analysis.keys())}")
            
            # Normalize selected_lora
            if analysis.get("selected_lora") in ["null", "None", "", None]:
                analysis["selected_lora"] = None
            
            logger.info(f"✓ Cerebras Analysis: LoRA='{analysis.get('selected_lora')}', Intent={analysis.get('intent')}")
            return analysis
            
        except asyncio.TimeoutError:
            logger.error("Cerebras API timeout")
            return self._get_fallback_analysis(message, has_image)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}. Raw response: {result_text[:200]}")
            return self._get_fallback_analysis(message, has_image)
        except Exception as e:
            logger.error(f"Error in analyze_and_select_tools: {e}", exc_info=True)
            return self._get_fallback_analysis(message, has_image)
    
    def _get_fallback_analysis(self, message: str, has_image: bool) -> Dict[str, Any]:
        """Fallback response when Cerebras fails"""
        return {
            "intent": "i2i" if has_image else "t2i",
            "enhanced_prompt": message,
            "selected_lora": None,
            "reasoning": "Fallback due to analysis error"
        }

# ===================================================================
# MODAL API CLIENT
# ===================================================================
class ModalAPIClient:
    def __init__(self):
        self.base_url = Config.MODAL_API_URL.rstrip('/')
        self.api_key = Config.MODAL_API_KEY
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(
                total=Config.REQUEST_TIMEOUT,
                connect=10,
                sock_read=Config.REQUEST_TIMEOUT
            )
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def get_all_loras(self) -> List[str]:
        """Fetch all LoRAs from API with pagination"""
        all_loras = []
        page = 1
        session = await self._get_session()
        
        while True:
            try:
                async with session.get(
                    f"{self.base_url}/loras",
                    params={"page": page, "limit": 100}
                ) as response:
                    if response.status != 200:
                        logger.error(f"API returned status {response.status} on page {page}")
                        break
                    
                    data = await response.json()
                    loras_on_page = data.get("loras", [])
                    
                    if not loras_on_page:
                        break
                    
                    all_loras.extend(loras_on_page)
                    logger.info(f"Fetched page {page}: {len(loras_on_page)} LoRAs")
                    page += 1
                    
            except asyncio.TimeoutError:
                logger.error(f"Timeout fetching LoRAs on page {page}")
                break
            except Exception as e:
                logger.error(f"Failed to fetch LoRAs on page {page}: {e}")
                break
        
        logger.info(f"✓ Successfully fetched {len(all_loras)} LoRAs in total.")
        return all_loras
    
    async def generate_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image with retry logic"""
        endpoint = "img2img" if params.get("init_image") else "text2img"
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                return await self._make_request(endpoint, params)
            except asyncio.TimeoutError:
                if attempt < Config.MAX_RETRIES - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Request timeout, retrying in {wait_time}s... (attempt {attempt + 1}/{Config.MAX_RETRIES})")
                    await asyncio.sleep(wait_time)
                else:
                    raise
            except aiohttp.ClientError as e:
                if attempt < Config.MAX_RETRIES - 1:
                    logger.warning(f"Request failed: {e}, retrying... (attempt {attempt + 1}/{Config.MAX_RETRIES})")
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to Modal API"""
        url = f"{self.base_url}/{endpoint}"
        params["api_key"] = self.api_key
        
        session = await self._get_session()
        prompt_preview = params.get('prompt', '')[:50]
        logger.info(f"→ POST {endpoint}: {prompt_preview}...")
        
        async with session.post(url, json=params) as response:
            if response.status != 200:
                error_text = await response.text()
                raise aiohttp.ClientError(f"API error {response.status}: {error_text[:200]}")
            
            result = await response.json()
            logger.info(f"✓ {endpoint} completed successfully")
            return result

    async def close(self):
        """Close aiohttp session"""
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
        """Admin command to sync LoRAs from API to database"""
        user_id = str(update.effective_user.id)
        if user_id != Config.ADMIN_USER_ID:
            await update.message.reply_text("⛔ Anda tidak memiliki izin untuk menjalankan perintah ini.")
            return

        status_msg = await update.message.reply_text("🔄 Memulai sinkronisasi LoRA dari Modal API...\nMohon tunggu, ini mungkin memakan waktu.")
        
        try:
            # Fetch from API
            api_loras = await self.modal.get_all_loras()
            
            if not api_loras:
                await status_msg.edit_text("❌ Gagal mengambil daftar LoRA dari Modal API atau daftar kosong.")
                return
            
            # Sync to database
            sync_result = await database.sync_loras_to_db(api_loras)
            
            # Clear cache
            self.cerebras.cache.clear()
            
            # Send report
            report = (
                f"✅ **Sinkronisasi LoRA Selesai!**\n\n"
                f"➕ **Ditambahkan:** {sync_result['added']}\n"
                f"➖ **Dihapus:** {sync_result['removed']}\n"
                f"📊 **Total LoRA:** {sync_result['total']}\n"
                f"🕒 **Waktu:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            await status_msg.edit_text(report, parse_mode='Markdown')
            logger.info(f"Admin sync completed: {sync_result}")
            
        except Exception as e:
            logger.error(f"Error in update_loras_command: {e}", exc_info=True)
            await status_msg.edit_text(f"❌ Terjadi kesalahan:\n`{str(e)[:200]}`", parse_mode='Markdown')

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Main message handler for image generation"""
        status_message = None
        
        try:
            # Send typing indicator
            await update.message.chat.send_action("typing")
            
            # Extract prompt and check for image
            prompt_text = update.message.text or update.message.caption or ""
            has_image = bool(update.message.photo)
            
            if not prompt_text.strip():
                await update.message.reply_text("📝 Silakan berikan deskripsi gambar yang ingin Anda buat!")
                return
            
            # Step 1: Initial status
            status_message = await update.message.reply_text("🤖 Menganalisis prompt Anda...")
            
            # Step 2: Analyze with Cerebras
            analysis = await self.cerebras.analyze_and_select_tools(
                message=prompt_text,
                has_image=has_image
            )
            
            # Step 3: Prepare generation parameters
            params = {
                "prompt": analysis.get("enhanced_prompt", prompt_text),
                "num_steps": 35,
                "lora_name": analysis.get("selected_lora")
            }
            
            # Handle image-to-image
            if has_image:
                await status_message.edit_text("📥 Memproses gambar yang diunggah...")
                
                file = await context.bot.get_file(update.message.photo[-1].file_id)
                
                # Use context manager for BytesIO
                with io.BytesIO() as img_bytes:
                    await file.download_to_memory(img_bytes)
                    img_bytes.seek(0)
                    params["init_image"] = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
                
                params["strength"] = 0.75
            
            # Step 4: Update status with LoRA info
            lora_info = f"LoRA: {params['lora_name']}" if params['lora_name'] else "Model default"
            await status_message.edit_text(f"🎨 Membuat gambar...\n{lora_info}")
            
            # Step 5: Generate image
            result = await self.modal.generate_image(params)
            
            # Step 6: Decode and send image
            image_data = base64.b64decode(result["image"])
            
            caption = (
                f"✨ **Hasil Gambar**\n\n"
                f"**Prompt:**\n`{result.get('prompt', 'N/A')[:200]}{'...' if len(result.get('prompt', '')) > 200 else ''}`\n\n"
                f"**LoRA:** `{result.get('lora', 'None')}`\n"
                f"**Seed:** `{result.get('seed', 'N/A')}`\n"
                f"**Steps:** `{result.get('num_steps', 'N/A')}`"
            )
            
            # Send photo
            with io.BytesIO(image_data) as photo_bytes:
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id,
                    photo=photo_bytes,
                    caption=caption,
                    parse_mode='Markdown'
                )
            
            # Delete status message
            await status_message.delete()
            logger.info(f"✓ Successfully generated image for user {update.effective_user.id}")
            
        except asyncio.TimeoutError:
            error_msg = "⏱️ Permintaan timeout. Server membutuhkan waktu terlalu lama. Silakan coba lagi."
            logger.error("Request timeout in handle_message")
            await self._send_error_message(status_message, update, error_msg)
            
        except aiohttp.ClientError as e:
            error_msg = f"🌐 Kesalahan koneksi ke server: {str(e)[:100]}"
            logger.error(f"API client error: {e}")
            await self._send_error_message(status_message, update, error_msg)
            
        except Exception as e:
            error_msg = "❌ Terjadi kesalahan saat memproses permintaan. Silakan coba lagi."
            logger.error(f"Error in handle_message: {e}", exc_info=True)
            await self._send_error_message(status_message, update, error_msg)
    
    async def _send_error_message(
        self,
        status_message: Optional[Any],
        update: Update,
        error_text: str
    ):
        """Helper to send error messages"""
        try:
            if status_message:
                await status_message.edit_text(error_text)
            else:
                await update.message.reply_text(error_text)
        except Exception as e:
            logger.error(f"Failed to send error message: {e}")

# ===================================================================
# MAIN APPLICATION & STARTUP
# ===================================================================
async def post_init(application: Application):
    """Post-initialization setup"""
    logger.info("=" * 60)
    logger.info("Bot initialized. Running post-init setup...")
    logger.info("=" * 60)
    
    # Initialize database
    await database.init_db()
    
    # Get bot instance
    bot_instance = application.bot_data["bot_instance"]
    
    # Run initial LoRA sync
    logger.info("Running initial LoRA sync on startup...")
    try:
        api_loras = await bot_instance.modal.get_all_loras()
        
        if api_loras:
            sync_result = await database.sync_loras_to_db(api_loras)
            logger.info(f"✓ Initial sync complete: {sync_result['added']} added, {sync_result['removed']} removed, {sync_result['total']} total.")
            
            # Notify admin
            try:
                await application.bot.send_message(
                    chat_id=Config.ADMIN_USER_ID,
                    text=(
                        f"✅ **Bot Started Successfully**\n\n"
                        f"LoRA sync completed:\n"
                        f"➕ Added: {sync_result['added']}\n"
                        f"➖ Removed: {sync_result['removed']}\n"
                        f"📊 Total: {sync_result['total']}\n"
                        f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    ),
                    parse_mode='Markdown'
                )
            except Exception:
                pass  # Non-critical if notification fails
        else:
            logger.warning("⚠️ Could not fetch LoRAs for initial sync.")
            
    except Exception as e:
        logger.error(f"❌ Initial LoRA sync failed: {e}", exc_info=True)
        
        # Notify admin of failure
        try:
            await application.bot.send_message(
                chat_id=Config.ADMIN_USER_ID,
                text=(
                    f"⚠️ **Bot Started with Warnings**\n\n"
                    f"Initial LoRA sync FAILED.\n\n"
                    f"Error: `{str(e)[:200]}`\n\n"
                    f"Bot is running but may have incomplete LoRA list. "
                    f"Use /update_loras to retry sync."
                ),
                parse_mode='Markdown'
            )
        except Exception:
            pass
    
    logger.info("=" * 60)
    logger.info("Post-init complete. Bot is ready!")
    logger.info("=" * 60)

async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler"""
    logger.error("Exception while handling an update:", exc_info=context.error)
    
    # Format error message
    tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
    tb_string = "".join(tb_list)
    
    update_str = update.to_dict() if isinstance(update, Update) else str(update)
    
    message = (
        f"⚠️ <b>Exception occurred</b>\n\n"
        f"<b>Error:</b> <code>{str(context.error)[:200]}</code>\n\n"
        f"<b>Update:</b>\n<pre>{json.dumps(update_str, indent=2, ensure_ascii=False)[:500]}</pre>\n\n"
        f"<b>Traceback:</b>\n<pre>{tb_string[:2000]}</pre>"
    )
    
    # Truncate if too long
    max_length = 4096
    if len(message) > max_length:
        message = message[:max_length - 20] + "\n... (truncated)"
    
    # Send to admin
    try:
        await context.bot.send_message(
            chat_id=Config.ADMIN_USER_ID,
            text=message,
            parse_mode='HTML'
        )
    except Exception as e:
        logger.error(f"Failed to send error notification to admin: {e}")

async def main():
    """Main entry point"""
    bot = SmartImageBot()
    
    # Build application
    application = (
        Application.builder()
        .token(Config.TELEGRAM_TOKEN)
        .post_init(post_init)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .pool_timeout(30)
        .build()
    )
    
    # Store bot instance
    application.bot_data["bot_instance"] = bot

    # Register handlers
    application.add_handler(CommandHandler("update_loras", bot.update_loras_command))
    application.add_handler(MessageHandler(filters.TEXT | filters.PHOTO, bot.handle_message))
    application.add_error_handler(handle_error)
    
    logger.info("🚀 Starting bot...")

    try:
        await application.initialize()
        await application.start()
        await application.updater.start_polling(
            poll_interval=2.0,
            timeout=30,
            drop_pending_updates=True,
            allowed_updates=Update.ALL_TYPES
        )
        logger.info("✅ Bot is polling and ready to receive messages.")
        
        # Keep running
        await asyncio.Event().wait()
            
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 Bot stopped by user.")
    except Exception as e:
        logger.critical(f"💥 A critical error occurred in the main loop: {e}", exc_info=True)
    finally:
        logger.info("🔄 Shutting down bot...")
        try:
            if application.updater and application.updater.running:
                await application.updater.stop()
            if application.running:
                await application.stop()
            await application.shutdown()
            await bot.modal.close()
            await database.close_pool()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        logger.info("👋 Bot shutdown complete.")

if __name__ == "__main__":
    asyncio.run(main())
