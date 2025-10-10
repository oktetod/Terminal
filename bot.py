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

from telethon import TelegramClient, events
from telethon.tl.types import DocumentAttributeFilename

import database

# ===================================================================
# LOGGING & CONFIGURATION
# ===================================================================
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telethon").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class Config:
    # Get from https://my.telegram.org
    API_ID = 28535133  # Replace with your API ID
    API_HASH = "dede41f452ebdbb66ee56b50c95b53ba"  # Replace with your API Hash
    BOT_TOKEN = "8222362928:AAG85K4WRPmf2yBPb_6j3uJiMHDYgscgolc"
    
    CEREBRAS_API_KEY = "csk-j439vyke89px4we44r29wcvetwcfm6mjmp5xwmxx4m2mpmcn"
    MODAL_API_URL = "https://oktetod--civitai-api-fastapi-fastapi-app.modal.run"
    MODAL_API_KEY = "gilpad008"
    ADMIN_USER_ID = 8484686373  # Integer for Telethon
    REQUEST_TIMEOUT = 180
    MAX_RETRIES = 3
    CACHE_DURATION = 300

# ===================================================================
# IN-MEMORY CACHE
# ===================================================================
class SimpleCache:
    """Simple in-memory cache dengan TTL"""
    def __init__(self):
        self._cache: Dict[str, tuple] = {}
    
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
            
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            analysis = json.loads(result_text.strip())
            
            required_keys = ["intent", "enhanced_prompt", "selected_lora", "reasoning"]
            if not all(key in analysis for key in required_keys):
                raise ValueError(f"Missing required keys. Got: {list(analysis.keys())}")
            
            if analysis.get("selected_lora") in ["null", "None", "", None]:
                analysis["selected_lora"] = None
            
            logger.info(f"✓ Cerebras Analysis: LoRA='{analysis.get('selected_lora')}', Intent={analysis.get('intent')}")
            return analysis
            
        except asyncio.TimeoutError:
            logger.error("Cerebras API timeout")
            return self._get_fallback_analysis(message, has_image)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
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
                    wait_time = 2 ** attempt
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
# TELETHON BOT
# ===================================================================
class SmartImageBot:
    def __init__(self, client: TelegramClient):
        self.client = client
        self.cerebras = CerebrasAI()
        self.modal = ModalAPIClient()
        
    async def start(self):
        """Initialize bot and register handlers"""
        logger.info("Initializing bot...")
        
        # Initialize database
        await database.init_db()
        
        # Run initial LoRA sync
        await self._initial_sync()
        
        # Register event handlers
        self.client.add_event_handler(
            self.handle_update_loras,
            events.NewMessage(pattern='/update_loras')
        )
        
        self.client.add_event_handler(
            self.handle_message,
            events.NewMessage(incoming=True, func=lambda e: not e.text.startswith('/'))
        )
        
        logger.info("✅ Bot handlers registered and ready!")
    
    async def _initial_sync(self):
        """Run initial LoRA synchronization"""
        logger.info("Running initial LoRA sync on startup...")
        try:
            api_loras = await self.modal.get_all_loras()
            
            if api_loras:
                sync_result = await database.sync_loras_to_db(api_loras)
                logger.info(f"✓ Initial sync complete: {sync_result}")
                
                # Notify admin
                try:
                    await self.client.send_message(
                        Config.ADMIN_USER_ID,
                        f"✅ **Bot Started Successfully**\n\n"
                        f"LoRA sync completed:\n"
                        f"➕ Added: {sync_result['added']}\n"
                        f"➖ Removed: {sync_result['removed']}\n"
                        f"📊 Total: {sync_result['total']}\n"
                        f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                except Exception as e:
                    logger.error(f"Failed to notify admin: {e}")
            else:
                logger.warning("⚠️ Could not fetch LoRAs for initial sync.")
                
        except Exception as e:
            logger.error(f"❌ Initial LoRA sync failed: {e}", exc_info=True)
            
            try:
                await self.client.send_message(
                    Config.ADMIN_USER_ID,
                    f"⚠️ **Bot Started with Warnings**\n\n"
                    f"Initial LoRA sync FAILED.\n\n"
                    f"Error: `{str(e)[:200]}`\n\n"
                    f"Bot is running but may have incomplete LoRA list. "
                    f"Use /update_loras to retry sync."
                )
            except Exception:
                pass
    
    async def handle_update_loras(self, event):
        """Handle /update_loras command"""
        sender = await event.get_sender()
        
        if sender.id != Config.ADMIN_USER_ID:
            await event.respond("⛔ Anda tidak memiliki izin untuk menjalankan perintah ini.")
            return
        
        status_msg = await event.respond(
            "🔄 Memulai sinkronisasi LoRA dari Modal API...\n"
            "Mohon tunggu, ini mungkin memakan waktu."
        )
        
        try:
            api_loras = await self.modal.get_all_loras()
            
            if not api_loras:
                await status_msg.edit("❌ Gagal mengambil daftar LoRA dari Modal API atau daftar kosong.")
                return
            
            sync_result = await database.sync_loras_to_db(api_loras)
            self.cerebras.cache.clear()
            
            report = (
                f"✅ **Sinkronisasi LoRA Selesai!**\n\n"
                f"➕ **Ditambahkan:** {sync_result['added']}\n"
                f"➖ **Dihapus:** {sync_result['removed']}\n"
                f"📊 **Total LoRA:** {sync_result['total']}\n"
                f"🕒 **Waktu:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            await status_msg.edit(report)
            logger.info(f"Admin sync completed: {sync_result}")
            
        except Exception as e:
            logger.error(f"Error in update_loras_command: {e}", exc_info=True)
            await status_msg.edit(f"❌ Terjadi kesalahan:\n`{str(e)[:200]}`")
    
    async def handle_message(self, event):
        """Handle incoming messages for image generation"""
        status_message = None
        
        try:
            # Send typing action
            async with self.client.action(event.chat_id, 'typing'):
                # Extract prompt and check for photo
                prompt_text = event.message.text or event.message.message or ""
                has_photo = bool(event.message.photo)
                
                if not prompt_text.strip():
                    await event.respond("📝 Silakan berikan deskripsi gambar yang ingin Anda buat!")
                    return
                
                # Initial status
                status_message = await event.respond("🤖 Menganalisis prompt Anda...")
                
                # Analyze with Cerebras
                analysis = await self.cerebras.analyze_and_select_tools(
                    message=prompt_text,
                    has_image=has_photo
                )
                
                # Prepare generation parameters
                params = {
                    "prompt": analysis.get("enhanced_prompt", prompt_text),
                    "num_steps": 35,
                    "lora_name": analysis.get("selected_lora")
                }
                
                # Handle image-to-image
                if has_photo:
                    await status_message.edit("📥 Memproses gambar yang diunggah...")
                    
                    # Download photo
                    photo_bytes = await event.message.download_media(file=bytes)
                    params["init_image"] = base64.b64encode(photo_bytes).decode('utf-8')
                    params["strength"] = 0.75
                
                # Update status
                lora_info = f"LoRA: {params['lora_name']}" if params['lora_name'] else "Model default"
                await status_message.edit(f"🎨 Membuat gambar...\n{lora_info}")
                
                # Generate image
                result = await self.modal.generate_image(params)
                
                # Decode image
                image_data = base64.b64decode(result["image"])
                
                # Prepare caption
                caption = (
                    f"✨ **Hasil Gambar**\n\n"
                    f"**Prompt:**\n`{result.get('prompt', 'N/A')[:200]}{'...' if len(result.get('prompt', '')) > 200 else ''}`\n\n"
                    f"**LoRA:** `{result.get('lora', 'None')}`\n"
                    f"**Seed:** `{result.get('seed', 'N/A')}`\n"
                    f"**Steps:** `{result.get('num_steps', 'N/A')}`"
                )
                
                # Send photo
                await self.client.send_file(
                    event.chat_id,
                    file=image_data,
                    caption=caption,
                    force_document=False
                )
                
                # Delete status message
                await status_message.delete()
                logger.info(f"✓ Successfully generated image for user {event.sender_id}")
                
        except asyncio.TimeoutError:
            error_msg = "⏱️ Permintaan timeout. Server membutuhkan waktu terlalu lama. Silakan coba lagi."
            logger.error("Request timeout in handle_message")
            await self._send_error_message(status_message, event, error_msg)
            
        except aiohttp.ClientError as e:
            error_msg = f"🌐 Kesalahan koneksi ke server: {str(e)[:100]}"
            logger.error(f"API client error: {e}")
            await self._send_error_message(status_message, event, error_msg)
            
        except Exception as e:
            error_msg = "❌ Terjadi kesalahan saat memproses permintaan. Silakan coba lagi."
            logger.error(f"Error in handle_message: {e}", exc_info=True)
            await self._send_error_message(status_message, event, error_msg)
    
    async def _send_error_message(self, status_message, event, error_text: str):
        """Helper to send error messages"""
        try:
            if status_message:
                await status_message.edit(error_text)
            else:
                await event.respond(error_text)
        except Exception as e:
            logger.error(f"Failed to send error message: {e}")
    
    async def shutdown(self):
        """Clean shutdown"""
        logger.info("🔄 Shutting down bot...")
        await self.modal.close()
        await database.close_pool()
        logger.info("👋 Bot shutdown complete.")

# ===================================================================
# MAIN ENTRY POINT
# ===================================================================
async def main():
    """Main entry point"""
    logger.info("🚀 Starting Telethon bot...")
    
    # Create Telethon client
    client = TelegramClient(
        'bot_session',
        Config.API_ID,
        Config.API_HASH
    ).start(bot_token=Config.BOT_TOKEN)
    
    # Create bot instance
    bot = SmartImageBot(client)
    
    try:
        # Initialize bot
        await bot.start()
        
        logger.info("✅ Bot is running and ready to receive messages!")
        logger.info("Press Ctrl+C to stop the bot.")
        
        # Run until disconnected
        await client.run_until_disconnected()
        
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 Bot stopped by user.")
    except Exception as e:
        logger.critical(f"💥 A critical error occurred: {e}", exc_info=True)
    finally:
        await bot.shutdown()
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
