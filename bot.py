"""
Smart Telegram Bot for CivitAI Image Generation
Powered by Cerebras AI - No Command Handlers Required
"""

import os
import io
import json
import base64
import logging
import asyncio
import aiohttp
from typing import Optional, Dict, Any, List
from datetime import datetime

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from cerebras.cloud.sdk import Cerebras

# ===================================================================
# LOGGING SETUP
# ===================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================================================================
# CONFIGURATION
# ===================================================================
class Config:
    TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
    MODAL_API_URL = os.environ.get("MODAL_API_URL")
    MODAL_API_KEY = os.environ.get("MODAL_API_KEY")
    
    MAX_MESSAGE_LENGTH = 4096
    REQUEST_TIMEOUT = 180
    
    # User context expiration (minutes)
    CONTEXT_EXPIRY = 30

# ===================================================================
# CEREBRAS AI CLIENT
# ===================================================================
class CerebrasAI:
    def __init__(self):
        self.client = Cerebras(api_key=Config.CEREBRAS_API_KEY)
        self.system_prompt = """You are an expert AI assistant for an image generation bot. Your role is to:

1. **Understand User Intent**: Analyze messages to determine if user wants to:
   - Generate new image (text-to-image)
   - Modify existing image (image-to-image)
   - Use ControlNet features
   - Browse/select LoRA models
   - Get help/information

2. **Enhance Prompts**: When user provides image description:
   - Keep the CORE concept intact
   - Add technical quality enhancers
   - Improve clarity and detail
   - Stay within original context
   - Don't change subject, theme, or style unless explicitly requested

3. **Response Format**: Return JSON with this structure:
{
  "intent": "t2i|i2i|controlnet|browse_lora|help|chat",
  "enhanced_prompt": "enhanced version if intent is generation",
  "reasoning": "brief explanation",
  "confidence": 0.0-1.0,
  "suggested_params": {
    "lora": "suggested LoRA name or null",
    "steps": 25-50,
    "guidance_scale": 7-15
  }
}

4. **Rules**:
   - NEVER change the main subject/theme
   - Don't add unwanted elements
   - Maintain user's creative vision
   - Be concise and helpful
   - If unsure, ask for clarification"""

    async def analyze_message(
        self, 
        message: str, 
        has_image: bool = False,
        conversation_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """Analyze user message and determine intent"""
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add conversation history
            if conversation_history:
                messages.extend(conversation_history[-5:])  # Last 5 messages
            
            # Add current message with context
            context = f"User message: {message}\n"
            context += f"Has attached image: {has_image}\n"
            context += "Analyze intent and enhance if needed."
            
            messages.append({"role": "user", "content": context})
            
            response = self.client.chat.completions.create(
                messages=messages,
                model="gpt-oss-120b",
                temperature=0.7,
                max_completion_tokens=1024,
                reasoning_effort="medium"
            )
            
            result_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                # Extract JSON from markdown if present
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0]
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0]
                
                result = json.loads(result_text.strip())
                return result
            except json.JSONDecodeError:
                # Fallback parsing
                logger.warning("Failed to parse JSON, using fallback")
                return {
                    "intent": "chat",
                    "enhanced_prompt": None,
                    "reasoning": result_text,
                    "confidence": 0.5,
                    "suggested_params": {}
                }
                
        except Exception as e:
            logger.error(f"Cerebras analysis error: {e}")
            return {
                "intent": "error",
                "reasoning": str(e),
                "confidence": 0.0
            }

    async def enhance_prompt_stream(self, prompt: str) -> str:
        """Stream enhanced prompt generation"""
        try:
            stream = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a prompt enhancement expert. Enhance the given prompt for image generation while keeping the core concept intact. Add quality tags and details but DON'T change the main subject."
                    },
                    {
                        "role": "user",
                        "content": f"Enhance this prompt: {prompt}"
                    }
                ],
                model="gpt-oss-120b",
                stream=True,
                temperature=0.8,
                max_completion_tokens=512
            )
            
            enhanced = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    enhanced += chunk.choices[0].delta.content
            
            return enhanced.strip()
            
        except Exception as e:
            logger.error(f"Stream enhancement error: {e}")
            return prompt

# ===================================================================
# MODAL API CLIENT
# ===================================================================
class ModalAPIClient:
    def __init__(self):
        self.base_url = Config.MODAL_API_URL.rstrip('/')
        self.api_key = Config.MODAL_API_KEY
        self.session = None
    
    async def get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    async def get_loras(self, page: int = 1, limit: int = 50) -> Dict[str, Any]:
        """Fetch available LoRA models"""
        try:
            session = await self.get_session()
            async with session.get(
                f"{self.base_url}/loras",
                params={"page": page, "limit": limit},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Failed to fetch LoRAs: {e}")
            return {"error": str(e)}
    
    async def text_to_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image from text"""
        try:
            params["api_key"] = self.api_key
            
            session = await self.get_session()
            async with session.post(
                f"{self.base_url}/text2img",
                json=params,
                timeout=aiohttp.ClientTimeout(total=Config.REQUEST_TIMEOUT)
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Text-to-image error: {e}")
            return {"error": str(e)}
    
    async def image_to_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Edit image with prompt"""
        try:
            params["api_key"] = self.api_key
            
            session = await self.get_session()
            async with session.post(
                f"{self.base_url}/img2img",
                json=params,
                timeout=aiohttp.ClientTimeout(total=Config.REQUEST_TIMEOUT)
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Image-to-image error: {e}")
            return {"error": str(e)}
    
    async def controlnet(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate with ControlNet"""
        try:
            params["api_key"] = self.api_key
            
            session = await self.get_session()
            async with session.post(
                f"{self.base_url}/controlnet",
                json=params,
                timeout=aiohttp.ClientTimeout(total=Config.REQUEST_TIMEOUT)
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"ControlNet error: {e}")
            return {"error": str(e)}

# ===================================================================
# USER CONTEXT MANAGER
# ===================================================================
class UserContextManager:
    def __init__(self):
        self.contexts = {}
    
    def get_context(self, user_id: int) -> Dict[str, Any]:
        """Get user context"""
        if user_id not in self.contexts:
            self.contexts[user_id] = {
                "conversation_history": [],
                "last_image": None,
                "current_mode": None,
                "selected_lora": None,
                "preferences": {},
                "timestamp": datetime.now()
            }
        return self.contexts[user_id]
    
    def update_context(self, user_id: int, updates: Dict[str, Any]):
        """Update user context"""
        context = self.get_context(user_id)
        context.update(updates)
        context["timestamp"] = datetime.now()
    
    def add_message_to_history(self, user_id: int, role: str, content: str):
        """Add message to conversation history"""
        context = self.get_context(user_id)
        context["conversation_history"].append({
            "role": role,
            "content": content
        })
        # Keep only last 10 messages
        if len(context["conversation_history"]) > 10:
            context["conversation_history"] = context["conversation_history"][-10:]

# ===================================================================
# SMART BOT HANDLER
# ===================================================================
class SmartImageBot:
    def __init__(self):
        self.cerebras = CerebrasAI()
        self.modal = ModalAPIClient()
        self.context_manager = UserContextManager()
        self.lora_cache = None
        self.last_lora_fetch = None
    
    async def get_lora_list(self, force_refresh: bool = False) -> List[str]:
        """Get cached LoRA list"""
        now = datetime.now()
        
        # Refresh cache every 10 minutes or on force
        if (force_refresh or 
            self.lora_cache is None or 
            self.last_lora_fetch is None or
            (now - self.last_lora_fetch).seconds > 600):
            
            result = await self.modal.get_loras(page=1, limit=100)
            if "loras" in result:
                self.lora_cache = result["loras"]
                self.last_lora_fetch = now
        
        return self.lora_cache or []
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Main message handler - smart detection"""
        try:
            user_id = update.effective_user.id
            message_text = update.message.text or ""
            has_image = bool(update.message.photo)
            
            # Get user context
            user_context = self.context_manager.get_context(user_id)
            
            # Show typing indicator
            await update.message.chat.send_action("typing")
            
            # Analyze message with Cerebras
            analysis = await self.cerebras.analyze_message(
                message=message_text,
                has_image=has_image,
                conversation_history=user_context["conversation_history"]
            )
            
            logger.info(f"User {user_id} - Intent: {analysis['intent']} - Confidence: {analysis.get('confidence', 0)}")
            
            # Add to conversation history
            self.context_manager.add_message_to_history(user_id, "user", message_text)
            
            # Route based on intent
            intent = analysis.get("intent", "chat")
            
            if intent == "t2i":
                await self.handle_text_to_image(update, context, analysis)
            
            elif intent == "i2i":
                await self.handle_image_to_image(update, context, analysis)
            
            elif intent == "controlnet":
                await self.handle_controlnet(update, context, analysis)
            
            elif intent == "browse_lora":
                await self.handle_browse_lora(update, context)
            
            elif intent == "help":
                await self.handle_help(update, context)
            
            else:  # chat or low confidence
                await self.handle_chat(update, context, analysis)
            
        except Exception as e:
            logger.error(f"Message handler error: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ An error occurred: {str(e)}\n\n"
                "Please try again or contact support."
            )
    
    async def handle_text_to_image(
        self, 
        update: Update, 
        context: ContextTypes.DEFAULT_TYPE,
        analysis: Dict[str, Any]
    ):
        """Handle text-to-image generation"""
        try:
            user_id = update.effective_user.id
            enhanced_prompt = analysis.get("enhanced_prompt", update.message.text)
            
            # Show progress
            progress_msg = await update.message.reply_text(
                "🎨 **Generating your image...**\n\n"
                f"📝 Prompt: {enhanced_prompt[:100]}...\n"
                "⏳ This may take 30-60 seconds...",
                parse_mode="Markdown"
            )
            
            # Prepare parameters
            params = {
                "prompt": enhanced_prompt,
                "num_steps": analysis.get("suggested_params", {}).get("steps", 30),
                "guidance_scale": analysis.get("suggested_params", {}).get("guidance_scale", 7.5),
                "width": 1024,
                "height": 1024,
                "enhance_prompt": False  # Already enhanced by Cerebras
            }
            
            # Add LoRA if suggested or user has preference
            user_context = self.context_manager.get_context(user_id)
            lora_name = (analysis.get("suggested_params", {}).get("lora") or 
                        user_context.get("selected_lora"))
            
            if lora_name:
                params["lora_name"] = lora_name
                params["lora_scale"] = 0.8
            
            # Generate image
            result = await self.modal.text_to_image(params)
            
            if "error" in result:
                await progress_msg.edit_text(f"❌ Generation failed: {result['error']}")
                return
            
            # Send image
            image_data = base64.b64decode(result["image"])
            
            caption = (
                f"✅ **Generated Successfully**\n\n"
                f"📝 Original: {update.message.text[:100]}...\n"
                f"✨ Enhanced: {enhanced_prompt[:100]}...\n"
            )
            
            if lora_name:
                caption += f"🎨 LoRA: {lora_name}\n"
            
            caption += f"\n🔧 Steps: {params['num_steps']} | CFG: {params['guidance_scale']}"
            
            await update.message.reply_photo(
                photo=io.BytesIO(image_data),
                caption=caption,
                parse_mode="Markdown"
            )
            
            await progress_msg.delete()
            
            # Update context
            self.context_manager.update_context(user_id, {
                "last_image": result["image"],
                "current_mode": "t2i"
            })
            
        except Exception as e:
            logger.error(f"T2I handler error: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def handle_image_to_image(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        analysis: Dict[str, Any]
    ):
        """Handle image-to-image transformation"""
        try:
            user_id = update.effective_user.id
            
            # Get image
            if update.message.photo:
                photo = await update.message.photo[-1].get_file()
                image_bytes = await photo.download_as_bytearray()
                image_b64 = base64.b64encode(bytes(image_bytes)).decode()
            else:
                # Use last image from context
                user_context = self.context_manager.get_context(user_id)
                image_b64 = user_context.get("last_image")
                
                if not image_b64:
                    await update.message.reply_text(
                        "📸 Please send an image to modify, or generate one first!"
                    )
                    return
            
            enhanced_prompt = analysis.get("enhanced_prompt", update.message.text or "improve quality")
            
            # Show progress
            progress_msg = await update.message.reply_text(
                "🖼️ **Modifying your image...**\n\n"
                f"📝 Instruction: {enhanced_prompt[:100]}...\n"
                "⏳ Processing...",
                parse_mode="Markdown"
            )
            
            # Prepare parameters
            params = {
                "init_image": image_b64,
                "prompt": enhanced_prompt,
                "strength": 0.75,
                "num_steps": 30,
                "guidance_scale": 7.5,
                "preserve_aspect_ratio": True,
                "enhance_prompt": False
            }
            
            # Generate
            result = await self.modal.image_to_image(params)
            
            if "error" in result:
                await progress_msg.edit_text(f"❌ Failed: {result['error']}")
                return
            
            # Send result
            image_data = base64.b64decode(result["image"])
            
            caption = (
                f"✅ **Image Modified**\n\n"
                f"📝 Changes: {enhanced_prompt[:100]}...\n"
                f"💪 Strength: {params['strength']}\n"
                f"🔧 Steps: {params['num_steps']}"
            )
            
            await update.message.reply_photo(
                photo=io.BytesIO(image_data),
                caption=caption,
                parse_mode="Markdown"
            )
            
            await progress_msg.delete()
            
            # Update context
            self.context_manager.update_context(user_id, {
                "last_image": result["image"],
                "current_mode": "i2i"
            })
            
        except Exception as e:
            logger.error(f"I2I handler error: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def handle_controlnet(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        analysis: Dict[str, Any]
    ):
        """Handle ControlNet generation"""
        # Show ControlNet type selection
        keyboard = [
            [
                InlineKeyboardButton("🕺 OpenPose", callback_data="cn_openpose"),
                InlineKeyboardButton("📐 Canny", callback_data="cn_canny")
            ],
            [InlineKeyboardButton("🌊 Depth", callback_data="cn_depth")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🎮 **ControlNet Mode**\n\n"
            "Select the type of control:\n"
            "• OpenPose: Pose/body structure\n"
            "• Canny: Edge detection\n"
            "• Depth: Depth map\n\n"
            "Send an image after selection.",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
    
    async def handle_browse_lora(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        """Show LoRA browser"""
        try:
            loras = await self.get_lora_list()
            
            if not loras:
                await update.message.reply_text("❌ No LoRAs available")
                return
            
            # Create paginated keyboard (10 per page)
            page_size = 10
            keyboard = []
            
            for i, lora in enumerate(loras[:page_size]):
                keyboard.append([
                    InlineKeyboardButton(
                        f"🎨 {lora[:40]}", 
                        callback_data=f"lora_select_{lora}"
                    )
                ])
            
            if len(loras) > page_size:
                keyboard.append([
                    InlineKeyboardButton("Next ➡️", callback_data="lora_page_2")
                ])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                f"🎨 **Available LoRA Models**\n\n"
                f"Total: {len(loras)}\n"
                f"Showing: 1-{min(page_size, len(loras))}\n\n"
                "Select a LoRA to use in your next generation:",
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
            
        except Exception as e:
            logger.error(f"Browse LoRA error: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def handle_help(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        """Show help information"""
        help_text = """
🤖 **Smart Image Generation Bot**

I understand natural language! Just tell me what you want:

**Text-to-Image:**
• "Create a sunset over mountains"
• "Generate a cyberpunk cityscape"
• "Make a portrait of a cat in space"

**Image-to-Image:**
• Send an image + "make it anime style"
• Send an image + "turn this into watercolor"
• "Improve the colors of my last image"

**Features:**
✨ Natural language understanding
🎨 200+ LoRA style models
🎮 ControlNet support
🧠 Powered by Cerebras AI
🚀 Fast generation (30-60 sec)

**Tips:**
• Be descriptive but natural
• I'll enhance your prompts automatically
• Send images to modify them
• Browse LoRAs: "show me available styles"

**No commands needed - just chat naturally!**
        """
        
        await update.message.reply_text(help_text, parse_mode="Markdown")
    
    async def handle_chat(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        analysis: Dict[str, Any]
    ):
        """Handle general chat/unclear intent"""
        reasoning = analysis.get("reasoning", "")
        confidence = analysis.get("confidence", 0)
        
        if confidence < 0.3:
            # Very unclear - ask for clarification
            response = (
                "🤔 I'm not quite sure what you'd like to do.\n\n"
                "You can:\n"
                "• Describe an image to generate\n"
                "• Send an image to modify\n"
                "• Ask to browse LoRA styles\n"
                "• Ask for help\n\n"
                "What would you like to create?"
            )
        else:
            # Provide contextual response
            response = reasoning
        
        await update.message.reply_text(response)
        
        # Add to conversation history
        self.context_manager.add_message_to_history(
            update.effective_user.id,
            "assistant",
            response
        )
    
    async def handle_callback(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle inline button callbacks"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        user_id = update.effective_user.id
        
        try:
            if data.startswith("lora_select_"):
                lora_name = data.replace("lora_select_", "")
                self.context_manager.update_context(user_id, {
                    "selected_lora": lora_name
                })
                await query.edit_message_text(
                    f"✅ **LoRA Selected**\n\n"
                    f"🎨 {lora_name}\n\n"
                    "This LoRA will be used in your next generation!\n"
                    "Now describe what you want to create.",
                    parse_mode="Markdown"
                )
            
            elif data.startswith("lora_page_"):
                page = int(data.split("_")[-1])
                await self.show_lora_page(query, page)
            
            elif data.startswith("cn_"):
                cn_type = data.replace("cn_", "")
                self.context_manager.update_context(user_id, {
                    "controlnet_type": cn_type
                })
                await query.edit_message_text(
                    f"✅ **ControlNet: {cn_type.upper()}**\n\n"
                    "Now send an image and describe what you want to create.",
                    parse_mode="Markdown"
                )
        
        except Exception as e:
            logger.error(f"Callback error: {e}")
            await query.edit_message_text(f"❌ Error: {str(e)}")
    
    async def show_lora_page(self, query, page: int):
        """Show paginated LoRA list"""
        loras = await self.get_lora_list()
        page_size = 10
        start = (page - 1) * page_size
        end = start + page_size
        
        keyboard = []
        for lora in loras[start:end]:
            keyboard.append([
                InlineKeyboardButton(
                    f"🎨 {lora[:40]}", 
                    callback_data=f"lora_select_{lora}"
                )
            ])
        
        # Navigation buttons
        nav_buttons = []
        if page > 1:
            nav_buttons.append(
                InlineKeyboardButton("⬅️ Prev", callback_data=f"lora_page_{page-1}")
            )
        if end < len(loras):
            nav_buttons.append(
                InlineKeyboardButton("Next ➡️", callback_data=f"lora_page_{page+1}")
            )
        
        if nav_buttons:
            keyboard.append(nav_buttons)
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            f"🎨 **LoRA Models - Page {page}**\n\n"
            f"Showing: {start+1}-{min(end, len(loras))} of {len(loras)}",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

# ===================================================================
# MAIN APPLICATION
# ===================================================================
async def post_init(application: Application):
    """Post initialization"""
    logger.info("🤖 Bot initialized successfully")
    logger.info("🧠 Cerebras AI: Ready")
    logger.info("🚀 Modal API: Connected")
    logger.info("✨ Smart mode: Active (no commands needed)")

async def post_shutdown(application: Application):
    """Cleanup on shutdown"""
    bot = application.bot_data["bot_instance"]
    await bot.modal.close()
    logger.info("👋 Bot shutdown complete")

def main():
    """Main entry point"""
    # Validate environment variables
    required_vars = [
        "TELEGRAM_BOT_TOKEN",
        "CEREBRAS_API_KEY", 
        "MODAL_API_URL",
        "MODAL_API_KEY"
    ]
    
    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        logger.error(f"❌ Missing environment variables: {', '.join(missing)}")
        return
    
    logger.info("="*70)
    logger.info("🤖 SMART IMAGE GENERATION BOT")
    logger.info("🧠 Powered by Cerebras AI")
    logger.info("="*70)
    
    # Create bot instance
    bot = SmartImageBot()
    
    # Create application
    application = (
        Application.builder()
        .token(Config.TELEGRAM_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    
    # Store bot instance
    application.bot_data["bot_instance"] = bot
    
    # Add handlers - NO COMMAND HANDLERS, only message handlers
    application.add_handler(
        MessageHandler(
            filters.TEXT | filters.PHOTO,
            bot.handle_message
        )
    )
    
    application.add_handler(
        CallbackQueryHandler(bot.handle_callback)
    )
    
    # Start bot
    logger.info("🚀 Starting bot polling...")
    logger.info("💬 Send any message to begin - no commands needed!")
    
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True
    )

if __name__ == "__main__":
    main()