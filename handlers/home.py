# handlers/home.py
from aiogram import Router, F
from aiogram.types import Message, CallbackQuery
from keyboards.home import kb_home
from utils.cleaner import show_screen, replace_screen
from utils.state import set_mode

router = Router()

@router.message(F.text == "/start")
async def cmd_start(message: Message):
    set_mode(message.chat.id, "home")
    text = (
        "🤖 <b>NY Liquidity Bot</b>\n"
        "בחר פעולה:\n"
        "• 📥 הורד שנה אחורה לכל סימבול\n"
        "• 📊 הרץ בק־טסט + סטטיסטיקות סשנים\n"
        "• 🔧 בצע אופטימיזציה פר-סימבול\n"
        "• 🔴 עבור לפאנל לייב/התראות\n"
    )
    await replace_screen(message, text, kb_home())

@router.callback_query(F.data == "nav:home")
async def cb_home(call: CallbackQuery):
    set_mode(call.message.chat.id, "home")
    await call.answer()
    await show_screen(call.message.bot, call.message.chat.id, "🏠 מסך הבית", kb_home())
