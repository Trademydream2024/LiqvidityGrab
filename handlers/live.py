from aiogram import Router, F
from aiogram.types import CallbackQuery, Message
from utils.cleaner import show_screen
from keyboards.common import kb_back_home

router = Router()

# נשמור message_id של הפאנל כדי "לעדכן" אותו בלבד (ולא להציף את הבית)
_live_panel: dict[int, int] = {}

@router.callback_query(F.data == "nav:live")
async def nav_live(call: CallbackQuery):
    await call.answer()
    m = await show_screen(call.message.bot, call.message.chat.id, "🔴 פאנל לייב — אין עסקאות פעילות כרגע.", kb_back_home())
    _live_panel[call.message.chat.id] = m.message_id

# דוגמה לפונקציה שתעדכן את הפאנל (כשיהיו מילויים/סטטוסים):
async def live_update(bot, chat_id: int, text: str):
    mid = _live_panel.get(chat_id)
    if not mid:
        # אם אין פאנל פתוח, אפשר לשלוח התראת PUSH שקטה, אבל לא לגעת במסך הבית
        await bot.send_message(chat_id, f"🔔 {text}")
        return
    try:
        await bot.edit_message_text(text, chat_id, mid, reply_markup=kb_back_home())
    except:
        await bot.send_message(chat_id, f"🔔 {text}", reply_markup=kb_back_home())
