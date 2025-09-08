from aiogram import Bot
from aiogram.types import Message

# נשמור message_id אחרון לכל משתמש כדי למחוק/לעדכן
_last_msg: dict[int, int] = {}

async def show_screen(bot: Bot, chat_id: int, text: str, reply_markup=None):
    # מחיקת המסך הקודם (אם יש)
    if chat_id in _last_msg:
        try:
            await bot.delete_message(chat_id, _last_msg[chat_id])
        except: pass
    m = await bot.send_message(chat_id, text, reply_markup=reply_markup)
    _last_msg[chat_id] = m.message_id
    return m

async def replace_screen(message: Message, text: str, reply_markup=None):
    # עדכון במקום/מחיקה ושליחה מחדש אם צריך
    bot = message.bot
    chat_id = message.chat.id
    if chat_id in _last_msg:
        try:
            await bot.delete_message(chat_id, _last_msg[chat_id])
        except: pass
    m = await message.answer(text, reply_markup=reply_markup)
    _last_msg[chat_id] = m.message_id
    return m
