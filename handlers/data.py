import asyncio
import re
from aiogram import Router, F
from aiogram.types import CallbackQuery, Message

from keyboards.common import kb_back_home
from utils.cleaner import show_screen, replace_screen
from utils.state import set_mode, get_mode
from services.exchange import BybitEx
from services.data_dl import download_year_1m
from config import CFG
from keyboards.symbols import kb_symbols


router = Router()

@router.callback_query(F.data == "nav:data")
async def nav_data(call: CallbackQuery):
    set_mode(call.message.chat.id, "data")
    await call.answer()
    await show_screen(
        call.message.bot,
        call.message.chat.id,
        "📥 הורדת דאטה לשנה אחורה\nבחר סימבול מהרשימה:",
        kb_symbols("dl", page=1, only_downloaded=False)
    )


@router.message(F.text)
async def on_symbol_download(message: Message):
    # בדיקה ראשונה - האם אנחנו במוד הנכון
    if get_mode(message.chat.id) != "data":
        return
    
    # בדיקה שניה - האם זה סימבול תקין
    if not re.match(r"^[A-Za-z0-9_\-]+USDT$", message.text.upper()):
        await message.reply("❌ פורמט לא תקין. שלח סימבול כמו: BTCUSDT")
        return

    sym = message.text.upper()
    m = await replace_screen(message, f"⏳ מוריד שנה 1m עבור <b>{sym}</b>… 0%", kb_back_home())

    ex = BybitEx(CFG.BYBIT_API_KEY, CFG.BYBIT_API_SECRET, CFG.BYBIT_TESTNET)

    loop = asyncio.get_running_loop()
    bot = message.bot
    chat_id = message.chat.id
    msg_id = m.message_id
    last_pct = -1

    def progress_cb(done: int, total: int):
        nonlocal last_pct
        pct = min(100, int(done / max(1, total) * 100))
        if pct != last_pct and (pct % 5 == 0 or pct == 100):
            last_pct = pct
            asyncio.run_coroutine_threadsafe(
                bot.edit_message_text(
                    f"⏳ מוריד שנה 1m עבור <b>{sym}</b>… {pct}%\n"
                    f"📊 הורדו {done:,} נרות",
                    chat_id=chat_id,
                    message_id=msg_id,
                    reply_markup=kb_back_home(),
                    parse_mode="HTML"
                ),
                loop
            )

    try:
        path = await loop.run_in_executor(
            None,
            lambda: download_year_1m(ex, sym, CFG.DATA_DIR, "1", progress_cb)
        )
    except Exception as e:
        await bot.edit_message_text(
            f"❌ שגיאה בהורדה: <code>{e}</code>",
            chat_id=chat_id,
            message_id=msg_id,
            reply_markup=kb_back_home(),
            parse_mode="HTML"
        )
        return

    if not path:
        await bot.edit_message_text(
            f"❌ לא נמצא דאטה עבור {sym}",
            chat_id=chat_id,
            message_id=msg_id,
            reply_markup=kb_back_home(),
            parse_mode="HTML"
        )
        return

    import os
    file_size = os.path.getsize(path) / (1024 * 1024)
    
    await bot.edit_message_text(
        f"✅ <b>הורדה הושלמה!</b>\n\n"
        f"📁 נשמר ב:\n<code>{path}</code>\n\n"
        f"📊 גודל קובץ: {file_size:.2f} MB\n"
        f"🏠 לחץ לחזרה לבית",
        chat_id=chat_id,
        message_id=msg_id,
        reply_markup=kb_back_home(),
        parse_mode="HTML"
    )

@router.callback_query(F.data.startswith("sym:nav:dl:"))
async def cb_data_nav(call: CallbackQuery):
    try:
        page = int(call.data.split(":")[-1])
    except:
        page = 1
    await call.answer()
    await show_screen(
        call.message.bot,
        call.message.chat.id,
        "📥 הורדת דאטה לשנה אחורה\nבחר סימבול מהרשימה:",
        kb_symbols("dl", page=page, only_downloaded=False)
    )

@router.callback_query(F.data.startswith("sym:sel:dl:"))
async def cb_data_select(call: CallbackQuery):
    parts = call.data.split(":")  # sym:sel:dl:<SYMBOL>:<PAGE>
    sym = parts[3].upper()
    bot = call.message.bot
    chat_id = call.message.chat.id

    m = await show_screen(bot, chat_id, f"⏳ מוריד שנה 1m עבור <b>{sym}</b>… 0%", kb_back_home())
    msg_id = m.message_id
    loop = asyncio.get_running_loop()
    last_pct = -1

    def progress_cb(done: int, total: int):
        nonlocal last_pct
        pct = min(100, int(done / max(1, total) * 100))
        if pct != last_pct and (pct % 5 == 0 or pct == 100):
            last_pct = pct
            asyncio.run_coroutine_threadsafe(
                bot.edit_message_text(
                    f"⏳ מוריד שנה 1m עבור <b>{sym}</b>… {pct}%\n"
                    f"📊 הורדו {done:,} נרות",
                    chat_id=chat_id,
                    message_id=msg_id,
                    reply_markup=kb_back_home(),
                    parse_mode="HTML"
                ),
                loop
            )

    ex = BybitEx(CFG.BYBIT_API_KEY, CFG.BYBIT_API_SECRET, CFG.BYBIT_TESTNET)
    try:
        path = await loop.run_in_executor(
            None,
            lambda: download_year_1m(ex, sym, CFG.DATA_DIR, "1", progress_cb)
        )
    except Exception as e:
        await bot.edit_message_text(
            f"❌ שגיאה בהורדה: <code>{e}</code>",
            chat_id=chat_id, message_id=msg_id,
            reply_markup=kb_back_home(), parse_mode="HTML"
        )
        await call.answer()
        return

    if not path:
        await bot.edit_message_text(
            f"❌ לא התקבל קובץ עבור <b>{sym}</b>.",
            chat_id=chat_id, message_id=msg_id,
            reply_markup=kb_back_home(), parse_mode="HTML"
        )
        await call.answer()
        return

    import os
    file_size = os.path.getsize(path) / (1024 * 1024)
    await bot.edit_message_text(
        f"✅ <b>הורדה הושלמה!</b>\n\n"
        f"📁 נשמר ב:\n<code>{path}</code>\n\n"
        f"📊 גודל קובץ: {file_size:.2f} MB\n"
        f"🏠 לחץ לחזרה לבית",
        chat_id=chat_id, message_id=msg_id,
        reply_markup=kb_back_home(), parse_mode="HTML"
    )
    await call.answer()

@router.callback_query(F.data == "sym:noop")
async def cb_noop(call: CallbackQuery):
    await call.answer()
    