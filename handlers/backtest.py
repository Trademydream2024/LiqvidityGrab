import os
import re
import pandas as pd
import asyncio
from aiogram import Router, F
from aiogram.types import CallbackQuery, Message
from keyboards.common import kb_back_home
from utils.cleaner import show_screen, replace_screen
from utils.state import set_mode, get_mode
from services.session_stats import session_movement_stats
from services.backtester import backtest
from services.reporting import render_session_stats, render_bt
from config import CFG
from keyboards.symbols import kb_symbols


router = Router()

@router.callback_query(F.data == "nav:bt")
async def nav_bt(call: CallbackQuery):
    set_mode(call.message.chat.id, "bt")
    await call.answer()
    await show_screen(
        call.message.bot,
        call.message.chat.id,
        "📊 בק־טסט/סטטיסטיקות — בחר סימבול מתוך הקבצים שהורדת:",
        kb_symbols("bt", page=1, only_downloaded=True)
    )


@router.message(F.text)
async def run_bt(message: Message):
    # בדיקה ראשונה - האם אנחנו במוד הנכון
    if get_mode(message.chat.id) != "bt":
        return
    
    # בדיקה שניה - האם זה סימבול תקין
    if not re.match(r"^[A-Za-z0-9_\-]+USDT$", message.text.upper()):
        await message.reply("❌ פורמט לא תקין. שלח סימבול כמו: BTCUSDT")
        return

    sym = message.text.upper()
    path = os.path.join(CFG.DATA_DIR, f"{sym}_1m.parquet")
    
    if not os.path.exists(path):
        return await replace_screen(
            message,
            f"⚠️ אין קובץ עבור <b>{sym}</b>. עבור ל־📥 הורדת דאטה.",
            kb_back_home()
        )

    m = await replace_screen(message, f"📊 טוען קובץ עבור <b>{sym}</b>… 10%")
    
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return await replace_screen(message, f"❌ שגיאה בקריאת קובץ: <code>{e}</code>", kb_back_home())

    # סטטיסטיקות סשן
    await message.bot.edit_message_text(
        f"📊 מחשב סטטיסטיקות סשנים עבור <b>{sym}</b>… 50%",
        chat_id=message.chat.id, 
        message_id=m.message_id, 
        reply_markup=kb_back_home(),
        parse_mode="HTML"
    )
    
    try:
        stats = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: session_movement_stats(df, CFG.TZ)
        )
        txt = render_session_stats(sym, stats)
    except Exception as e:
        return await replace_screen(
            message, 
            f"❌ שגיאה בחישוב סטטיסטיקות: <code>{e}</code>", 
            kb_back_home()
        )
    
    # בק-טסט
    await message.bot.edit_message_text(
        txt + "\n\n⏳ מריץ בק־טסט NY… 80%",
        chat_id=message.chat.id, 
        message_id=m.message_id, 
        reply_markup=kb_back_home(), 
        parse_mode="HTML"
    )

    try:
        bt = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: backtest(df, CFG.TZ, session_name="NY", 
                           span=2, wick_ratio=1.5, r_mult=1.8, vol_mult=1.2)
        )
        
        final_text = txt + "\n\n" + render_bt(sym, "NY", bt)
        
        await message.bot.edit_message_text(
            final_text,
            chat_id=message.chat.id, 
            message_id=m.message_id, 
            reply_markup=kb_back_home(), 
            parse_mode="HTML"
        )
    except Exception as e:
        await message.bot.edit_message_text(
            txt + f"\n\n❌ שגיאה בבק־טסט: <code>{str(e)}</code>",
            chat_id=message.chat.id, 
            message_id=m.message_id, 
            reply_markup=kb_back_home(), 
            parse_mode="HTML"
        )

@router.callback_query(F.data.startswith("sym:nav:bt:"))
async def cb_bt_nav(call: CallbackQuery):
    try:
        page = int(call.data.split(":")[-1])
    except:
        page = 1
    await call.answer()
    await show_screen(
        call.message.bot,
        call.message.chat.id,
        "📊 בק־טסט/סטטיסטיקות — בחר סימבול מתוך הקבצים שהורדת:",
        kb_symbols("bt", page=page, only_downloaded=True)
    )

@router.callback_query(F.data.startswith("sym:sel:bt:"))
async def cb_bt_select(call: CallbackQuery):
    parts = call.data.split(":")
    sym = parts[3].upper()
    bot = call.message.bot
    chat_id = call.message.chat.id

    import os, pandas as pd, asyncio
    from services.session_stats import session_movement_stats
    from services.backtester import backtest
    from services.reporting import render_session_stats, render_bt
    from keyboards.common import kb_back_home
    from config import CFG

    path = os.path.join(CFG.DATA_DIR, f"{sym}_1m.parquet")
    if not os.path.exists(path):
        await show_screen(bot, chat_id, f"⚠️ אין קובץ עבור <b>{sym}</b>. עבור ל־📥 הורדת דאטה.", kb_back_home())
        await call.answer()
        return

    m = await show_screen(bot, chat_id, f"📊 טוען קובץ עבור <b>{sym}</b>… 10%", kb_back_home())

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        await bot.edit_message_text(
            f"❌ שגיאה בקריאת קובץ: <code>{e}</code>",
            chat_id=chat_id, message_id=m.message_id,
            reply_markup=kb_back_home(), parse_mode="HTML"
        )
        await call.answer()
        return

    await bot.edit_message_text(
        f"📊 מחשב סטטיסטיקות סשנים עבור <b>{sym}</b>… 50%",
        chat_id=chat_id, message_id=m.message_id,
        reply_markup=kb_back_home(), parse_mode="HTML"
    )

    try:
        stats = await asyncio.get_event_loop().run_in_executor(
            None, lambda: session_movement_stats(df, CFG.TZ)
        )
        txt = render_session_stats(sym, stats)
    except Exception as e:
        await bot.edit_message_text(
            f"❌ שגיאה בחישוב סטטיסטיקות: <code>{e}</code>",
            chat_id=chat_id, message_id=m.message_id,
            reply_markup=kb_back_home(), parse_mode="HTML"
        )
        await call.answer()
        return

    await bot.edit_message_text(
        txt + "\n\n⏳ מריץ בק־טסט NY… 80%",
        chat_id=chat_id, message_id=m.message_id,
        reply_markup=kb_back_home(), parse_mode="HTML"
    )

    try:
        bt = await asyncio.get_event_loop().run_in_executor(
            None, lambda: backtest(df, CFG.TZ, session_name="NY",
                                   span=2, wick_ratio=1.5, r_mult=1.8, vol_mult=1.2)
        )
        final_text = txt + "\n\n" + render_bt(sym, "NY", bt)
        await bot.edit_message_text(final_text, chat_id=chat_id, message_id=m.message_id,
                                    reply_markup=kb_back_home(), parse_mode="HTML")
    except Exception as e:
        await bot.edit_message_text(
            txt + f"\n\n❌ שגיאה בבק־טסט: <code>{str(e)}</code>",
            chat_id=chat_id, message_id=m.message_id,
            reply_markup=kb_back_home(), parse_mode="HTML"
        )
    await call.answer()

@router.callback_query(F.data == "sym:noop")
async def cb_noop(call: CallbackQuery):
    await call.answer()
        