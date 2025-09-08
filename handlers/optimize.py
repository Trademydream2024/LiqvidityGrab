import os
import re
import pandas as pd
import asyncio
import itertools
from aiogram import Router, F
from aiogram.types import CallbackQuery, Message

from keyboards.common import kb_back_home
from utils.cleaner import show_screen, replace_screen
from utils.state import set_mode, get_mode
from services.optimizer import grid_optimize
from services.reporting import render_bt
from storage import init_db, save_result
from config import CFG
from keyboards.symbols import kb_symbols

router = Router()
init_db()

@router.callback_query(F.data == "nav:opt")
async def nav_opt(call: CallbackQuery):
    set_mode(call.message.chat.id, "opt")
    await call.answer()
    await show_screen(
        call.message.bot,
        call.message.chat.id,
        "🔧 אופטימיזציה — בחר סימבול מתוך הקבצים שהורדת:",
        kb_symbols("opt", page=1, only_downloaded=True)
    )


@router.message(F.text)
async def run_opt(message: Message):
    # בדיקה ראשונה - האם אנחנו במוד הנכון
    if get_mode(message.chat.id) != "opt":
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
            f"⚠️ אין קובץ עבור <b>{sym}</b>. הורד דאטה קודם.",
            kb_back_home()
        )

    m = await replace_screen(message, f"⏳ טוען קובץ עבור <b>{sym}</b>… 10%")
    
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return await replace_screen(message, f"❌ שגיאה בקריאת קובץ: <code>{e}</code>", kb_back_home())

    # הגדרת גריד
    spans = (2, 3)
    wicks = (1.3, 1.5, 1.8)
    rms = (1.5, 1.8, 2.0)
    volm = (1.0, 1.2, 1.5)
    
    grid = list(itertools.product(spans, wicks, rms, volm))
    total = len(grid)
    best = None

    await message.bot.edit_message_text(
        f"🔧 מריץ Grid על <b>{sym}</b>… 15%\n"
        f"📊 בוחן {total} קומבינציות...",
        chat_id=message.chat.id, 
        message_id=m.message_id, 
        reply_markup=kb_back_home(), 
        parse_mode="HTML"
    )

    from services.backtester import backtest
    done = 0
    
    for s, w, r, vm in grid:
        try:
            # חשוב! להעביר את הפרמטרים נכון ללמבדה
            metrics = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda s=s, w=w, r=r, vm=vm: backtest(
                    df, CFG.TZ, "NY", 
                    span=s, wick_ratio=w, r_mult=r, vol_mult=vm
                )
            )
            
            score = metrics["pf"] * metrics["winrate"] * max(1, metrics["trades"]/40)
            cand = {"span":s, "wick":w, "r":r, "volm":vm, **metrics, "score":round(score,3)}
            
            if not best or cand["score"] > best["score"]:
                best = cand

            done += 1
            
            # עדכון התקדמות
            if done % 3 == 0 or done == total:
                pct = 15 + int((done/total) * 85)
                
                status_text = f"🔧 מריץ Grid על <b>{sym}</b>… {pct}%\n"
                status_text += f"📊 נבדקו {done}/{total} קומבינציות\n\n"
                
                if best and best["trades"] > 0:
                    status_text += f"🏆 <b>כרגע מוביל:</b>\n"
                    status_text += f"• Span: {best['span']} bars\n"
                    status_text += f"• Wick Ratio: ≥{best['wick']}\n"
                    status_text += f"• R:R: 1:{best['r']}\n"
                    status_text += f"• Volume Filter: ≥{best['volm']}x avg\n\n"
                    status_text += f"📈 <b>ביצועים:</b>\n"
                    status_text += f"• Profit Factor: {best['pf']}\n"
                    status_text += f"• Win Rate: {best['winrate']*100:.1f}%\n"
                    status_text += f"• Trades: {best['trades']}\n"
                    status_text += f"• Score: {best['score']}"
                
                try:
                    await message.bot.edit_message_text(
                        status_text,
                        chat_id=message.chat.id, 
                        message_id=m.message_id,
                        reply_markup=kb_back_home(), 
                        parse_mode="HTML"
                    )
                except:
                    pass
                    
        except Exception as e:
            print(f"Error in optimization: {e}")
            continue

    if not best or best["trades"] == 0:
        await message.bot.edit_message_text(
            f"❌ לא נמצאו עסקאות עבור <b>{sym}</b>\n"
            "נסה סימבול אחר או בדוק שיש מספיק דאטה.",
            chat_id=message.chat.id, 
            message_id=m.message_id,
            reply_markup=kb_back_home(), 
            parse_mode="HTML"
        )
        return

    # שמירה
    params = {k: best[k] for k in ("span", "wick", "r", "volm")}
    
    try:
        save_result(sym, "NY", params, best)
    except Exception as e:
        print(f"Error saving: {e}")

    # תוצאות סופיות
    final_text = f"✅ <b>אופטימיזציה הסתיימה עבור {sym}!</b>\n\n"
    final_text += render_bt(sym, "NY", best, params)
    
    await message.bot.edit_message_text(
        final_text,
        chat_id=message.chat.id, 
        message_id=m.message_id,
        reply_markup=kb_back_home(), 
        parse_mode="HTML"
    )
    
@router.callback_query(F.data.startswith("sym:nav:opt:"))
async def cb_opt_nav(call: CallbackQuery):
    try:
        page = int(call.data.split(":")[-1])
    except:
        page = 1
    await call.answer()
    await show_screen(
        call.message.bot,
        call.message.chat.id,
        "🔧 אופטימיזציה — בחר סימבול מתוך הקבצים שהורדת:",
        kb_symbols("opt", page=page, only_downloaded=True)
    )

@router.callback_query(F.data.startswith("sym:sel:opt:"))
async def cb_opt_select(call: CallbackQuery):
    parts = call.data.split(":")
    sym = parts[3].upper()
    bot = call.message.bot
    chat_id = call.message.chat.id

    import os, pandas as pd, asyncio, itertools
    from services.optimizer import grid_optimize
    from services.reporting import render_bt
    from storage import save_result, init_db
    from keyboards.common import kb_back_home
    from config import CFG

    path = os.path.join(CFG.DATA_DIR, f"{sym}_1m.parquet")
    if not os.path.exists(path):
        await show_screen(bot, chat_id, f"⚠️ אין קובץ עבור <b>{sym}</b>. הורד דאטה קודם.", kb_back_home())
        await call.answer()
        return

    m = await show_screen(bot, chat_id, f"⏳ טוען קובץ עבור <b>{sym}</b>… 10%", kb_back_home())

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

    spans = (2, 3)
    wicks = (1.3, 1.5, 1.8)
    rms = (1.5, 1.8, 2.0)
    volm = (1.0, 1.2, 1.5)

    grid = list(itertools.product(spans, wicks, rms, volm))
    total = len(grid)
    best = None

    await bot.edit_message_text(
        f"🔧 מריץ Grid על <b>{sym}</b>… 15%\n"
        f"📊 בוחן {total} קומבינציות...",
        chat_id=chat_id, message_id=m.message_id,
        reply_markup=kb_back_home(), parse_mode="HTML"
    )

    done = 0
    for (s, w, r, vm) in grid:
        try:
            mtr = await asyncio.get_event_loop().run_in_executor(
                None, lambda: grid_optimize(df, CFG.TZ, "NY", (s,), (w,), (r,), (vm,))
            )
            if mtr and (not best or mtr["score"] > best["score"]):
                best = mtr
        except Exception:
            pass
        finally:
            done += 1
            if done % 3 == 0 or done == total:
                pct = 15 + int((done/total) * 85)
                status_text = f"🔧 מריץ Grid על <b>{sym}</b>… {pct}%\n"
                status_text += f"📊 נבדקו {done}/{total} קומבינציות\n\n"
                if best and best.get("trades",0) > 0:
                    status_text += "🏆 <b>כרגע מוביל:</b>\n"
                    status_text += f"• Span: {best['span']} bars\n"
                    status_text += f"• Wick Ratio: ≥{best['wick']}\n"
                    status_text += f"• R:R: 1:{best['r']}\n"
                    status_text += f"• Volume Filter: ≥{best['volm']}x avg\n\n"
                    status_text += "📈 <b>ביצועים:</b>\n"
                    status_text += f"• Profit Factor: {best['pf']}\n"
                    status_text += f"• Win Rate: {best['winrate']*100:.1f}%\n"
                    status_text += f"• Trades: {best['trades']}\n"
                    status_text += f"• Score: {best['score']}"
                try:
                    await bot.edit_message_text(
                        status_text, chat_id=chat_id, message_id=m.message_id,
                        reply_markup=kb_back_home(), parse_mode="HTML"
                    )
                except:
                    pass

    try:
        params = {"span": best["span"], "wick": best["wick"], "r": best["r"], "volm": best["volm"]}
        init_db()
        save_result(sym, "NY", params, best)
    except Exception:
        pass

    final_text = f"✅ <b>אופטימיזציה הסתיימה עבור {sym}!</b>\n\n" + render_bt(sym, "NY", best, params)
    await bot.edit_message_text(
        final_text, chat_id=chat_id, message_id=m.message_id,
        reply_markup=kb_back_home(), parse_mode="HTML"
    )
    await call.answer()

@router.callback_query(F.data == "sym:noop")
async def cb_noop(call: CallbackQuery):
    await call.answer()
    