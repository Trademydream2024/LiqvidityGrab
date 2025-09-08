import asyncio
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties

from config import CFG
from handlers.home import router as home_router
from handlers.data import router as data_router
from handlers.backtest import router as bt_router
from handlers.optimize import router as opt_router
from handlers.live import router as live_router

async def main():
    # יצירת אובייקט הבוט עם HTML כברירת מחדל
    bot = Bot(
        token=CFG.TELEGRAM_TOKEN,
        default=DefaultBotProperties(parse_mode="HTML")
    )

    dp = Dispatcher()

    # רישום ראוטרים (מודולרי)
    dp.include_router(home_router)
    dp.include_router(data_router)
    dp.include_router(bt_router)
    dp.include_router(opt_router)
    dp.include_router(live_router)

    # שמירת config בתוך ה-Dispatcher (נגיש ב- dp["cfg"])
    dp["cfg"] = CFG

    # התחלת פולינג
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
