# keyboards/symbols.py
import json, math
from pathlib import Path
from aiogram.utils.keyboard import InlineKeyboardBuilder
from config import CFG

ALL_SYMBOLS_FILE = Path("bybit_scan_results/all_symbols.json")

def _load_all_symbols() -> list[str]:
    # אם יש קובץ סריקה — נטען ממנו; אחרת ברירת מחדל
    symbols: list[str] = []
    try:
        if ALL_SYMBOLS_FILE.exists():
            data = json.loads(ALL_SYMBOLS_FILE.read_text(encoding="utf-8"))
            for item in data:
                s = (item.get("symbol") or "").upper()
                if s.endswith("USDT"):
                    symbols.append(s)
    except Exception:
        pass
    if not symbols:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]
    return sorted(set(symbols))

def _load_downloaded_symbols() -> list[str]:
    data_dir = Path(CFG.DATA_DIR)
    out = []
    if data_dir.exists():
        for p in data_dir.glob("*_1m.parquet"):
            name = p.name
            if name.endswith("_1m.parquet"):
                out.append(name.removesuffix("_1m.parquet"))  # ✅ פייתון 3.10
    return sorted(set(out))

def kb_symbols(action: str, page: int = 1, per_page: int = 12, only_downloaded: bool = False):
    # action ∈ {"dl","bt","opt"}
    syms = _load_downloaded_symbols() if only_downloaded else _load_all_symbols()
    total = max(1, math.ceil(len(syms)/per_page))
    page = max(1, min(page, total))
    start = (page-1)*per_page
    subset = syms[start:start+per_page]

    kb = InlineKeyboardBuilder()
    for s in subset:
        kb.button(text=s, callback_data=f"sym:sel:{action}:{s}:{page}")
    kb.adjust(3)  # 3 בשורה

    if total > 1:
        if page > 1:
            kb.button(text="⬅️", callback_data=f"sym:nav:{action}:{page-1}")
        kb.button(text=f"{page}/{total}", callback_data="sym:noop")
        if page < total:
            kb.button(text="➡️", callback_data=f"sym:nav:{action}:{page+1}")
        kb.adjust(3)

    kb.button(text="🏠 בית", callback_data="nav:home")
    kb.adjust(3)
    return kb.as_markup()
