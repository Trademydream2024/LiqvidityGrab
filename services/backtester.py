import pandas as pd, numpy as np
from utils.time_windows import mask_session, SESSIONS
from .strategy import find_fractals, detect_sweep, compute_ob_entry

from typing import List, Dict
# אם get_session_mask אצלך בקובץ אחר – השאר כפי שהיה.
# ננסה לייבא אם קיימת פונקציה בקוד שלך; אחרת נשתמש ב-Fallback
try:
    from services.session_stats import get_session_mask as _get_session_mask
except Exception:
    _get_session_mask = None

import pandas as pd

def get_session_mask(df: pd.DataFrame, tz: str, session_name: str):
    """
    אם קיימת אצלך פונקציה מקורית – נשתמש בה.
    אחרת: מימוש Fallback עם חלונות זמן טיפוסיים לסשנים.
    """
    if _get_session_mask is not None:
        return _get_session_mask(df, tz, session_name)

    # --- Fallback: חלונות זמנים טיפוסיים ---
    # הופך את האינדקס לזמן מקומי
    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    try:
        local = idx.tz_convert(tz)
    except Exception:
        local = idx  # אם TZ לא תקין – נשארים ב-UTC

    hm = local.hour * 60 + local.minute  # דקות מהתחלת היום

    def in_window(start_hm: int, end_hm: int):
        # תומך גם בחלונות עוברים-חצות (למשל 21:00–06:00)
        if start_hm <= end_hm:
            return (hm >= start_hm) & (hm < end_hm)
        else:
            return (hm >= start_hm) | (hm < end_hm)

    # מיפוי בסיסי לסשנים הנפוצים אצלך
    windows = {
        "NY":           (13*60, 22*60),  # 13:00–22:00
        "London":       (7*60, 16*60),   # 07:00–16:00
        "Asia":         (0,     9*60),   # 00:00–09:00
        "Sydney":       (21*60, 6*60),   # 21:00–06:00 (עובר חצות)
        "Crypto_Peak":  (12*60, 20*60),  # 12:00–20:00
        "Crypto_Asia":  (2*60,  10*60),  # 02:00–10:00
        "Crypto_US":    (14*60, 22*60),  # 14:00–22:00
    }
    start, end = windows.get(session_name, (0, 24*60))
    mask_arr = in_window(start, end)
    return pd.Series(mask_arr, index=df.index)



def backtest(df: pd.DataFrame, tz, session_name: str = "NY",
             span: int = 2, wick_ratio: float = 1.5, r_mult: float = 1.8, vol_mult: float = 1.2) -> Dict:
    """
    Backtest מבוסס LSRR (Liquidity Sweep + Return to Range).
    שומר את החתימה הישנה כדי לא לשבור קריאות קיימות.
    """
    # 1) מסיכת סשן
    session_mask = get_session_mask(df, tz, session_name)

    # 2) יצירת סיגנלים
    sigs = detect_lsrr_signals(
        df,
        session_mask=session_mask,
        lookback_bars_primary=180,   # ~3 שעות
        lookback_bars_daily=1440,    # יום שלם
        wick_atr_min=0.6,            # אם מעט — תרד ל-0.4-0.5
        body_atr_min=0.5,
        use_fvg=True,                # אפשר False להגדלת כמות
        vol_mult=vol_mult            # שולט על דרישת נפח
    )

    # 3) סימולציית טריידים: כניסה בבר הבא, SL לפי הסוויפ, TP לפי R-multiple
    trades: List[Dict] = []
    for s in sigs:
        i = df.index.get_indexer([s["time"]], method="backfill")[0]
        if i + 1 >= len(df):
            continue

        dir_  = s["dir"]
        entry = float(s["entry"])
        sl    = float(s["sl"])

        if dir_ == "long":
            risk = max(1e-9, entry - sl)
            tp   = entry + r_mult * risk
        else:
            risk = max(1e-9, sl - entry)
            tp   = entry - r_mult * risk

        hit = None
        last_j = i + 1
        for j in range(i+1, min(i+1+60*12, len(df))):  # עד 12 שעות קדימה
            last_j = j
            h, l = float(df["high"].iloc[j]), float(df["low"].iloc[j])
            if dir_ == "long":
                if l <= sl:   hit = ("SL", df.index[j]); break
                if h >= tp:   hit = ("TP", df.index[j]); break
            else:
                if h >= sl:   hit = ("SL", df.index[j]); break
                if l <= tp:   hit = ("TP", df.index[j]); break

        if not hit:
            close_px = float(df["close"].iloc[last_j])
            pnl_r = (close_px - entry)/risk if dir_ == "long" else (entry - close_px)/risk
            trades.append({"dir": dir_, "entry_time": s["time"], "exit_time": df.index[last_j],
                           "entry": entry, "exit": close_px, "result": "TIMEOUT", "r": pnl_r, "note": s["note"]})
        else:
            tag, t_exit = hit
            exit_px = sl if tag == "SL" else tp
            pnl_r   = -1.0 if tag == "SL" else r_mult
            trades.append({"dir": dir_, "entry_time": s["time"], "exit_time": t_exit,
                           "entry": entry, "exit": float(exit_px), "result": tag, "r": pnl_r, "note": s["note"]})

    # 4) מדדים
    if not trades:
        return {"trades": 0, "winrate": 0.0, "pf": 0, "maxdd_r": 0, "list": [], "session": session_name}

    wins    = [t for t in trades if t["r"] > 0]
    losses  = [t for t in trades if t["r"] < 0]
    gross_w = sum(t["r"] for t in wins)
    gross_l = abs(sum(t["r"] for t in losses)) or 1e-9
    pf      = gross_w / gross_l
    winrate = len(wins) / max(1, len(trades))

    eq = 0.0
    peak = 0.0
    maxdd = 0.0
    for t in trades:
        eq += t["r"]
        peak = max(peak, eq)
        maxdd = max(maxdd, peak - eq)

    return {
        "trades": len(trades),
        "winrate": winrate,
        "pf": pf,
        "maxdd_r": maxdd,
        "list": trades,
        "session": session_name,
    }



def _atr(df: pd.DataFrame, n: int = 50) -> pd.Series:
    hl = (df["high"] - df["low"]).abs()
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = np.nanmax(np.vstack([hl.values, hc.values, lc.values]), axis=0)
    atr = pd.Series(tr, index=df.index).rolling(n, min_periods=n//2).mean()
    return atr

def _rolling_extrema(df: pd.DataFrame, lookback_bars: int = 180):
    hh = df["high"].shift(1).rolling(lookback_bars, min_periods=lookback_bars//2).max()
    ll = df["low"].shift(1).rolling(lookback_bars, min_periods=lookback_bars//2).min()
    return hh, ll

def _has_bear_fvg(df: pd.DataFrame, i: int) -> bool:
    # Bearish FVG: low[i+1] > high[i-1]
    if i - 1 < 0 or i + 1 >= len(df):
        return False
    return df["low"].iloc[i+1] > df["high"].iloc[i-1]

def _has_bull_fvg(df: pd.DataFrame, i: int) -> bool:
    # Bullish FVG: high[i+1] < low[i-1]
    if i - 1 < 0 or i + 1 >= len(df):
        return False
    return df["high"].iloc[i+1] < df["low"].iloc[i-1]

def detect_lsrr_signals(
    df: pd.DataFrame,
    *,
    session_mask: pd.Series,
    lookback_bars_primary: int = 180,     # ~3 שעות על 1m
    lookback_bars_daily: int   = 1440,    # יום
    wick_atr_min: float        = 0.6,     # חדירה מינימלית ביחס ל-ATR
    body_atr_min: float        = 0.5,     # גודל גוף נר הדחיפה
    use_fvg: bool              = True,
    vol_mult: float            = 1.2,     # ספייק נפח
) -> list[dict]:
    """
    מחזיר רשימת סיגנלים בפורמט:
    { 'time': ts, 'dir': 'short'|'long', 'entry': price, 'sl': price, 'note': str }
    """
    dff = df.loc[session_mask].copy()
    if dff.empty:
        return []

    atr = _atr(dff, 50).fillna(method="bfill")
    vol_ma = dff["volume"].rolling(200, min_periods=50).mean()

    hh_p, ll_p = _rolling_extrema(dff, lookback_bars_primary)
    hh_d, ll_d = _rolling_extrema(dff, lookback_bars_daily)

    out = []
    for i in range(3, len(dff)-2):
        o, h, l, c, v = dff["open"].iloc[i], dff["high"].iloc[i], dff["low"].iloc[i], dff["close"].iloc[i], dff["volume"].iloc[i]
        ts = dff.index[i]
        a  = max(1e-9, atr.iloc[i])
        vma = max(1e-9, vol_ma.iloc[i])

        # --- Bearish sweep: שובר HH קודם עם Wick וחוזר פנימה ---
        swept_hh = h > hh_p.iloc[i] * 1.0001 or h > hh_d.iloc[i] * 1.0001
        wick_up  = (h - max(o, c))
        body_dn  = max(0.0, (o - c))
        bear_ok  = (
            swept_hh and
            wick_up >= wick_atr_min * a and
            body_dn >= body_atr_min * a and
            (v >= vol_mult * vma)
        )
        if bear_ok:
            fvg_ok = (not use_fvg) or _has_bear_fvg(dff, i)
            if fvg_ok:
                # כניסה בשורט על ריטסט לאזור הסוויפ (mid בין HH לבין close)
                hh_level = max(hh_p.iloc[i], hh_d.iloc[i])
                entry = (hh_level + c) / 2.0
                sl    = h + 0.5 * a * 0.1  # מעט מעל הסווינג
                out.append({"time": ts, "dir": "short", "entry": float(entry), "sl": float(sl),
                            "note": f"Bear sweep HH→RTR at {ts}"})
                continue

        # --- Bullish sweep: שובר LL קודם עם Wick וחוזר פנימה ---
        swept_ll = l < ll_p.iloc[i] * 0.9999 or l < ll_d.iloc[i] * 0.9999
        wick_dn  = (min(o, c) - l)
        body_up  = max(0.0, (c - o))
        bull_ok  = (
            swept_ll and
            wick_dn >= wick_atr_min * a and
            body_up >= body_atr_min * a and
            (v >= vol_mult * vma)
        )
        if bull_ok:
            fvg_ok = (not use_fvg) or _has_bull_fvg(dff, i)
            if fvg_ok:
                ll_level = min(ll_p.iloc[i], ll_d.iloc[i])
                entry = (ll_level + c) / 2.0
                sl    = l - 0.5 * a * 0.1
                out.append({"time": ts, "dir": "long", "entry": float(entry), "sl": float(sl),
                            "note": f"Bull sweep LL→RTR at {ts}"})

    # הגבלה: טרייד אחד לכל יום בסשן (מונע ספאם)
    if len(out) > 1:
        chosen = []
        seen_dates = set()
        for s in out:
            d = pd.Timestamp(s["time"]).date()
            if d not in seen_dates:
                chosen.append(s)
                seen_dates.add(d)
        out = chosen
    return out
    