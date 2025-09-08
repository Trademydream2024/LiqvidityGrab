# services/data_dl.py
import os, time, pandas as pd
from datetime import datetime, timedelta, timezone

ONE_MIN_MS = 60_000

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def download_year_1m(ex, symbol: str, data_dir: str, interval="1", progress_cb=None):
    """
    מוריד שנה 1m ל-symbol. אם progress_cb קיים, נקרא ככה:
      progress_cb(done_bars:int, total_bars:int)
    """
    ensure_dir(data_dir)
    end = int(datetime.now(timezone.utc).timestamp()*1000)
    start = end - 365*24*60*ONE_MIN_MS

    # הערכת כמות ברים לשנה (בקירוב — מספיקה לפרוגרס)
    total_bars = int((end - start) / ONE_MIN_MS)
    fetched_bars = 0

    out = []
    t = start
    last_pct = -1
    while t < end:
        res = ex.get_klines(symbol, interval, t, min(end, t+1000*ONE_MIN_MS))
        rows = res.get("result", {}).get("list", [])
        if not rows:
            break
        rows = rows[::-1]
        out += rows

        # התקדמות
        fetched_bars = len(out)
        if progress_cb:
            pct = min(100, int(fetched_bars / max(1, total_bars) * 100))
            if pct != last_pct:
                last_pct = pct
                try:
                    progress_cb(fetched_bars, total_bars)
                except Exception:
                    pass

        t = int(rows[-1][0]) + ONE_MIN_MS
        time.sleep(0.08)  # לרכך rate-limit

    if not out:
        return None

    df = pd.DataFrame(out, columns=["ts","open","high","low","close","volume","turnover"])
    df[["open","high","low","close","volume","turnover"]] = df[["open","high","low","close","volume","turnover"]].astype(float)
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    df = df.sort_values("ts")
    path = os.path.join(data_dir, f"{symbol}_1m.parquet")
    df.to_parquet(path, index=False)
    return path
