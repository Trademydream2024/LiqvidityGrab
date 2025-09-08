import pandas as pd
from utils.time_windows import mask_session, SESSIONS

def session_movement_stats(df: pd.DataFrame, tz_name: str):
    df = df.copy()
    df["range_pct"] = (df["high"] - df["low"]) / df["open"] * 100.0
    out = {}
    for name, (start, end) in SESSIONS.items():
        m = mask_session(df["ts"], tz_name, start, end)
        sub = df[m]
        if len(sub) == 0:
            out[name] = {"bars":0,"avg_range_pct":0,"p95_range_pct":0}
            continue
        out[name] = {
            "bars": len(sub),
            "avg_range_pct": round(sub["range_pct"].mean(), 3),
            "p95_range_pct": round(sub["range_pct"].quantile(0.95), 3),
        }
    return out
