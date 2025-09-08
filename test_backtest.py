#!/usr/bin/env python3
"""
סורק מלא של Bybit - מוריד את כל הצמדים, דאטה ובק-טסט
"""

import os
import json
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
from pybit.unified_trading import HTTP
from services.backtester import backtest
from services.data_dl import download_year_1m
from services.exchange import BybitEx
from config import CFG

# Constants
OUTPUT_DIR = "bybit_scan_results"
SYMBOLS_FILE = f"{OUTPUT_DIR}/all_symbols.json"
RESULTS_FILE = f"{OUTPUT_DIR}/backtest_results.csv"
PROGRESS_FILE = f"{OUTPUT_DIR}/progress.json"

def ensure_dirs():
    """יצירת תיקיות"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/data", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/reports", exist_ok=True)

def get_all_symbols():
    """מקבל את כל הצמדים מ-Bybit"""
    print("🔍 מחפש את כל הצמדים ב-Bybit...")
    
    try:
        session = HTTP(testnet=False)
        
        # Linear perpetuals (USDT pairs)
        result = session.get_instruments_info(category="linear")
        
        symbols = []
        for item in result.get("result", {}).get("list", []):
            symbol = item.get("symbol", "")
            status = item.get("status", "")
            
            # רק צמדים פעילים עם USDT
            if status == "Trading" and symbol.endswith("USDT"):
                # פילטר צמדים עיקריים (לא leveraged tokens)
                if not any(x in symbol for x in ["2L", "3L", "2S", "3S", "UP", "DOWN"]):
                    symbols.append({
                        "symbol": symbol,
                        "base": symbol.replace("USDT", ""),
                        "quote": "USDT",
                        "min_price": float(item.get("priceFilter", {}).get("minPrice", 0)),
                        "max_price": float(item.get("priceFilter", {}).get("maxPrice", 0)),
                    })
        
        print(f"✅ נמצאו {len(symbols)} צמדים")
        
        # שמירה לקובץ
        with open(SYMBOLS_FILE, 'w') as f:
            json.dump(symbols, f, indent=2)
        
        return symbols
        
    except Exception as e:
        print(f"❌ שגיאה בקבלת צמדים: {e}")
        return []

def clean_data(df, symbol_info):
    """ניקוי דאטה לא תקין"""
    if len(df) == 0:
        return df
    
    # בדיקת טווח מחירים הגיוני
    max_reasonable = symbol_info.get("max_price", 1000000)
    min_reasonable = symbol_info.get("min_price", 0.00001)
    
    # אם אין מידע, נחמיר לפי הסימבול
    if symbol_info["base"] in ["BTC", "ETH"]:
        if max_reasonable > 200000:  # BTC
            max_reasonable = 200000
        if symbol_info["base"] == "ETH" and max_reasonable > 10000:
            max_reasonable = 10000
    
    # סינון
    mask = (
        (df['high'] <= max_reasonable) & 
        (df['low'] >= min_reasonable) &
        (df['high'] >= df['low']) &
        (df['close'] >= df['low']) &
        (df['close'] <= df['high'])
    )
    
    df_clean = df[mask].copy()
    
    if len(df_clean) < len(df) * 0.9:  # אם איבדנו יותר מ-10%
        print(f"   ⚠️ נוקו {len(df) - len(df_clean)} שורות פגומות")
    
    return df_clean

def download_and_test(symbol_info, test_params):
    """מוריד דאטה ומריץ בק-טסט"""
    symbol = symbol_info["symbol"]
    print(f"\n{'='*50}")
    print(f"📊 מעבד {symbol}")
    print(f"{'='*50}")
    
    try:
        # בדיקה אם כבר עיבדנו
        progress = {}
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r') as f:
                progress = json.load(f)
        
        if symbol in progress.get("completed", []):
            print(f"   ✓ כבר עובד")
            return progress["results"].get(symbol)
        
        # הורדת דאטה
        data_path = f"{OUTPUT_DIR}/data/{symbol}_1m.parquet"
        
        if not os.path.exists(data_path):
            print(f"   ⬇️ מוריד דאטה...")
            ex = BybitEx(CFG.BYBIT_API_KEY, CFG.BYBIT_API_SECRET, False)
            
            # הורדה בסיסית (בלי progress callback כדי לא להציף)
            path = download_year_1m(ex, symbol, f"{OUTPUT_DIR}/data", "1")
            
            if not path:
                print(f"   ❌ לא הצליח להוריד דאטה")
                return None
                
            time.sleep(0.5)  # מנוחה בין הורדות
        
        # טעינה וניקוי
        df = pd.read_parquet(data_path)
        df = clean_data(df, symbol_info)
        
        if len(df) < 10000:
            print(f"   ⚠️ מעט מדי דאטה ({len(df)} נרות)")
            return None
        
        # בק-טסט על כמה קונפיגורציות
        print(f"   🧪 מריץ בק-טסט...")
        results = []
        
        for params in test_params:
            result = backtest(df, CFG.TZ, params["session"], 
                            span=params["span"], 
                            wick_ratio=params["wick"], 
                            r_mult=params["r_mult"], 
                            vol_mult=0)
            
            if result["trades"] > 0:
                results.append({
                    "symbol": symbol,
                    "session": params["session"],
                    "params": params["name"],
                    **result,
                    "score": result["pf"] * result["winrate"] * (result["trades"] / 50)
                })
        
        # שמירת התקדמות
        if not progress.get("completed"):
            progress["completed"] = []
        if not progress.get("results"):
            progress["results"] = {}
            
        progress["completed"].append(symbol)
        progress["results"][symbol] = results
        
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=2)
        
        # דיווח
        if results:
            best = max(results, key=lambda x: x["score"])
            print(f"   ✅ {len(results)} תוצאות, הטוב: {best['trades']} עסקאות, WR={best['winrate']*100:.0f}%")
        else:
            print(f"   ❌ אין עסקאות")
        
        return results
        
    except Exception as e:
        print(f"   ❌ שגיאה: {e}")
        return None

def main():
    """תהליך ראשי"""
    ensure_dirs()
    
    print("🚀 Bybit Full Scanner")
    print("=" * 60)
    
    # שלב 1: קבלת כל הצמדים
    symbols = get_all_symbols()
    if not symbols:
        print("❌ לא נמצאו צמדים")
        return
    
    # פילטר לצמדים עיקריים (TOP 50 לפי נפח)
    # אפשר לשנות את המספר או להסיר את השורה הזו
    top_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", 
                   "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
                   "LINKUSDT", "LTCUSDT", "UNIUSDT", "ATOMUSDT", "XLMUSDT",
                   "NEARUSDT", "FILUSDT", "APTUSDT", "ARBUSDT", "OPUSDT"]
    
    symbols_to_test = [s for s in symbols if s["symbol"] in top_symbols]
    
    print(f"\n📊 יבדקו {len(symbols_to_test)} צמדים עיקריים")
    
    # פרמטרים לבדיקה
    test_params = [
        {"name": "Conservative", "session": "NY", "span": 2, "wick": 1.5, "r_mult": 2.0},
        {"name": "Standard", "session": "NY", "span": 2, "wick": 1.2, "r_mult": 1.8},
        {"name": "Aggressive", "session": "Asia", "span": 1, "wick": 0.8, "r_mult": 1.5},
    ]
    
    # שלב 2: הורדה ובדיקה
    all_results = []
    
    for i, symbol_info in enumerate(symbols_to_test, 1):
        print(f"\n[{i}/{len(symbols_to_test)}] {symbol_info['symbol']}")
        
        results = download_and_test(symbol_info, test_params)
        if results:
            all_results.extend(results)
        
        # שמירה מעת לעת
        if i % 5 == 0:
            save_results(all_results)
    
    # שלב 3: סיכום וניתוח
    save_results(all_results)
    print_summary(all_results)

def save_results(results):
    """שמירת תוצאות ל-CSV"""
    if not results:
        return
        
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\n💾 נשמרו {len(results)} תוצאות ל-{RESULTS_FILE}")

def print_summary(results):
    """הדפסת סיכום"""
    if not results:
        print("\n❌ אין תוצאות")
        return
    
    print("\n" + "="*60)
    print("📊 סיכום סופי")
    print("="*60)
    
    df = pd.DataFrame(results)
    
    # TOP 10 צמדים
    top_by_score = df.nlargest(10, 'score')
    print("\n🏆 TOP 10 צמדים לפי Score:")
    for _, row in top_by_score.iterrows():
        print(f"  {row['symbol']:12} | {row['session']:8} | "
              f"Trades: {row['trades']:3} | WR: {row['winrate']*100:5.1f}% | "
              f"PF: {row['pf']:5.2f} | Score: {row['score']:.3f}")
    
    # סטטיסטיקות
    print(f"\n📈 סטטיסטיקות:")
    print(f"  • סה\"כ צמדים שנבדקו: {df['symbol'].nunique()}")
    print(f"  • צמדים עם עסקאות: {df[df['trades']>0]['symbol'].nunique()}")
    print(f"  • סה\"כ קונפיגורציות: {len(df)}")
    print(f"  • ממוצע עסקאות: {df['trades'].mean():.1f}")
    print(f"  • ממוצע Win Rate: {df['winrate'].mean()*100:.1f}%")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ הופסק ידנית - התקדמות נשמרה")
        print(f"📁 כדי להמשיך, הרץ שוב את הסקריפט")
    except Exception as e:
        print(f"\n❌ שגיאה כללית: {e}")
        import traceback
        traceback.print_exc()