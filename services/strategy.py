import numpy as np
import pandas as pd


def find_fractals(high, low, span=2):
    """מוצא נקודות שיא ושפל (fractals) - ללא שינוי"""
    sh, sl = [], []
    for i in range(span, len(high)-span):
        if high[i] == max(high[i-span:i+span+1]):
            sh.append(i)
        if low[i] == min(low[i-span:i+span+1]):
            sl.append(i)
    return sh, sl

def detect_sweep(df: pd.DataFrame, swing_idx: int, direction="long", min_wick_ratio=1.5):
    """מזהה sweep עם פילטר נוסף - הנר צריך לסגור בכיוון הנכון"""
    i = swing_idx
    
    if i >= len(df):
        return None
    
    for j in range(i+1, min(i+15, len(df))):
        try:
            o = df.iloc[j]['open']
            h = df.iloc[j]['high']
            l = df.iloc[j]['low']
            c = df.iloc[j]['close']
            swing_high = df.iloc[i]['high']
            swing_low = df.iloc[i]['low']
        except (IndexError, KeyError):
            continue
            
        if direction == "long":
            # sweep של high עם סגירה מתחת + נר אדום
            if h > swing_high and c < swing_high and c < o:  # הוספנו תנאי שהנר אדום
                body = abs(c - o)
                upper_wick = h - max(c, o)
                if upper_wick > min_wick_ratio * body:
                    # בדיקה נוספת - האם זה באמת rejection חזק?
                    rejection_strength = upper_wick / (h - l) if (h - l) > 0 else 0
                    if rejection_strength > 0.5:  # לפחות 50% מהנר הוא wick עליון
                        return j
                    
        elif direction == "short":
            # sweep של low עם סגירה מעל + נר ירוק
            if l < swing_low and c > swing_low and c > o:  # הוספנו תנאי שהנר ירוק
                body = abs(c - o)
                lower_wick = min(c, o) - l
                if lower_wick > min_wick_ratio * body:
                    # בדיקה נוספת - האם זה באמת rejection חזק?
                    rejection_strength = lower_wick / (h - l) if (h - l) > 0 else 0
                    if rejection_strength > 0.5:  # לפחות 50% מהנר הוא wick תחתון
                        return j
    
    return None

def find_order_block(df: pd.DataFrame, sweep_idx: int, direction="long"):
    """מוצא order block חזק יותר - מחפש נר עם נפח גבוה"""
    j = sweep_idx
    
    if j >= len(df) or j < 1:
        return None
    
    # חישוב נפח ממוצע
    if 'volume' in df.columns:
        avg_vol = df['volume'].rolling(20, min_periods=1).mean()
    else:
        avg_vol = pd.Series([1] * len(df))  # אם אין נפח
    
    best_ob = None
    best_volume = 0
    
    if direction == "long":
        # חיפוש bearish order block עם נפח גבוה
        for k in range(j-1, max(j-10, 0), -1):
            try:
                if df.iloc[k]['close'] < df.iloc[k]['open']:  # נר אדום
                    # בדיקת נפח - רוצים נר עם נפח גבוה
                    if 'volume' in df.columns:
                        vol_ratio = df.iloc[k]['volume'] / avg_vol.iloc[k] if avg_vol.iloc[k] > 0 else 1
                        if vol_ratio > 1.5 and vol_ratio > best_volume:  # נפח גבוה מהממוצע
                            best_volume = vol_ratio
                            best_ob = k
                    else:
                        # אם אין נפח, פשוט קח את הראשון
                        return k
            except (IndexError, KeyError):
                continue
                
    else:  # short
        # חיפוש bullish order block עם נפח גבוה
        for k in range(j-1, max(j-10, 0), -1):
            try:
                if df.iloc[k]['close'] > df.iloc[k]['open']:  # נר ירוק
                    # בדיקת נפח
                    if 'volume' in df.columns:
                        vol_ratio = df.iloc[k]['volume'] / avg_vol.iloc[k] if avg_vol.iloc[k] > 0 else 1
                        if vol_ratio > 1.5 and vol_ratio > best_volume:
                            best_volume = vol_ratio
                            best_ob = k
                    else:
                        return k
            except (IndexError, KeyError):
                continue
    
    return best_ob

def compute_ob_entry(df: pd.DataFrame, sweep_idx: int, direction="long"):
    """מחשב נקודת כניסה משופרת - משתמש ב-order block טוב יותר"""
    j = sweep_idx
    
    if j >= len(df) or j < 1:
        return None, None
    
    # מצא order block עם נפח גבוה
    ob_idx = find_order_block(df, j, direction)
    if ob_idx is None:
        return None, None
    
    try:
        if direction == "long":
            # כניסה ב-50% של ה-order block
            entry = (df.iloc[ob_idx]['high'] + df.iloc[ob_idx]['low']) / 2
            # Stop loss מתחת לנמוך של ה-sweep עם buffer קטן
            sl = df.iloc[j]['low'] * 0.998  # 0.2% buffer
            
            # בדיקה שה-R:R הגיוני
            if entry <= sl:
                return None, None
                
        else:  # short
            entry = (df.iloc[ob_idx]['high'] + df.iloc[ob_idx]['low']) / 2
            # Stop loss מעל לגבוה של ה-sweep עם buffer קטן
            sl = df.iloc[j]['high'] * 1.002  # 0.2% buffer
            
            # בדיקה שה-R:R הגיוני
            if entry >= sl:
                return None, None
                
        return entry, sl
        
    except (IndexError, KeyError):
        return None, None

def validate_setup(df: pd.DataFrame, sweep_idx: int, direction="long"):
    """בדיקה נוספת - האם יש momentum בכיוון הנכון אחרי ה-sweep?"""
    j = sweep_idx
    
    if j >= len(df) - 3:  # צריך לפחות 3 נרות אחרי
        return False
    
    try:
        # בדוק את 3 הנרות אחרי ה-sweep
        next_candles = df.iloc[j+1:j+4]
        
        if direction == "long":
            # לפחות נר אחד ירוק עם סגירה מעל אמצע ה-sweep candle
            sweep_mid = (df.iloc[j]['high'] + df.iloc[j]['low']) / 2
            for _, candle in next_candles.iterrows():
                if candle['close'] > candle['open'] and candle['close'] > sweep_mid:
                    return True
                    
        else:  # short
            # לפחות נר אחד אדום עם סגירה מתחת לאמצע ה-sweep candle
            sweep_mid = (df.iloc[j]['high'] + df.iloc[j]['low']) / 2
            for _, candle in next_candles.iterrows():
                if candle['close'] < candle['open'] and candle['close'] < sweep_mid:
                    return True
    except Exception as e:
        # Handle the exception (e.g., log it or pass)
        pass
                    
def validate_setup(df: pd.DataFrame, sweep_idx: int, direction="long"):
    """בדיקה נוספת - האם יש momentum בכיוון הנכון אחרי ה-sweep?"""
    j = sweep_idx
    
    if j >= len(df) - 3:  # צריך לפחות 3 נרות אחרי
        return False
    
    try:
        # בדוק את 3 הנרות אחרי ה-sweep
        next_candles = df.iloc[j+1:j+4]
        
        if direction == "long":
            # לפחות נר אחד ירוק עם סגירה מעל אמצע ה-sweep candle
            sweep_mid = (df.iloc[j]['high'] + df.iloc[j]['low']) / 2
            for _, candle in next_candles.iterrows():
                if candle['close'] > candle['open'] and candle['close'] > sweep_mid:
                    return True
                    
        else:  # short
            # לפחות נר אחד אדום עם סגירה מתחת לאמצע ה-sweep candle
            sweep_mid = (df.iloc[j]['high'] + df.iloc[j]['low']) / 2
            for _, candle in next_candles.iterrows():
                if candle['close'] < candle['open'] and candle['close'] < sweep_mid:
                    return True
                    
    except:
        pass
        
    return False