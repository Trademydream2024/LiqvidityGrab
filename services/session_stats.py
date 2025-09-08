"""
session_stats.py - מודול ניתוח סשנים וזיהוי Liquidity Grabs
עדכון: הוספת זיהוי liquidity grabs וניתוח לפי סשנים
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, List, Tuple, Optional
import json
import logging

class SessionAnalyzer:
    """
    מנתח סשנים וזיהוי Liquidity Grabs
    """
    
    # הגדרת זמני הסשנים (UTC)
    SESSIONS = {
        'sydney': {
            'start': time(21, 0),  # 21:00 UTC
            'end': time(6, 0),     # 06:00 UTC
            'active_pairs': ['AUDUSD', 'NZDUSD', 'AUDNZD'],
            'volatility_factor': 0.8
        },
        'tokyo': {
            'start': time(0, 0),   # 00:00 UTC
            'end': time(9, 0),     # 09:00 UTC
            'active_pairs': ['USDJPY', 'EURJPY', 'GBPJPY'],
            'volatility_factor': 0.9
        },
        'london': {
            'start': time(7, 0),   # 07:00 UTC
            'end': time(16, 0),    # 16:00 UTC
            'active_pairs': ['EURUSD', 'GBPUSD', 'EURGBP'],
            'volatility_factor': 1.2
        },
        'newyork': {
            'start': time(12, 0),  # 12:00 UTC
            'end': time(21, 0),    # 21:00 UTC
            'active_pairs': ['EURUSD', 'GBPUSD', 'USDCAD'],
            'volatility_factor': 1.3
        },
        'london_newyork_overlap': {
            'start': time(12, 0),  # 12:00 UTC
            'end': time(16, 0),    # 16:00 UTC
            'active_pairs': ['EURUSD', 'GBPUSD'],
            'volatility_factor': 1.5  # הכי הרבה נזילות
        }
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session_patterns = {}
        self.liquidity_levels = {}
        
    def identify_session(self, timestamp: datetime) -> str:
        """
        מזהה איזה סשן פעיל בזמן נתון
        
        Args:
            timestamp: הזמן לבדיקה
            
        Returns:
            שם הסשן הפעיל
        """
        current_time = timestamp.time()
        
        # בדיקת חפיפה קודם (לונדון-ניו יורק)
        overlap = self.SESSIONS['london_newyork_overlap']
        if overlap['start'] <= current_time < overlap['end']:
            return 'london_newyork_overlap'
            
        # בדיקת סשנים רגילים
        for session_name, session_info in self.SESSIONS.items():
            if session_name == 'london_newyork_overlap':
                continue
                
            start = session_info['start']
            end = session_info['end']
            
            # טיפול במקרה שהסשן עובר חצות
            if start > end:
                if current_time >= start or current_time < end:
                    return session_name
            else:
                if start <= current_time < end:
                    return session_name
                    
        return 'no_session'
        
    def find_liquidity_zones(self, df: pd.DataFrame, lookback: int = 20, 
                            sensitivity: float = 0.5) -> pd.DataFrame:
        """
        מזהה אזורי נזילות (Stop Loss Clusters)
        
        Args:
            df: DataFrame עם נתוני OHLCV
            lookback: כמה נרות אחורה לבדוק
            sensitivity: רגישות הזיהוי (0-1)
            
        Returns:
            DataFrame עם אזורי נזילות מסומנים
        """
        df = df.copy()
        
        # חישוב Swing Points
        df['swing_high'] = self._find_swing_points(df['high'], lookback, 'high')
        df['swing_low'] = self._find_swing_points(df['low'], lookback, 'low')
        
        # חישוב אזורי נזילות
        df['liquidity_above'] = df['swing_high'].rolling(window=lookback).max()
        df['liquidity_below'] = df['swing_low'].rolling(window=lookback).min()
        
        # זיהוי Liquidity Grabs
        df['liquidity_grab_bullish'] = False
        df['liquidity_grab_bearish'] = False
        
        for i in range(lookback, len(df)):
            # Bullish Liquidity Grab - מחיר יורד מתחת לנזילות ואז חוזר מעליה
            if (df.iloc[i]['low'] < df.iloc[i-1]['liquidity_below'] and 
                df.iloc[i]['close'] > df.iloc[i]['open'] and
                df.iloc[i]['close'] > df.iloc[i-1]['liquidity_below']):
                df.loc[df.index[i], 'liquidity_grab_bullish'] = True
                
            # Bearish Liquidity Grab - מחיר עולה מעל הנזילות ואז חוזר מתחתיה
            if (df.iloc[i]['high'] > df.iloc[i-1]['liquidity_above'] and
                df.iloc[i]['close'] < df.iloc[i]['open'] and
                df.iloc[i]['close'] < df.iloc[i-1]['liquidity_above']):
                df.loc[df.index[i], 'liquidity_grab_bearish'] = True
                
        # חישוב עוצמת ה-grab
        df['grab_strength'] = self._calculate_grab_strength(df)
        
        return df
        
    def _find_swing_points(self, series: pd.Series, lookback: int, 
                           point_type: str) -> pd.Series:
        """
        מוצא נקודות swing (פיבוט)
        
        Args:
            series: הסדרה לניתוח
            lookback: כמה נרות לבדוק
            point_type: 'high' או 'low'
            
        Returns:
            סדרה עם נקודות הswing
        """
        swing_points = pd.Series(index=series.index, dtype=float)
        
        for i in range(lookback, len(series) - lookback):
            if point_type == 'high':
                # Swing High - הנקודה הגבוהה ביותר בסביבה
                if series.iloc[i] == series.iloc[i-lookback:i+lookback+1].max():
                    swing_points.iloc[i] = series.iloc[i]
            else:
                # Swing Low - הנקודה הנמוכה ביותר בסביבה
                if series.iloc[i] == series.iloc[i-lookback:i+lookback+1].min():
                    swing_points.iloc[i] = series.iloc[i]
                    
        return swing_points.fillna(method='ffill')
        
    def _calculate_grab_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        מחשב את עוצמת ה-Liquidity Grab
        
        Args:
            df: DataFrame עם נתוני grabs
            
        Returns:
            סדרה עם עוצמת הgrabs (0-100)
        """
        strength = pd.Series(index=df.index, data=0.0)
        
        for i in range(1, len(df)):
            if df.iloc[i]['liquidity_grab_bullish']:
                # חישוב עוצמה לפי: עומק החדירה, נפח, וגודל הנר
                penetration = abs(df.iloc[i]['low'] - df.iloc[i-1]['liquidity_below'])
                volume_ratio = df.iloc[i]['volume'] / df['volume'].rolling(20).mean().iloc[i]
                candle_size = (df.iloc[i]['close'] - df.iloc[i]['low']) / (df.iloc[i]['high'] - df.iloc[i]['low'])
                
                strength.iloc[i] = min(100, (penetration * volume_ratio * candle_size) * 100)
                
            elif df.iloc[i]['liquidity_grab_bearish']:
                penetration = abs(df.iloc[i]['high'] - df.iloc[i-1]['liquidity_above'])
                volume_ratio = df.iloc[i]['volume'] / df['volume'].rolling(20).mean().iloc[i]
                candle_size = (df.iloc[i]['high'] - df.iloc[i]['close']) / (df.iloc[i]['high'] - df.iloc[i]['low'])
                
                strength.iloc[i] = min(100, (penetration * volume_ratio * candle_size) * 100)
                
        return strength
        
    def analyze_session_patterns(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        מנתח patterns לפי סשנים
        
        Args:
            df: DataFrame עם נתונים היסטוריים
            symbol: הסימבול
            
        Returns:
            דיקשנרי עם ניתוח לכל סשן
        """
        df = df.copy()
        
        # הוספת נתוני סשן
        df['session'] = df.index.map(self.identify_session)
        
        # הוספת נתוני liquidity
        df = self.find_liquidity_zones(df)
        
        patterns = {}
        
        for session_name in self.SESSIONS.keys():
            session_data = df[df['session'] == session_name]
            
            if len(session_data) < 50:
                continue
                
            # חישוב סטטיסטיקות
            patterns[session_name] = {
                # נתונים בסיסיים
                'avg_range': float((session_data['high'] - session_data['low']).mean()),
                'avg_volume': float(session_data['volume'].mean()),
                'volatility': float(session_data['close'].pct_change().std()),
                
                # Liquidity Grabs
                'bullish_grab_count': int(session_data['liquidity_grab_bullish'].sum()),
                'bearish_grab_count': int(session_data['liquidity_grab_bearish'].sum()),
                'bullish_grab_prob': float(session_data['liquidity_grab_bullish'].mean()),
                'bearish_grab_prob': float(session_data['liquidity_grab_bearish'].mean()),
                'avg_grab_strength': float(session_data['grab_strength'].mean()),
                
                # תוצאות אחרי Grabs
                'avg_move_after_bullish_grab': self._calculate_post_grab_movement(session_data, 'bullish'),
                'avg_move_after_bearish_grab': self._calculate_post_grab_movement(session_data, 'bearish'),
                
                # זמנים אופטימליים
                'best_hours': self._find_best_trading_hours(session_data),
                
                # Win Rate
                'win_rate_bullish_grab': self._calculate_grab_win_rate(session_data, 'bullish'),
                'win_rate_bearish_grab': self._calculate_grab_win_rate(session_data, 'bearish'),
            }
            
        # שמירה לקובץ
        self.session_patterns[symbol] = patterns
        self._save_patterns_to_file(symbol)
        
        return patterns
        
    def _calculate_post_grab_movement(self, df: pd.DataFrame, grab_type: str, 
                                     candles_forward: int = 10) -> Dict:
        """
        מחשב את התנועה הממוצעת אחרי Liquidity Grab
        
        Args:
            df: הדאטה
            grab_type: 'bullish' או 'bearish'
            candles_forward: כמה נרות קדימה לבדוק
            
        Returns:
            דיקשנרי עם סטטיסטיקות התנועה
        """
        grab_col = f'liquidity_grab_{grab_type}'
        movements = []
        
        for idx in df[df[grab_col] == True].index:
            pos = df.index.get_loc(idx)
            
            if pos + candles_forward >= len(df):
                continue
                
            entry_price = df.iloc[pos]['close']
            
            if grab_type == 'bullish':
                # בדיקת תנועה למעלה
                future_highs = df.iloc[pos+1:pos+candles_forward+1]['high']
                max_move = ((future_highs.max() - entry_price) / entry_price) * 100
                movements.append(max_move)
            else:
                # בדיקת תנועה למטה
                future_lows = df.iloc[pos+1:pos+candles_forward+1]['low']
                max_move = ((entry_price - future_lows.min()) / entry_price) * 100
                movements.append(max_move)
                
        if movements:
            return {
                'avg_move': float(np.mean(movements)),
                'max_move': float(np.max(movements)),
                'min_move': float(np.min(movements)),
                'std_move': float(np.std(movements))
            }
        else:
            return {'avg_move': 0, 'max_move': 0, 'min_move': 0, 'std_move': 0}
            
    def _find_best_trading_hours(self, session_data: pd.DataFrame) -> List[int]:
        """
        מוצא את השעות הטובות ביותר למסחר בסשן
        
        Args:
            session_data: נתוני הסשן
            
        Returns:
            רשימת השעות הטובות ביותר
        """
        session_data = session_data.copy()
        session_data['hour'] = session_data.index.hour
        
        # חישוב ציון לכל שעה
        hourly_stats = session_data.groupby('hour').agg({
            'liquidity_grab_bullish': 'sum',
            'liquidity_grab_bearish': 'sum',
            'grab_strength': 'mean',
            'volume': 'mean'
        })
        
        # נרמול וחישוב ציון משוקלל
        for col in hourly_stats.columns:
            hourly_stats[col] = (hourly_stats[col] - hourly_stats[col].min()) / (hourly_stats[col].max() - hourly_stats[col].min())
            
        hourly_stats['score'] = (
            hourly_stats['liquidity_grab_bullish'] * 0.3 +
            hourly_stats['liquidity_grab_bearish'] * 0.3 +
            hourly_stats['grab_strength'] * 0.2 +
            hourly_stats['volume'] * 0.2
        )
        
        # החזרת 3 השעות הטובות ביותר
        best_hours = hourly_stats.nlargest(3, 'score').index.tolist()
        return best_hours
        
    def _calculate_grab_win_rate(self, df: pd.DataFrame, grab_type: str,
                                 profit_threshold: float = 0.5) -> float:
        """
        מחשב Win Rate של Liquidity Grabs
        
        Args:
            df: הדאטה
            grab_type: 'bullish' או 'bearish'
            profit_threshold: סף הרווח להגדרת "ניצחון" (באחוזים)
            
        Returns:
            Win Rate (0-100)
        """
        grab_col = f'liquidity_grab_{grab_type}'
        wins = 0
        total = 0
        
        for idx in df[df[grab_col] == True].index:
            pos = df.index.get_loc(idx)
            
            if pos + 10 >= len(df):
                continue
                
            entry_price = df.iloc[pos]['close']
            total += 1
            
            if grab_type == 'bullish':
                # בדיקה אם הגענו ליעד רווח
                future_highs = df.iloc[pos+1:pos+11]['high']
                max_profit = ((future_highs.max() - entry_price) / entry_price) * 100
                if max_profit >= profit_threshold:
                    wins += 1
            else:
                # בדיקה אם הגענו ליעד רווח בשורט
                future_lows = df.iloc[pos+1:pos+11]['low']
                max_profit = ((entry_price - future_lows.min()) / entry_price) * 100
                if max_profit >= profit_threshold:
                    wins += 1
                    
        if total > 0:
            return (wins / total) * 100
        return 0
        
    def get_current_session_info(self, symbol: str) -> Dict:
        """
        מחזיר מידע על הסשן הנוכחי
        
        Args:
            symbol: הסימבול
            
        Returns:
            מידע על הסשן הנוכחי
        """
        current_session = self.identify_session(datetime.now())
        
        if symbol in self.session_patterns and current_session in self.session_patterns[symbol]:
            return {
                'session': current_session,
                'info': self.session_patterns[symbol][current_session],
                'volatility_factor': self.SESSIONS[current_session]['volatility_factor']
            }
        else:
            return {
                'session': current_session,
                'info': None,
                'volatility_factor': 1.0
            }
            
    def _save_patterns_to_file(self, symbol: str):
        """
        שומר את הpatterns לקובץ
        
        Args:
            symbol: הסימבול
        """
        try:
            filename = f'session_patterns_{symbol.replace("/", "_")}.json'
            with open(filename, 'w') as f:
                json.dump(self.session_patterns[symbol], f, indent=2, default=str)
            self.logger.info(f"Patterns saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving patterns: {e}")
            
    def load_patterns_from_file(self, symbol: str) -> bool:
        """
        טוען patterns מקובץ
        
        Args:
            symbol: הסימבול
            
        Returns:
            האם הטעינה הצליחה
        """
        try:
            filename = f'session_patterns_{symbol.replace("/", "_")}.json'
            with open(filename, 'r') as f:
                self.session_patterns[symbol] = json.load(f)
            self.logger.info(f"Patterns loaded from {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading patterns: {e}")
            return False