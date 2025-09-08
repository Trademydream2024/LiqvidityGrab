"""
strategy.py - אסטרטגיית Liquidity Grab
עדכון: אסטרטגיה מלאה עם ניהול סיכונים וסיגנלים
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from services.session_stats import SessionAnalyzer
from services.exchange import BybitEx

class LiquidityGrabStrategy:
    """
    אסטרטגיית Liquidity Grab עם ניהול סיכונים
    """
    
    def __init__(self, exchange: BybitEx, analyzer: SessionAnalyzer, config: Dict = None):
        """
        אתחול האסטרטגיה
        
        Args:
            exchange: אובייקט החיבור לבורסה
            analyzer: מנתח הסשנים
            config: הגדרות האסטרטגיה
        """
        self.exchange = exchange
        self.analyzer = analyzer
        self.logger = logging.getLogger(__name__)
        
        # הגדרות ברירת מחדל
        default_config = {
            'risk_per_trade': 1.0,           # אחוז סיכון לטרייד
            'max_daily_risk': 5.0,            # מקסימום סיכון יומי
            'rr_ratio': 2.0,                  # Risk:Reward ratio
            'min_grab_strength': 30,          # עוצמת grab מינימלית (0-100)
            'min_win_rate': 40,               # Win rate מינימלי נדרש
            'use_session_filter': True,       # סינון לפי סשנים
            'use_volume_filter': True,        # סינון לפי נפח
            'max_positions': 3,               # מקסימום פוזיציות במקביל
            'trailing_stop': True,            # שימוש ב-trailing stop
            'trailing_stop_percent': 1.0,     # אחוז ה-trailing
            'leverage': 10,                   # מינוף
            'timeframes': ['5', '15'],        # Timeframes לניתוח
            'confirmation_required': True     # דרוש אישור מ-timeframe גבוה
        }
        
        # מיזוג עם config שהתקבל
        self.config = {**default_config, **(config or {})}
        
        # מעקב אחרי ביצועים
        self.active_positions = {}
        self.daily_stats = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'pnl': 0,
            'risk_used': 0
        }
        
        # טעינת patterns היסטוריים
        self.patterns = {}
        
    def analyze_market(self, symbol: str) -> Dict:
        """
        ניתוח מקיף של השוק
        
        Args:
            symbol: הסימבול לניתוח
            
        Returns:
            ניתוח מפורט
        """
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'signals': [],
            'strength': 0,
            'recommendation': 'NEUTRAL'
        }
        
        try:
            # משיכת נתונים מכמה timeframes
            data_5m = self.exchange.get_klines(symbol, '5', limit=200)
            data_15m = self.exchange.get_klines(symbol, '15', limit=100)
            
            if data_5m.empty or data_15m.empty:
                self.logger.warning(f"No data available for {symbol}")
                return analysis
                
            # ניתוח liquidity zones
            data_5m = self.analyzer.find_liquidity_zones(data_5m)
            data_15m = self.analyzer.find_liquidity_zones(data_15m)
            
            # זיהוי סיגנלים
            signal_5m = self._check_for_signal(data_5m, symbol, '5m')
            signal_15m = self._check_for_signal(data_15m, symbol, '15m')
            
            # אם יש סיגנל ב-5 דקות
            if signal_5m:
                analysis['signals'].append(signal_5m)
                
                # בדיקת אישור מ-15 דקות
                if self.config['confirmation_required']:
                    if signal_15m and signal_15m['direction'] == signal_5m['direction']:
                        signal_5m['confirmed'] = True
                        signal_5m['strength'] *= 1.5
                    else:
                        signal_5m['confirmed'] = False
                        signal_5m['strength'] *= 0.7
                        
            # חישוב המלצה סופית
            if analysis['signals']:
                strongest_signal = max(analysis['signals'], key=lambda x: x['strength'])
                
                if strongest_signal['strength'] >= self.config['min_grab_strength']:
                    analysis['recommendation'] = strongest_signal['direction']
                    analysis['strength'] = strongest_signal['strength']
                    analysis['entry_price'] = strongest_signal['entry_price']
                    analysis['stop_loss'] = strongest_signal['stop_loss']
                    analysis['take_profit'] = strongest_signal['take_profit']
                    
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            
        return analysis
    
    def _check_for_signal(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Optional[Dict]:
        """
        בדיקה לסיגנל בדאטה
        
        Args:
            df: הדאטה לבדיקה
            symbol: הסימבול
            timeframe: ה-timeframe
            
        Returns:
            סיגנל אם נמצא
        """
        if len(df) < 50:
            return None
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # קבלת מידע על הסשן הנוכחי
        session_info = self.analyzer.get_current_session_info(symbol)
        
        signal = None
        
        # בדיקת Bullish Liquidity Grab
        if latest['liquidity_grab_bullish']:
            signal = self._create_bullish_signal(df, symbol, timeframe, session_info)
            
        # בדיקת Bearish Liquidity Grab
        elif latest['liquidity_grab_bearish']:
            signal = self._create_bearish_signal(df, symbol, timeframe, session_info)
            
        # החלת פילטרים נוספים
        if signal:
            signal = self._apply_filters(signal, df, session_info)
            
        return signal
    
    def _create_bullish_signal(self, df: pd.DataFrame, symbol: str, 
                              timeframe: str, session_info: Dict) -> Dict:
        """
        יצירת סיגנל bullish
        
        Args:
            df: הדאטה
            symbol: הסימבול
            timeframe: ה-timeframe
            session_info: מידע על הסשן
            
        Returns:
            הסיגנל
        """
        latest = df.iloc[-1]
        
        # חישוב רמות
        entry_price = latest['close']
        stop_loss = latest['low'] * 0.995  # 0.5% מתחת ל-low
        take_profit = entry_price + (entry_price - stop_loss) * self.config['rr_ratio']
        
        # חישוב עוצמת הסיגנל
        strength = latest['grab_strength']
        
        # הוספת משקל לפי הסשן
        if session_info['info']:
            session_prob = session_info['info'].get('bullish_grab_prob', 0)
            session_win_rate = session_info['info'].get('win_rate_bullish_grab', 0)
            
            # התאמת העוצמה לפי נתוני הסשן
            if session_prob > 0.02 and session_win_rate > self.config['min_win_rate']:
                strength *= 1.2
                
        return {
            'direction': 'BUY',
            'symbol': symbol,
            'timeframe': timeframe,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'strength': strength,
            'session': session_info['session'],
            'timestamp': datetime.now()
        }
    
    def _create_bearish_signal(self, df: pd.DataFrame, symbol: str,
                               timeframe: str, session_info: Dict) -> Dict:
        """
        יצירת סיגנל bearish
        
        Args:
            df: הדאטה
            symbol: הסימבול
            timeframe: ה-timeframe
            session_info: מידע על הסשן
            
        Returns:
            הסיגנל
        """
        latest = df.iloc[-1]
        
        # חישוב רמות
        entry_price = latest['close']
        stop_loss = latest['high'] * 1.005  # 0.5% מעל ה-high
        take_profit = entry_price - (stop_loss - entry_price) * self.config['rr_ratio']
        
        # חישוב עוצמת הסיגנל
        strength = latest['grab_strength']
        
        # הוספת משקל לפי הסשן
        if session_info['info']:
            session_prob = session_info['info'].get('bearish_grab_prob', 0)
            session_win_rate = session_info['info'].get('win_rate_bearish_grab', 0)
            
            if session_prob > 0.02 and session_win_rate > self.config['min_win_rate']:
                strength *= 1.2
                
        return {
            'direction': 'SELL',
            'symbol': symbol,
            'timeframe': timeframe,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'strength': strength,
            'session': session_info['session'],
            'timestamp': datetime.now()
        }
    
    def _apply_filters(self, signal: Dict, df: pd.DataFrame, session_info: Dict) -> Optional[Dict]:
        """
        החלת פילטרים על הסיגנל
        
        Args:
            signal: הסיגנל
            df: הדאטה
            session_info: מידע על הסשן
            
        Returns:
            הסיגנל אחרי פילטרים או None
        """
        # פילטר סשן
        if self.config['use_session_filter']:
            if session_info['session'] == 'no_session':
                signal['strength'] *= 0.5
                
        # פילטר נפח
        if self.config['use_volume_filter']:
            latest_volume = df.iloc[-1]['volume']
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            
            if latest_volume < avg_volume * 0.8:
                signal['strength'] *= 0.8
            elif latest_volume > avg_volume * 1.5:
                signal['strength'] *= 1.2
                
        # פילטר מומנטום
        rsi = self._calculate_rsi(df['close'])
        latest_rsi = rsi.iloc[-1]
        
        if signal['direction'] == 'BUY' and latest_rsi > 70:
            signal['strength'] *= 0.7  # Overbought
        elif signal['direction'] == 'SELL' and latest_rsi < 30:
            signal['strength'] *= 0.7  # Oversold
            
        # החלטה סופית
        if signal['strength'] < self.config['min_grab_strength']:
            return None
            
        return signal
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        חישוב RSI
        
        Args:
            prices: סדרת המחירים
            period: תקופת החישוב
            
        Returns:
            סדרת RSI
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def should_enter_trade(self, analysis: Dict) -> bool:
        """
        החלטה האם להיכנס לטרייד
        
        Args:
            analysis: הניתוח
            
        Returns:
            האם להיכנס לטרייד
        """
        # בדיקת תנאים בסיסיים
        if analysis['recommendation'] == 'NEUTRAL':
            return False
            
        # בדיקת מגבלות סיכון יומיות
        if self.daily_stats['risk_used'] >= self.config['max_daily_risk']:
            self.logger.warning("Daily risk limit reached")
            return False
            
        # בדיקת מספר פוזיציות מקסימלי
        if len(self.active_positions) >= self.config['max_positions']:
            self.logger.warning("Maximum positions limit reached")
            return False
            
        # בדיקת חוזק הסיגנל
        if analysis['strength'] < self.config['min_grab_strength']:
            return False
            
        return True
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                               stop_loss: float) -> float:
        """
        חישוב גודל הפוזיציה
        
        Args:
            symbol: הסימבול
            entry_price: מחיר כניסה
            stop_loss: מחיר stop loss
            
        Returns:
            גודל הפוזיציה
        """
        # קבלת הבאלנס
        balance = self.exchange.get_balance()
        
        # חישוב גודל לפי סיכון
        risk_amount = balance * (self.config['risk_per_trade'] / 100)
        price_diff = abs(entry_price - stop_loss)
        
        if price_diff == 0:
            return 0
            
        # חישוב הכמות
        position_size = risk_amount / price_diff
        
        # התאמה למינוף
        position_size *= self.config['leverage']
        
        # עיגול לפי דרישות הבורסה
        return round(position_size, 3)
    
    def execute_trade(self, analysis: Dict) -> Optional[Dict]:
        """
        ביצוע הטרייד
        
        Args:
            analysis: הניתוח
            
        Returns:
            פרטי הפקודה אם בוצעה
        """
        if not self.should_enter_trade(analysis):
            return None
            
        symbol = analysis['symbol']
        
        # חישוב גודל הפוזיציה
        position_size = self.calculate_position_size(
            symbol,
            analysis['entry_price'],
            analysis['stop_loss']
        )
        
        if position_size == 0:
            self.logger.warning("Position size is 0")
            return None
            
        # הגדרת מינוף
        self.exchange.set_leverage(symbol, self.config['leverage'])
        
        # ביצוע הפקודה
        side = 'Buy' if analysis['recommendation'] == 'BUY' else 'Sell'
        
        order = self.exchange.place_order(
            symbol=symbol,
            side=side,
            qty=position_size,
            stop_loss=analysis['stop_loss'],
            take_profit=analysis['take_profit']
        )
        
        if order:
            # עדכון רישומים
            self.active_positions[symbol] = {
                'order': order,
                'analysis': analysis,
                'entry_time': datetime.now(),
                'trailing_stop_price': analysis['stop_loss']
            }
            
            self.daily_stats['trades'] += 1
            self.daily_stats['risk_used'] += self.config['risk_per_trade']
            
            self.logger.info(f"Trade executed: {side} {position_size} {symbol} @ {analysis['entry_price']}")
            
            return order
            
        return None
    
    def manage_positions(self):
        """
        ניהול פוזיציות פתוחות
        """
        for symbol, position_data in list(self.active_positions.items()):
            try:
                # קבלת מחיר נוכחי
                current_price = self.exchange._get_current_price(symbol)
                
                if current_price == 0:
                    continue
                    
                analysis = position_data['analysis']
                
                # עדכון trailing stop
                if self.config['trailing_stop']:
                    self._update_trailing_stop(symbol, current_price, position_data)
                    
                # בדיקת יציאה
                should_exit = self._check_exit_conditions(symbol, current_price, position_data)
                
                if should_exit:
                    self._close_position(symbol, position_data)
                    
            except Exception as e:
                self.logger.error(f"Error managing position {symbol}: {e}")
    
    def _update_trailing_stop(self, symbol: str, current_price: float, position_data: Dict):
        """
        עדכון trailing stop
        
        Args:
            symbol: הסימבול
            current_price: המחיר הנוכחי
            position_data: נתוני הפוזיציה
        """
        analysis = position_data['analysis']
        
        if analysis['recommendation'] == 'BUY':
            # Long position - הזז stop loss למעלה
            new_stop = current_price * (1 - self.config['trailing_stop_percent'] / 100)
            
            if new_stop > position_data['trailing_stop_price']:
                position_data['trailing_stop_price'] = new_stop
                self.logger.info(f"Updated trailing stop for {symbol} to {new_stop}")
                
        else:
            # Short position - הזז stop loss למטה
            new_stop = current_price * (1 + self.config['trailing_stop_percent'] / 100)
            
            if new_stop < position_data['trailing_stop_price']:
                position_data['trailing_stop_price'] = new_stop
                self.logger.info(f"Updated trailing stop for {symbol} to {new_stop}")
    
    def _check_exit_conditions(self, symbol: str, current_price: float, position_data: Dict) -> bool:
        """
        בדיקת תנאי יציאה
        
        Args:
            symbol: הסימבול
            current_price: המחיר הנוכחי
            position_data: נתוני הפוזיציה
            
        Returns:
            האם לצאת מהפוזיציה
        """
        analysis = position_data['analysis']
        
        # בדיקת stop loss
        if analysis['recommendation'] == 'BUY':
            if current_price <= position_data['trailing_stop_price']:
                self.logger.info(f"Stop loss hit for {symbol}")
                return True
        else:
            if current_price >= position_data['trailing_stop_price']:
                self.logger.info(f"Stop loss hit for {symbol}")
                return True
                
        # בדיקת זמן בפוזיציה
        time_in_position = datetime.now() - position_data['entry_time']
        if time_in_position > timedelta(hours=24):
            self.logger.info(f"Time exit for {symbol}")
            return True
            
        return False
    
    def _close_position(self, symbol: str, position_data: Dict):
        """
        סגירת פוזיציה
        
        Args:
            symbol: הסימבול
            position_data: נתוני הפוזיציה
        """
        result = self.exchange.close_position(symbol)
        
        if result:
            # חישוב רווח/הפסד
            # (כאן צריך לחשב את ה-PnL האמיתי מהבורסה)
            
            del self.active_positions[symbol]
            self.logger.info(f"Position closed for {symbol}")
    
    def reset_daily_stats(self):
        """
        איפוס סטטיסטיקות יומיות
        """
        self.daily_stats = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'pnl': 0,
            'risk_used': 0
        }
        self.logger.info("Daily stats reset")