"""
exchange.py - מודול החיבור ל-Bybit
עדכון: הוספת כל הפונקציות הנדרשות למסחר
"""

from pybit.unified_trading import HTTP
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple

class BybitEx:
    """
    מחלקת החיבור ל-Bybit עם כל הפונקציות הנדרשות
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        אתחול החיבור ל-Bybit
        """
        self.http = HTTP(api_key=api_key, api_secret=api_secret, testnet=testnet)
        self.testnet = testnet
        self.positions = {}
        self.logger = logging.getLogger(__name__)
        
        # בדיקת החיבור
        self._test_connection()
        
    def _test_connection(self):
        """
        בדיקת החיבור ל-API
        """
        try:
            result = self.http.get_wallet_balance(accountType="UNIFIED")
            if result['retCode'] == 0:
                self.logger.info("Successfully connected to Bybit")
            else:
                self.logger.error(f"Connection test failed: {result}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Bybit: {e}")
            
    def get_klines(self, symbol: str, interval: str, start_ms: int = None, 
                   end_ms: int = None, limit: int = 200, category="linear") -> pd.DataFrame:
        """
        משיכת נתוני נרות
        
        Args:
            symbol: הסימבול (למשל 'BTCUSDT')
            interval: אינטרוול ('1', '5', '15', '30', '60', '240', 'D')
            start_ms: זמן התחלה במילישניות
            end_ms: זמן סיום במילישניות
            limit: מספר נרות (מקסימום 1000)
            category: סוג המכשיר ('linear', 'inverse', 'spot')
        
        Returns:
            DataFrame עם נתוני OHLCV
        """
        try:
            # אם לא ניתנו זמנים, קח את ה-X נרות האחרונים
            if not start_ms:
                end_ms = int(datetime.now().timestamp() * 1000)
                
                # חישוב זמן התחלה לפי האינטרוול
                interval_minutes = self._interval_to_minutes(interval)
                start_ms = end_ms - (limit * interval_minutes * 60 * 1000)
                
            response = self.http.get_kline(
                category=category,
                symbol=symbol,
                interval=interval,
                start=start_ms,
                end=end_ms,
                limit=limit
            )
            
            if response['retCode'] != 0:
                self.logger.error(f"Error fetching klines: {response['retMsg']}")
                return pd.DataFrame()
                
            # המרה ל-DataFrame
            klines = response['result']['list']
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # המרת טיפוסים
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()  # Bybit מחזיר בסדר הפוך
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in get_klines: {e}")
            return pd.DataFrame()
            
    def place_order(self, symbol: str, side: str, qty: float, 
                   order_type: str = "Market", price: float = None,
                   stop_loss: float = None, take_profit: float = None,
                   category: str = "linear") -> Dict:
        """
        פתיחת פוזיציה
        
        Args:
            symbol: הסימבול
            side: 'Buy' או 'Sell'
            qty: כמות
            order_type: 'Market' או 'Limit'
            price: מחיר (רק ל-Limit orders)
            stop_loss: מחיר Stop Loss
            take_profit: מחיר Take Profit
            category: סוג המכשיר
        
        Returns:
            תוצאת הפקודה
        """
        try:
            params = {
                "category": category,
                "symbol": symbol,
                "side": side,
                "orderType": order_type,
                "qty": str(qty),
                "timeInForce": "GTC"
            }
            
            # הוספת מחיר ל-Limit orders
            if order_type == "Limit" and price:
                params["price"] = str(price)
                
            # הוספת SL ו-TP
            if stop_loss:
                params["stopLoss"] = str(stop_loss)
            if take_profit:
                params["takeProfit"] = str(take_profit)
                
            response = self.http.place_order(**params)
            
            if response['retCode'] == 0:
                order_id = response['result']['orderId']
                self.logger.info(f"Order placed successfully: {order_id}")
                
                # שמירת הפוזיציה
                self.positions[symbol] = {
                    'side': side,
                    'qty': qty,
                    'entry_price': price if price else self._get_current_price(symbol),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'order_id': order_id,
                    'timestamp': datetime.now()
                }
                
                return response['result']
            else:
                self.logger.error(f"Failed to place order: {response['retMsg']}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {}
            
    def cancel_all_orders(self, symbol: str = None, category: str = "linear") -> bool:
        """
        ביטול כל הפקודות הפתוחות
        
        Args:
            symbol: סימבול ספציפי (אופציונלי)
            category: סוג המכשיר
        
        Returns:
            האם הביטול הצליח
        """
        try:
            params = {"category": category}
            if symbol:
                params["symbol"] = symbol
                
            response = self.http.cancel_all_orders(**params)
            
            if response['retCode'] == 0:
                self.logger.info(f"All orders cancelled for {symbol if symbol else 'all symbols'}")
                return True
            else:
                self.logger.error(f"Failed to cancel orders: {response['retMsg']}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error cancelling orders: {e}")
            return False
            
    def get_positions(self, symbol: str = None, category: str = "linear") -> List[Dict]:
        """
        קבלת פוזיציות פתוחות
        
        Args:
            symbol: סימבול ספציפי (אופציונלי)
            category: סוג המכשיר
        
        Returns:
            רשימת פוזיציות
        """
        try:
            params = {"category": category, "settleCoin": "USDT"}
            if symbol:
                params["symbol"] = symbol
                
            response = self.http.get_positions(**params)
            
            if response['retCode'] == 0:
                positions = response['result']['list']
                
                # סינון רק פוזיציות פתוחות
                open_positions = [p for p in positions if float(p['size']) > 0]
                
                return open_positions
            else:
                self.logger.error(f"Failed to get positions: {response['retMsg']}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
            
    def close_position(self, symbol: str, category: str = "linear") -> Dict:
        """
        סגירת פוזיציה
        
        Args:
            symbol: הסימבול
            category: סוג המכשיר
        
        Returns:
            תוצאת הסגירה
        """
        try:
            # קבלת הפוזיציה הנוכחית
            positions = self.get_positions(symbol=symbol, category=category)
            
            if not positions:
                self.logger.warning(f"No open position for {symbol}")
                return {}
                
            position = positions[0]
            side = "Sell" if position['side'] == "Buy" else "Buy"
            qty = position['size']
            
            # סגירת הפוזיציה
            result = self.place_order(
                symbol=symbol,
                side=side,
                qty=qty,
                order_type="Market",
                category=category
            )
            
            if result:
                self.logger.info(f"Position closed for {symbol}")
                if symbol in self.positions:
                    del self.positions[symbol]
                    
            return result
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return {}
            
    def get_balance(self, coin: str = "USDT", accountType: str = "UNIFIED") -> float:
        """
        קבלת יתרה
        
        Args:
            coin: המטבע
            accountType: סוג החשבון
        
        Returns:
            היתרה הזמינה
        """
        try:
            response = self.http.get_wallet_balance(accountType=accountType, coin=coin)
            
            if response['retCode'] == 0:
                coins = response['result']['list'][0]['coin']
                for coin_data in coins:
                    if coin_data['coin'] == coin:
                        return float(coin_data['availableToWithdraw'])
                        
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting balance: {e}")
            return 0.0
            
    def get_tickers(self, symbol: str = None, category: str = "linear") -> List[Dict]:
        """
        קבלת מחירים נוכחיים
        
        Args:
            symbol: סימבול ספציפי (אופציונלי)
            category: סוג המכשיר
        
        Returns:
            רשימת טיקרים
        """
        try:
            params = {"category": category}
            if symbol:
                params["symbol"] = symbol
                
            response = self.http.get_tickers(**params)
            
            if response['retCode'] == 0:
                return response['result']['list']
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting tickers: {e}")
            return []
            
    def _get_current_price(self, symbol: str, category: str = "linear") -> float:
        """
        קבלת המחיר הנוכחי של סימבול
        
        Args:
            symbol: הסימבול
            category: סוג המכשיר
        
        Returns:
            המחיר הנוכחי
        """
        tickers = self.get_tickers(symbol=symbol, category=category)
        if tickers:
            return float(tickers[0]['lastPrice'])
        return 0.0
        
    def _interval_to_minutes(self, interval: str) -> int:
        """
        המרת אינטרוול לדקות
        
        Args:
            interval: האינטרוול ('1', '5', '15', '30', '60', '240', 'D')
        
        Returns:
            מספר הדקות
        """
        interval_map = {
            '1': 1,
            '5': 5,
            '15': 15,
            '30': 30,
            '60': 60,
            '240': 240,
            'D': 1440,
            '1D': 1440,
            'W': 10080,
            '1W': 10080
        }
        return interval_map.get(interval, 5)
        
    def calculate_position_size(self, balance: float, risk_percent: float,
                               entry_price: float, stop_loss: float) -> float:
        """
        חישוב גודל פוזיציה לפי ניהול סיכונים
        
        Args:
            balance: היתרה
            risk_percent: אחוז הסיכון
            entry_price: מחיר כניסה
            stop_loss: מחיר Stop Loss
        
        Returns:
            גודל הפוזיציה
        """
        risk_amount = balance * (risk_percent / 100)
        price_diff = abs(entry_price - stop_loss)
        
        if price_diff == 0:
            return 0
            
        position_size = risk_amount / price_diff
        
        # עיגול לפי דרישות Bybit
        return round(position_size, 3)
        
    def set_leverage(self, symbol: str, leverage: int, category: str = "linear") -> bool:
        """
        הגדרת מינוף
        
        Args:
            symbol: הסימבול
            leverage: המינוף (1-100)
            category: סוג המכשיר
        
        Returns:
            האם ההגדרה הצליחה
        """
        try:
            response = self.http.set_leverage(
                category=category,
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage)
            )
            
            if response['retCode'] == 0:
                self.logger.info(f"Leverage set to {leverage}x for {symbol}")
                return True
            else:
                self.logger.error(f"Failed to set leverage: {response['retMsg']}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error setting leverage: {e}")
            return False