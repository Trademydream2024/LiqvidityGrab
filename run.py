"""
run.py - הקובץ הראשי להרצת בוט Liquidity Grab
עדכון: אינטגרציה מלאה של כל המודולים
"""

"""
run.py - הקובץ הראשי להרצת בוט Liquidity Grab
עדכון: אינטגרציה מלאה של כל המודולים
"""

import asyncio
import logging
import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List
from dotenv import load_dotenv
import pandas as pd
import schedule
import time
import threading

# ייבוא המודולים שלנו - עם הנתיבים הנכונים!
from services.exchange import BybitEx
from services.session_stats import SessionAnalyzer
from services.strategy import LiquidityGrabStrategy
from services.backtester import Backtester
from services.optimizer import Optimizer
from services.data_dl import DataDownloader

# טעינת משתני סביבה
load_dotenv()

class LiquidityGrabBot:
    """
    הבוט הראשי
    """
    
    def __init__(self):
        """
        אתחול הבוט
        """
        # הגדרת logging
        self._setup_logging()
        
        # טעינת קונפיגורציה
        self.config = self._load_config()
        
        # אתחול הבורסה
        self.exchange = BybitEx(
            api_key=os.getenv('BYBIT_API_KEY'),
            api_secret=os.getenv('BYBIT_API_SECRET'),
            testnet=self.config.get('testnet', True)
        )
        
        # אתחול מנתח הסשנים
        self.analyzer = SessionAnalyzer()
        
        # אתחול האסטרטגיה
        self.strategy = LiquidityGrabStrategy(
            exchange=self.exchange,
            analyzer=self.analyzer,
            config=self.config.get('strategy', {})
        )
        
        # רשימת הסימבולים למסחר
        self.symbols = self.config.get('symbols', ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
        
        # מצב הבוט
        self.is_running = False
        self.mode = 'IDLE'  # IDLE, ANALYZING, TRADING, BACKTESTING
        
        self.logger.info("Bot initialized successfully")
    
    def _setup_logging(self):
        """
        הגדרת מערכת הלוגים
        """
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # יצירת תיקיית logs אם לא קיימת
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        # הגדרת logger ראשי
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(f'logs/bot_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict:
        """
        טעינת קונפיגורציה מקובץ
        
        Returns:
            הקונפיגורציה
        """
        config_file = 'config.json'
        
        # קונפיגורציה ברירת מחדל
        default_config = {
            'testnet': True,
            'symbols': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
            'strategy': {
                'risk_per_trade': 1.0,
                'max_daily_risk': 5.0,
                'rr_ratio': 2.0,
                'min_grab_strength': 30,
                'min_win_rate': 40,
                'max_positions': 3,
                'leverage': 10,
                'timeframes': ['5', '15']
            },
            'analysis': {
                'lookback_days': 365,
                'update_frequency': 'daily'
            },
            'telegram': {
                'enabled': True,
                'send_signals': True,
                'send_trades': True
            }
        }
        
        # נסיון לטעון קובץ קונפיגורציה
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # מיזוג עם ברירת מחדל
                    for key, value in loaded_config.items():
                        if isinstance(value, dict) and key in default_config:
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
                    self.logger.info(f"Config loaded from {config_file}")
            except Exception as e:
                self.logger.error(f"Error loading config: {e}")
        else:
            # שמירת קונפיגורציה ברירת מחדל
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
                self.logger.info(f"Default config saved to {config_file}")
                
        return default_config
    
    async def analyze_historical_data(self):
        """
        ניתוח נתונים היסטוריים
        """
        self.mode = 'ANALYZING'
        self.logger.info("Starting historical analysis...")
        
        for symbol in self.symbols:
            try:
                self.logger.info(f"Analyzing {symbol}...")
                
                # משיכת נתונים היסטוריים
                lookback_days = self.config['analysis']['lookback_days']
                df = self.exchange.get_klines(
                    symbol=symbol,
                    interval='15',
                    limit=min(1000, lookback_days * 96)  # 96 נרות של 15 דקות ביום
                )
                
                if df.empty:
                    self.logger.warning(f"No data for {symbol}")
                    continue
                    
                # ניתוח patterns לפי סשנים
                patterns = self.analyzer.analyze_session_patterns(df, symbol)
                
                # הדפסת סיכום
                self._print_analysis_summary(symbol, patterns)
                
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
                
        self.mode = 'IDLE'
        self.logger.info("Historical analysis completed")
    
    def _print_analysis_summary(self, symbol: str, patterns: Dict):
        """
        הדפסת סיכום הניתוח
        
        Args:
            symbol: הסימבול
            patterns: ה-patterns שנמצאו
        """
        print(f"\n{'='*60}")
        print(f"Analysis Summary for {symbol}")
        print(f"{'='*60}")
        
        for session, data in patterns.items():
            print(f"\n{session.upper()} Session:")
            print(f"  - Bullish Grab Probability: {data['bullish_grab_prob']:.2%}")
            print(f"  - Bearish Grab Probability: {data['bearish_grab_prob']:.2%}")
            print(f"  - Avg Grab Strength: {data['avg_grab_strength']:.1f}")
            print(f"  - Win Rate (Bullish): {data['win_rate_bullish_grab']:.1f}%")
            print(f"  - Win Rate (Bearish): {data['win_rate_bearish_grab']:.1f}%")
            print(f"  - Best Hours: {data['best_hours']}")
    
    async def run_live_trading(self):
        """
        הרצת מסחר חי
        """
        self.is_running = True
        self.mode = 'TRADING'
        self.logger.info("Starting live trading...")
        
        # לולאת המסחר הראשית
        while self.is_running:
            try:
                # סריקת כל הסימבולים
                for symbol in self.symbols:
                    # ניתוח השוק
                    analysis = self.strategy.analyze_market(symbol)
                    
                    # הדפסת הניתוח
                    if analysis['recommendation'] != 'NEUTRAL':
                        self._log_signal(analysis)
                        
                        # ביצוע טרייד אם מתאים
                        order = self.strategy.execute_trade(analysis)
                        
                        if order:
                            self._log_trade(order, analysis)
                            
                # ניהול פוזיציות פתוחות
                self.strategy.manage_positions()
                
                # המתנה לסריקה הבאה
                await asyncio.sleep(60)  # סריקה כל דקה
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(60)
                
        self.mode = 'IDLE'
        self.logger.info("Live trading stopped")
    
    def _log_signal(self, analysis: Dict):
        """
        רישום סיגנל בלוג
        
        Args:
            analysis: הניתוח
        """
        self.logger.info(
            f"SIGNAL: {analysis['symbol']} - {analysis['recommendation']} "
            f"Strength: {analysis['strength']:.1f} "
            f"Entry: {analysis.get('entry_price', 0):.2f} "
            f"SL: {analysis.get('stop_loss', 0):.2f} "
            f"TP: {analysis.get('take_profit', 0):.2f}"
        )
    
    def _log_trade(self, order: Dict, analysis: Dict):
        """
        רישום טרייד בלוג
        
        Args:
            order: הפקודה
            analysis: הניתוח
        """
        self.logger.info(
            f"TRADE EXECUTED: {analysis['symbol']} - {analysis['recommendation']} "
            f"Order ID: {order.get('orderId', 'N/A')}"
        )
    
    def run_backtest(self, start_date: str = None, end_date: str = None):
        """
        הרצת בקטסט
        
        Args:
            start_date: תאריך התחלה
            end_date: תאריך סיום
        """
        self.mode = 'BACKTESTING'
        self.logger.info("Starting backtest...")
        
        try:
            # יצירת אובייקט בקטסטר
            backtester = Backtester(
                strategy=self.strategy,
                initial_balance=10000
            )
            
            # הרצת בקטסט לכל סימבול
            for symbol in self.symbols:
                # משיכת נתונים היסטוריים
                df = self.exchange.get_klines(
                    symbol=symbol,
                    interval='5',
                    limit=1000
                )
                
                if df.empty:
                    continue
                    
                # הרצת הבקטסט
                results = backtester.run(df, symbol)
                
                # הדפסת תוצאות
                self._print_backtest_results(symbol, results)
                
        except Exception as e:
            self.logger.error(f"Error in backtest: {e}")
            
        self.mode = 'IDLE'
        self.logger.info("Backtest completed")
    
    def _print_backtest_results(self, symbol: str, results: Dict):
        """
        הדפסת תוצאות בקטסט
        
        Args:
            symbol: הסימבול
            results: התוצאות
        """
        print(f"\n{'='*60}")
        print(f"Backtest Results for {symbol}")
        print(f"{'='*60}")
        print(f"Total Trades: {results.get('total_trades', 0)}")
        print(f"Win Rate: {results.get('win_rate', 0):.1f}%")
        print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {results.get('max_drawdown', 0):.1f}%")
        print(f"Total Return: {results.get('total_return', 0):.1f}%")
    
    def optimize_strategy(self):
        """
        אופטימיזציה של האסטרטגיה
        """
        self.logger.info("Starting strategy optimization...")
        
        try:
            # יצירת אובייקט אופטימייזר
            optimizer = Optimizer(
                strategy=self.strategy,
                exchange=self.exchange
            )
            
            # הגדרת טווחי פרמטרים לבדיקה
            param_ranges = {
                'risk_per_trade': [0.5, 1.0, 1.5, 2.0],
                'rr_ratio': [1.5, 2.0, 2.5, 3.0],
                'min_grab_strength': [20, 30, 40, 50],
                'leverage': [5, 10, 15, 20]
            }
            
            # הרצת אופטימיזציה
            best_params = optimizer.optimize(
                symbols=self.symbols,
                param_ranges=param_ranges,
                metric='sharpe_ratio'
            )
            
            # הדפסת התוצאות
            print(f"\nBest Parameters Found:")
            print(json.dumps(best_params, indent=2))
            
            # שאלה האם לעדכן
            response = input("\nUpdate strategy with these parameters? (y/n): ")
            if response.lower() == 'y':
                self.strategy.config.update(best_params)
                self.logger.info("Strategy updated with optimized parameters")
                
        except Exception as e:
            self.logger.error(f"Error in optimization: {e}")
    
    def schedule_tasks(self):
        """
        תזמון משימות אוטומטיות
        """
        # ניתוח יומי
        schedule.every().day.at("00:00").do(lambda: asyncio.run(self.analyze_historical_data()))
        
        # איפוס סטטיסטיקות יומיות
        schedule.every().day.at("00:00").do(self.strategy.reset_daily_stats)
        
        # אופטימיזציה שבועית
        schedule.every().sunday.at("00:00").do(self.optimize_strategy)
        
        def run_schedule():
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)
                
        # הרצה בthread נפרד
        schedule_thread = threading.Thread(target=run_schedule)
        schedule_thread.daemon = True
        schedule_thread.start()
    
    def print_menu(self):
        """
        הדפסת תפריט
        """
        print("\n" + "="*60)
        print("LIQUIDITY GRAB BOT - MAIN MENU")
        print("="*60)
        print(f"Mode: {self.mode} | Testnet: {self.config['testnet']}")
        print("-"*60)
        print("1. Analyze Historical Data")
        print("2. Start Live Trading")
        print("3. Run Backtest")
        print("4. Optimize Strategy")
        print("5. Show Current Positions")
        print("6. Show Session Info")
        print("7. Toggle Testnet/Live")
        print("8. Reload Config")
        print("9. Exit")
        print("-"*60)
    
    def show_positions(self):
        """
        הצגת פוזיציות פתוחות
        """
        positions = self.exchange.get_positions()
        
        if not positions:
            print("No open positions")
            return
            
        print("\nOpen Positions:")
        print("-"*60)
        
        for pos in positions:
            print(f"Symbol: {pos['symbol']}")
            print(f"Side: {pos['side']}")
            print(f"Size: {pos['size']}")
            print(f"Entry Price: {pos['avgPrice']}")
            print(f"Unrealized PnL: {pos['unrealisedPnl']}")
            print("-"*60)
    
    def show_session_info(self):
        """
        הצגת מידע על הסשן הנוכחי
        """
        current_session = self.analyzer.identify_session(datetime.now())
        print(f"\nCurrent Session: {current_session}")
        
        for symbol in self.symbols:
            info = self.analyzer.get_current_session_info(symbol)
            if info['info']:
                print(f"\n{symbol}:")
                print(f"  Volatility Factor: {info['volatility_factor']}")
                print(f"  Bullish Grab Prob: {info['info']['bullish_grab_prob']:.2%}")
                print(f"  Bearish Grab Prob: {info['info']['bearish_grab_prob']:.2%}")
    
    async def main_loop(self):
        """
        הלולאה הראשית של הבוט
        """
        self.is_running = True
        
        # הפעלת תזמון משימות
        self.schedule_tasks()
        
        while self.is_running:
            self.print_menu()
            
            try:
                choice = input("Enter your choice: ")
                
                if choice == '1':
                    await self.analyze_historical_data()
                    
                elif choice == '2':
                    # הרצת המסחר בtask נפרד
                    trading_task = asyncio.create_task(self.run_live_trading())
                    print("Live trading started. Press Enter to stop...")
                    input()
                    self.is_running = False
                    await trading_task
                    self.is_running = True
                    
                elif choice == '3':
                    self.run_backtest()
                    
                elif choice == '4':
                    self.optimize_strategy()
                    
                elif choice == '5':
                    self.show_positions()
                    
                elif choice == '6':
                    self.show_session_info()
                    
                elif choice == '7':
                    self.config['testnet'] = not self.config['testnet']
                    print(f"Switched to {'Testnet' if self.config['testnet'] else 'Live'} mode")
                    # צריך לאתחל מחדש את החיבור
                    self.exchange = BybitEx(
                        api_key=os.getenv('BYBIT_API_KEY'),
                        api_secret=os.getenv('BYBIT_API_SECRET'),
                        testnet=self.config['testnet']
                    )
                    
                elif choice == '8':
                    self.config = self._load_config()
                    print("Config reloaded")
                    
                elif choice == '9':
                    self.is_running = False
                    print("Exiting...")
                    break
                    
                else:
                    print("Invalid choice")
                    
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                self.is_running = False
                break
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                
        print("Bot stopped")


async def main():
    """
    נקודת הכניסה הראשית
    """
    print("="*60)
    print("LIQUIDITY GRAB BOT v1.0")
    print("="*60)
    
    bot = LiquidityGrabBot()
    await bot.main_loop()


if __name__ == "__main__":
    asyncio.run(main())