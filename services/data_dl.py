"""
NY Liquidity Bot - Data Downloader Module
Handles downloading and managing historical market data from various sources.
Specialized for Bybit API integration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta, timezone
import logging
import os
import json
import time
import requests
from pathlib import Path
import yfinance as yf
import ccxt
import pickle
from pybit.unified_trading import HTTP

logger = logging.getLogger(__name__)

class DataDownloader:
    """
    Main data downloader class for NY Liquidity Bot.
    Supports multiple data sources with focus on Bybit API.
    """
    
    def __init__(self, 
                 data_source: str = 'bybit',
                 exchange: Optional[str] = None,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 testnet: bool = True,
                 cache_dir: str = 'data/cache'):
        """
        Initialize data downloader.
        
        Args:
            data_source: Source for data ('bybit', 'yfinance', 'ccxt')
            exchange: Exchange name for CCXT
            api_key: API key for data providers
            api_secret: API secret for authentication
            testnet: Use testnet (for Bybit)
            cache_dir: Directory for caching downloaded data
        """
        self.data_source = data_source.lower()
        self.exchange_name = exchange
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Bybit session if using Bybit
        self.bybit_session = None
        if self.data_source == 'bybit':
            self._init_bybit_session()
        
        # Initialize exchange if using CCXT
        self.exchange = None
        if self.data_source == 'ccxt' and exchange:
            self._init_ccxt_exchange()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # seconds
        
        logger.info(f"DataDownloader initialized with source: {data_source}")
    
    def _init_bybit_session(self):
        """Initialize Bybit API session."""
        try:
            if self.testnet:
                self.bybit_session = HTTP(
                    testnet=True,
                    api_key=self.api_key,
                    api_secret=self.api_secret
                )
            else:
                self.bybit_session = HTTP(
                    testnet=False,
                    api_key=self.api_key,
                    api_secret=self.api_secret
                )
            
            logger.info(f"Connected to Bybit {'testnet' if self.testnet else 'mainnet'}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Bybit session: {e}")
            self.bybit_session = None
    
    def download_bybit(self,
                      symbol: str,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      timeframe: str = '1m') -> pd.DataFrame:
        """
        Download data from Bybit API.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            timeframe: Timeframe ('1', '3', '5', '15', '30', '60', '120', '240', '360', '720', 'D', 'W', 'M')
        
        Returns:
            DataFrame with OHLCV data
        """
        if not self.bybit_session:
            logger.error("Bybit session not initialized")
            return pd.DataFrame()
        
        try:
            # Convert timeframe to Bybit interval
            interval_map = {
                '1m': '1',
                '3m': '3',
                '5m': '5',
                '15m': '15',
                '30m': '30',
                '1h': '60',
                '2h': '120',
                '4h': '240',
                '6h': '360',
                '12h': '720',
                '1d': 'D',
                '1w': 'W',
                '1M': 'M'
            }
            
            interval = interval_map.get(timeframe, '1')
            
            # Convert dates to timestamps
            if start_date:
                start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
            else:
                # Default to 30 days ago
                start_ts = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
            
            if end_date:
                end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
            else:
                end_ts = int(datetime.now().timestamp() * 1000)
            
            logger.info(f"Downloading {symbol} from Bybit: interval={interval}")
            
            all_data = []
            current_end = end_ts
            
            # Bybit returns max 200 bars per request
            while current_end > start_ts:
                try:
                    # Get kline data
                    response = self.bybit_session.get_kline(
                        category="linear",  # For USDT perpetuals
                        symbol=symbol,
                        interval=interval,
                        end=current_end,
                        limit=200
                    )
                    
                    if response['retCode'] != 0:
                        logger.error(f"Bybit API error: {response['retMsg']}")
                        break
                    
                    klines = response['result']['list']
                    
                    if not klines:
                        break
                    
                    # Add to all_data (reverse because Bybit returns newest first)
                    all_data.extend(reversed(klines))
                    
                    # Update current_end to the oldest timestamp
                    current_end = int(klines[-1][0]) - 1
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error fetching Bybit data: {e}")
                    break
            
            if not all_data:
                logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # Convert data types
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Select required columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Sort by index (oldest first)
            df.sort_index(inplace=True)
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='last')]
            
            logger.info(f"Downloaded {len(df)} bars for {symbol} from Bybit")
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading from Bybit: {e}")
            return pd.DataFrame()
    
    def download(self,
                symbol: str,
                start_date: Optional[str] = None,
                end_date: Optional[str] = None,
                timeframe: str = '1m',
                use_cache: bool = True) -> pd.DataFrame:
        """
        Download historical data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT', 'AAPL')
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            timeframe: Timeframe ('1m', '5m', '15m', '1h', '1d')
            use_cache: Whether to use cached data if available
        
        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        if use_cache:
            cached_data = self._load_from_cache(symbol, timeframe, start_date, end_date)
            if cached_data is not None:
                logger.info(f"Loaded {symbol} data from cache")
                return cached_data
        
        # Download based on data source
        if self.data_source == 'bybit':
            data = self.download_bybit(symbol, start_date, end_date, timeframe)
        elif self.data_source == 'yfinance':
            data = self._download_yfinance(symbol, start_date, end_date, timeframe)
        elif self.data_source == 'ccxt':
            data = self._download_ccxt(symbol, start_date, end_date, timeframe)
        else:
            raise ValueError(f"Unsupported data source: {self.data_source}")
        
        # Save to cache
        if data is not None and not data.empty and use_cache:
            self._save_to_cache(data, symbol, timeframe, start_date, end_date)
        
        return data
    
    def _download_yfinance(self, 
                          symbol: str,
                          start_date: Optional[str],
                          end_date: Optional[str],
                          timeframe: str) -> pd.DataFrame:
        """
        Download data using Yahoo Finance.
        
        Args:
            symbol: Stock/crypto symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert timeframe to yfinance interval
            interval_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '60m',
                '4h': '1d',  # yfinance doesn't support 4h
                '1d': '1d',
                '1w': '1wk'
            }
            
            interval = interval_map.get(timeframe, '1d')
            
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                # Default to 60 days ago for minute data, 2 years for daily
                if 'm' in interval:
                    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
                else:
                    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            
            logger.info(f"Downloading {symbol} from Yahoo Finance: {start_date} to {end_date}, interval: {interval}")
            
            # Download data
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            data.columns = data.columns.str.lower()
            
            # Ensure we have all required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in data.columns:
                    logger.warning(f"Missing column {col} in data")
                    if col == 'volume':
                        data[col] = 0
                    else:
                        data[col] = data['close'] if 'close' in data.columns else 0
            
            # Clean data
            data = data[required_cols]
            data = data.dropna()
            
            logger.info(f"Downloaded {len(data)} bars for {symbol}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error downloading from Yahoo Finance: {e}")
            return pd.DataFrame()
    
    def _download_ccxt(self,
                      symbol: str,
                      start_date: Optional[str],
                      end_date: Optional[str],
                      timeframe: str) -> pd.DataFrame:
        """
        Download data using CCXT from cryptocurrency exchanges.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
        
        Returns:
            DataFrame with OHLCV data
        """
        if not self.exchange:
            logger.error("CCXT exchange not initialized")
            return pd.DataFrame()
        
        try:
            # Convert dates to timestamps
            if start_date:
                since = int(pd.Timestamp(start_date).timestamp() * 1000)
            else:
                since = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
            
            if end_date:
                end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
            else:
                end_ts = int(datetime.now().timestamp() * 1000)
            
            # Check if exchange supports fetching OHLCV
            if not self.exchange.has['fetchOHLCV']:
                logger.error(f"{self.exchange_name} doesn't support OHLCV data")
                return pd.DataFrame()
            
            # Fetch data in chunks (exchanges have limits)
            all_data = []
            limit = 1000  # Most exchanges limit to 1000 bars per request
            
            current_since = since
            
            while current_since < end_ts:
                logger.debug(f"Fetching {symbol} from {current_since}")
                
                # Rate limiting
                self._rate_limit()
                
                # Fetch OHLCV data
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=limit
                )
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                
                # Update current_since to the last timestamp
                current_since = ohlcv[-1][0] + 1
                
                # Break if we've reached the end
                if len(ohlcv) < limit:
                    break
            
            if not all_data:
                logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Filter by end date
            if end_date:
                df = df[df.index <= pd.Timestamp(end_date)]
            
            logger.info(f"Downloaded {len(df)} bars for {symbol} from {self.exchange_name}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading from CCXT: {e}")
            return pd.DataFrame()
    
    def _download_polygon(self,
                         symbol: str,
                         start_date: Optional[str],
                         end_date: Optional[str],
                         timeframe: str) -> pd.DataFrame:
        """
        Download data using Polygon.io API.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
        
        Returns:
            DataFrame with OHLCV data
        """
        if not self.api_key:
            logger.error("Polygon.io requires an API key")
            return pd.DataFrame()
        
        try:
            # Convert timeframe to Polygon format
            timeframe_map = {
                '1m': ('minute', 1),
                '5m': ('minute', 5),
                '15m': ('minute', 15),
                '30m': ('minute', 30),
                '1h': ('hour', 1),
                '4h': ('hour', 4),
                '1d': ('day', 1),
                '1w': ('week', 1)
            }
            
            if timeframe not in timeframe_map:
                logger.error(f"Unsupported timeframe for Polygon: {timeframe}")
                return pd.DataFrame()
            
            timespan, multiplier = timeframe_map[timeframe]
            
            # Set default dates
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Build API URL
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
            
            params = {
                'apiKey': self.api_key,
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000
            }
            
            # Make request
            self._rate_limit()
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                logger.error(f"Polygon API error: {response.status_code} - {response.text}")
                return pd.DataFrame()
            
            data = response.json()
            
            if 'results' not in data or not data['results']:
                logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data['results'])
            
            # Rename columns
            column_map = {
                't': 'timestamp',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            }
            df.rename(columns=column_map, inplace=True)
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Select required columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"Downloaded {len(df)} bars for {symbol} from Polygon.io")
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading from Polygon: {e}")
            return pd.DataFrame()
    
    def _download_alpaca(self,
                        symbol: str,
                        start_date: Optional[str],
                        end_date: Optional[str],
                        timeframe: str) -> pd.DataFrame:
        """
        Download data using Alpaca API.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
        
        Returns:
            DataFrame with OHLCV data
        """
        logger.warning("Alpaca data source not yet implemented")
        return pd.DataFrame()
    
    def download_multiple(self,
                         symbols: List[str],
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         timeframe: str = '1m') -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple symbols.
        
        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
        
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data_dict = {}
        
        for symbol in symbols:
            logger.info(f"Downloading data for {symbol}...")
            data = self.download(symbol, start_date, end_date, timeframe)
            
            if data is not None and not data.empty:
                data_dict[symbol] = data
            else:
                logger.warning(f"Failed to download data for {symbol}")
            
            # Small delay between downloads
            time.sleep(0.5)
        
        logger.info(f"Downloaded data for {len(data_dict)} out of {len(symbols)} symbols")
        
        return data_dict
    
    def update_data(self,
                   existing_data: pd.DataFrame,
                   symbol: str,
                   timeframe: str = '1m') -> pd.DataFrame:
        """
        Update existing data with latest bars.
        
        Args:
            existing_data: Existing DataFrame
            symbol: Symbol to update
            timeframe: Timeframe
        
        Returns:
            Updated DataFrame
        """
        if existing_data.empty:
            return self.download(symbol, timeframe=timeframe)
        
        # Get last timestamp
        last_timestamp = existing_data.index[-1]
        start_date = (last_timestamp + timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"Updating {symbol} data from {start_date}")
        
        # Download new data
        new_data = self.download(symbol, start_date=start_date, timeframe=timeframe, use_cache=False)
        
        if new_data is not None and not new_data.empty:
            # Combine data
            updated_data = pd.concat([existing_data, new_data])
            
            # Remove duplicates
            updated_data = updated_data[~updated_data.index.duplicated(keep='last')]
            
            # Sort by index
            updated_data.sort_index(inplace=True)
            
            logger.info(f"Updated with {len(new_data)} new bars")
            
            return updated_data
        
        return existing_data
    
    def resample_data(self,
                     data: pd.DataFrame,
                     target_timeframe: str) -> pd.DataFrame:
        """
        Resample data to a different timeframe.
        
        Args:
            data: Original DataFrame
            target_timeframe: Target timeframe
        
        Returns:
            Resampled DataFrame
        """
        # Convert timeframe to pandas resample format
        timeframe_map = {
            '1m': '1T',
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D',
            '1w': '1W'
        }
        
        resample_freq = timeframe_map.get(target_timeframe)
        if not resample_freq:
            logger.error(f"Unsupported timeframe: {target_timeframe}")
            return data
        
        # Resample
        resampled = data.resample(resample_freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Remove NaN rows
        resampled = resampled.dropna()
        
        logger.info(f"Resampled from {len(data)} to {len(resampled)} bars")
        
        return resampled
    
    def _rate_limit(self):
        """Implement rate limiting between requests."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_cache_filename(self,
                           symbol: str,
                           timeframe: str,
                           start_date: Optional[str],
                           end_date: Optional[str]) -> Path:
        """
        Generate cache filename for given parameters.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
        
        Returns:
            Path to cache file
        """
        # Clean symbol for filename
        clean_symbol = symbol.replace('/', '_').replace('\\', '_')
        
        # Create filename
        filename_parts = [
            clean_symbol,
            timeframe,
            start_date or 'none',
            end_date or 'none'
        ]
        
        filename = '_'.join(filename_parts) + '.pkl'
        
        return self.cache_dir / filename
    
    def _save_to_cache(self,
                      data: pd.DataFrame,
                      symbol: str,
                      timeframe: str,
                      start_date: Optional[str],
                      end_date: Optional[str]):
        """
        Save data to cache.
        
        Args:
            data: DataFrame to save
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
        """
        try:
            cache_file = self._get_cache_filename(symbol, timeframe, start_date, end_date)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.debug(f"Saved data to cache: {cache_file}")
            
        except Exception as e:
            logger.error(f"Failed to save to cache: {e}")
    
    def _load_from_cache(self,
                        symbol: str,
                        timeframe: str,
                        start_date: Optional[str],
                        end_date: Optional[str]) -> Optional[pd.DataFrame]:
        """
        Load data from cache.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
        
        Returns:
            Cached DataFrame or None
        """
        try:
            cache_file = self._get_cache_filename(symbol, timeframe, start_date, end_date)
            
            if not cache_file.exists():
                return None
            
            # Check if cache is recent (less than 1 day old for minute data, 7 days for daily)
            cache_age = time.time() - cache_file.stat().st_mtime
            
            if 'm' in timeframe and cache_age > 86400:  # 1 day for minute data
                logger.debug("Cache too old for minute data")
                return None
            elif 'd' in timeframe and cache_age > 604800:  # 7 days for daily data
                logger.debug("Cache too old for daily data")
                return None
            
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            logger.debug(f"Loaded data from cache: {cache_file}")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load from cache: {e}")
            return None
    
    def clear_cache(self):
        """Clear all cached data."""
        try:
            for cache_file in self.cache_dir.glob('*.pkl'):
                cache_file.unlink()
            
            logger.info("Cache cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols from the data source.
        
        Returns:
            List of available symbols
        """
        if self.data_source == 'ccxt' and self.exchange:
            return list(self.exchange.markets.keys())
        
        # For other sources, return common symbols
        return [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
            'SPY', 'QQQ', 'DIA', 'IWM',
            'BTC-USD', 'ETH-USD'
        ]


logger.info("DataDownloader module loaded successfully")