"""
NY Liquidity Bot - Complete Strategy Module
Implements ICT concepts: Fractals, Liquidity Sweeps, Order Blocks, FVG, Mitigation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, time, timezone
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Fractal:
    """Represents a fractal point"""
    index: int
    price: float
    time: pd.Timestamp
    type: str  # 'high' or 'low'
    strength: int  # Number of bars used for confirmation

@dataclass
class LiquiditySweep:
    """Represents a liquidity sweep"""
    index: int
    price: float
    time: pd.Timestamp
    type: str  # 'bullish' or 'bearish'
    swept_level: float
    volume: float
    strength: float  # How decisive the sweep was

@dataclass
class OrderBlock:
    """Represents an order block"""
    start_index: int
    end_index: int
    high: float
    low: float
    time: pd.Timestamp
    type: str  # 'bullish' or 'bearish'
    mitigated: bool = False
    strength: float = 0.0

def find_fractals(data: pd.DataFrame, period: int = 5, min_strength: int = 2) -> Dict[str, List[Fractal]]:
    """
    Identifies fractal highs and lows in price data.
    
    Args:
        data: DataFrame with OHLC data
        period: Number of bars to look back/forward (default 5)
        min_strength: Minimum number of lower/higher bars required
    
    Returns:
        Dictionary with 'highs' and 'lows' lists containing Fractal objects
    """
    if len(data) < period * 2 + 1:
        logger.warning(f"Insufficient data for fractal detection. Need at least {period * 2 + 1} bars")
        return {'highs': [], 'lows': []}
    
    fractals = {'highs': [], 'lows': []}
    
    # Ensure we have required columns
    required_cols = ['high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in data.columns:
            logger.error(f"Missing required column: {col}")
            return fractals
    
    # Find fractal highs (swing highs)
    for i in range(period, len(data) - period):
        current_high = data['high'].iloc[i]
        
        # Count bars with lower highs on each side
        left_lower = sum(1 for j in range(1, period + 1) if data['high'].iloc[i - j] < current_high)
        right_lower = sum(1 for j in range(1, period + 1) if data['high'].iloc[i + j] < current_high)
        
        # Check if it's a valid fractal high
        if left_lower >= min_strength and right_lower >= min_strength:
            fractal = Fractal(
                index=i,
                price=current_high,
                time=data.index[i] if isinstance(data.index, pd.DatetimeIndex) else pd.Timestamp.now(),
                type='high',
                strength=min(left_lower, right_lower)
            )
            fractals['highs'].append(fractal)
            logger.debug(f"Found fractal high at index {i}: {current_high}")
    
    # Find fractal lows (swing lows)
    for i in range(period, len(data) - period):
        current_low = data['low'].iloc[i]
        
        # Count bars with higher lows on each side
        left_higher = sum(1 for j in range(1, period + 1) if data['low'].iloc[i - j] > current_low)
        right_higher = sum(1 for j in range(1, period + 1) if data['low'].iloc[i + j] > current_low)
        
        # Check if it's a valid fractal low
        if left_higher >= min_strength and right_higher >= min_strength:
            fractal = Fractal(
                index=i,
                price=current_low,
                time=data.index[i] if isinstance(data.index, pd.DatetimeIndex) else pd.Timestamp.now(),
                type='low',
                strength=min(left_higher, right_higher)
            )
            fractals['lows'].append(fractal)
            logger.debug(f"Found fractal low at index {i}: {current_low}")
    
    return fractals

def detect_sweep(data: pd.DataFrame, fractals: Dict[str, List[Fractal]], 
                 threshold: float = 0.001, lookback: int = 20) -> List[LiquiditySweep]:
    """
    Detects liquidity sweeps of fractal levels.
    
    Args:
        data: DataFrame with OHLC data
        fractals: Dictionary of fractal highs and lows
        threshold: Minimum penetration required (as percentage)
        lookback: Number of bars to look back for fractal levels
    
    Returns:
        List of LiquiditySweep objects
    """
    sweeps = []
    
    if len(data) < 2:
        return sweeps
    
    # Check for bullish sweeps (sweep of lows)
    for i in range(1, len(data)):
        current_low = data['low'].iloc[i]
        current_close = data['close'].iloc[i]
        current_volume = data['volume'].iloc[i] if 'volume' in data.columns else 0
        
        # Look for recent fractal lows that might be swept
        for fractal_low in fractals['lows']:
            if fractal_low.index < i - lookback or fractal_low.index >= i:
                continue
                
            # Check if price swept below fractal low and closed above
            if current_low < fractal_low.price * (1 - threshold) and current_close > fractal_low.price:
                sweep_strength = abs(fractal_low.price - current_low) / fractal_low.price
                
                sweep = LiquiditySweep(
                    index=i,
                    price=current_low,
                    time=data.index[i] if isinstance(data.index, pd.DatetimeIndex) else pd.Timestamp.now(),
                    type='bullish',
                    swept_level=fractal_low.price,
                    volume=current_volume,
                    strength=sweep_strength
                )
                sweeps.append(sweep)
                logger.info(f"Bullish sweep detected at index {i}: swept {fractal_low.price}")
    
    # Check for bearish sweeps (sweep of highs)
    for i in range(1, len(data)):
        current_high = data['high'].iloc[i]
        current_close = data['close'].iloc[i]
        current_volume = data['volume'].iloc[i] if 'volume' in data.columns else 0
        
        # Look for recent fractal highs that might be swept
        for fractal_high in fractals['highs']:
            if fractal_high.index < i - lookback or fractal_high.index >= i:
                continue
                
            # Check if price swept above fractal high and closed below
            if current_high > fractal_high.price * (1 + threshold) and current_close < fractal_high.price:
                sweep_strength = abs(current_high - fractal_high.price) / fractal_high.price
                
                sweep = LiquiditySweep(
                    index=i,
                    price=current_high,
                    time=data.index[i] if isinstance(data.index, pd.DatetimeIndex) else pd.Timestamp.now(),
                    type='bearish',
                    swept_level=fractal_high.price,
                    volume=current_volume,
                    strength=sweep_strength
                )
                sweeps.append(sweep)
                logger.info(f"Bearish sweep detected at index {i}: swept {fractal_high.price}")
    
    return sweeps

def find_order_blocks(data: pd.DataFrame, lookback: int = 50, min_imbalance: float = 0.5) -> List[OrderBlock]:
    """
    Identifies order blocks in price data.
    
    Args:
        data: DataFrame with OHLC data
        lookback: Number of bars to analyze
        min_imbalance: Minimum imbalance ratio required
    
    Returns:
        List of OrderBlock objects
    """
    order_blocks = []
    
    if len(data) < 3:
        return order_blocks
    
    for i in range(2, min(len(data), lookback)):
        # Bullish order block: last down candle before up move
        if i < len(data) - 1:
            prev_candle = data.iloc[i-1]
            current_candle = data.iloc[i]
            next_candle = data.iloc[i+1]
            
            # Check for bullish OB
            if (current_candle['close'] < current_candle['open'] and  # Down candle
                next_candle['close'] > next_candle['open'] and  # Up candle
                next_candle['close'] > current_candle['high']):  # Strong move up
                
                ob = OrderBlock(
                    start_index=i,
                    end_index=i,
                    high=current_candle['high'],
                    low=current_candle['low'],
                    time=data.index[i] if isinstance(data.index, pd.DatetimeIndex) else pd.Timestamp.now(),
                    type='bullish',
                    strength=abs(next_candle['close'] - current_candle['low']) / current_candle['low']
                )
                order_blocks.append(ob)
                
            # Check for bearish OB
            elif (current_candle['close'] > current_candle['open'] and  # Up candle
                  next_candle['close'] < next_candle['open'] and  # Down candle
                  next_candle['close'] < current_candle['low']):  # Strong move down
                
                ob = OrderBlock(
                    start_index=i,
                    end_index=i,
                    high=current_candle['high'],
                    low=current_candle['low'],
                    time=data.index[i] if isinstance(data.index, pd.DatetimeIndex) else pd.Timestamp.now(),
                    type='bearish',
                    strength=abs(current_candle['high'] - next_candle['close']) / current_candle['high']
                )
                order_blocks.append(ob)
    
    return order_blocks

def compute_ob_entry(data: pd.DataFrame, sweep: Optional[LiquiditySweep], 
                    order_blocks: Optional[List[OrderBlock]] = None,
                    risk_reward: float = 2.0) -> Optional[Dict[str, Any]]:
    """
    Computes optimal entry point based on order blocks and sweeps.
    
    Args:
        data: DataFrame with OHLC data
        sweep: Detected liquidity sweep
        order_blocks: List of order blocks (optional)
        risk_reward: Desired risk/reward ratio
    
    Returns:
        Dictionary with entry details or None
    """
    if sweep is None or len(data) == 0:
        return None
    
    current_price = data['close'].iloc[-1]
    entry = None
    
    # If no order blocks provided, find them
    if order_blocks is None:
        order_blocks = find_order_blocks(data)
    
    # Find relevant order block for entry
    relevant_ob = None
    
    if sweep.type == 'bullish':
        # Look for bullish order blocks below current price
        for ob in order_blocks:
            if ob.type == 'bullish' and not ob.mitigated:
                if ob.high < current_price and ob.low < sweep.swept_level:
                    if relevant_ob is None or ob.strength > relevant_ob.strength:
                        relevant_ob = ob
    
    elif sweep.type == 'bearish':
        # Look for bearish order blocks above current price
        for ob in order_blocks:
            if ob.type == 'bearish' and not ob.mitigated:
                if ob.low > current_price and ob.high > sweep.swept_level:
                    if relevant_ob is None or ob.strength > relevant_ob.strength:
                        relevant_ob = ob
    
    # Calculate entry parameters
    if relevant_ob:
        if sweep.type == 'bullish':
            entry_price = relevant_ob.high
            stop_loss = relevant_ob.low * 0.999  # Just below OB
            take_profit = entry_price + (entry_price - stop_loss) * risk_reward
            
        else:  # bearish
            entry_price = relevant_ob.low
            stop_loss = relevant_ob.high * 1.001  # Just above OB
            take_profit = entry_price - (stop_loss - entry_price) * risk_reward
        
        entry = {
            'type': 'long' if sweep.type == 'bullish' else 'short',
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': risk_reward,
            'order_block': relevant_ob,
            'sweep': sweep,
            'timestamp': pd.Timestamp.now(),
            'confidence': min(sweep.strength * relevant_ob.strength * 100, 100)
        }
        
        logger.info(f"Entry signal generated: {entry['type']} at {entry_price:.5f}")
    
    return entry

def find_fvg(data: pd.DataFrame, min_gap_size: float = 0.001) -> List[Dict[str, Any]]:
    """
    Finds Fair Value Gaps (FVG) in price data.
    
    Args:
        data: DataFrame with OHLC data
        min_gap_size: Minimum gap size as percentage
    
    Returns:
        List of FVG dictionaries
    """
    fvgs = []
    
    if len(data) < 3:
        return fvgs
    
    for i in range(1, len(data) - 1):
        prev_candle = data.iloc[i-1]
        current_candle = data.iloc[i]
        next_candle = data.iloc[i+1]
        
        # Bullish FVG
        if next_candle['low'] > prev_candle['high']:
            gap_size = (next_candle['low'] - prev_candle['high']) / prev_candle['high']
            if gap_size >= min_gap_size:
                fvg = {
                    'type': 'bullish',
                    'index': i,
                    'top': next_candle['low'],
                    'bottom': prev_candle['high'],
                    'size': gap_size,
                    'filled': False,
                    'time': data.index[i] if isinstance(data.index, pd.DatetimeIndex) else pd.Timestamp.now()
                }
                fvgs.append(fvg)
        
        # Bearish FVG
        elif next_candle['high'] < prev_candle['low']:
            gap_size = (prev_candle['low'] - next_candle['high']) / next_candle['high']
            if gap_size >= min_gap_size:
                fvg = {
                    'type': 'bearish',
                    'index': i,
                    'top': prev_candle['low'],
                    'bottom': next_candle['high'],
                    'size': gap_size,
                    'filled': False,
                    'time': data.index[i] if isinstance(data.index, pd.DatetimeIndex) else pd.Timestamp.now()
                }
                fvgs.append(fvg)
    
    return fvgs

def is_ny_session(timestamp: pd.Timestamp) -> bool:
    """
    Checks if timestamp is within New York trading session.
    
    Args:
        timestamp: Timestamp to check
    
    Returns:
        True if within NY session (8:30 AM - 3:00 PM EST)
    """
    # Convert to EST/EDT
    ny_time = timestamp.tz_localize(None)  # Remove timezone if present
    
    # NY session: 8:30 AM - 3:00 PM EST (13:30 - 20:00 UTC)
    session_start = time(13, 30)  # 8:30 AM EST in UTC
    session_end = time(20, 0)     # 3:00 PM EST in UTC
    
    current_time = ny_time.time()
    
    return session_start <= current_time <= session_end

def calculate_position_size(account_balance: float, risk_percent: float, 
                          entry: float, stop_loss: float) -> float:
    """
    Calculates position size based on risk management.
    
    Args:
        account_balance: Total account balance
        risk_percent: Percentage of account to risk (e.g., 0.01 for 1%)
        entry: Entry price
        stop_loss: Stop loss price
    
    Returns:
        Position size in units
    """
    risk_amount = account_balance * risk_percent
    price_risk = abs(entry - stop_loss)
    
    if price_risk == 0:
        return 0
    
    position_size = risk_amount / price_risk
    
    return position_size

def validate_entry_conditions(data: pd.DataFrame, entry_signal: Dict[str, Any],
                             additional_filters: Dict[str, Any] = None) -> bool:
    """
    Validates if all entry conditions are met.
    
    Args:
        data: DataFrame with OHLC data
        entry_signal: Entry signal dictionary
        additional_filters: Additional validation filters
    
    Returns:
        True if entry is valid
    """
    if entry_signal is None:
        return False
    
    # Check if we're in NY session
    current_time = pd.Timestamp.now()
    if not is_ny_session(current_time):
        logger.debug("Outside NY session - entry not valid")
        return False
    
    # Check minimum confidence
    min_confidence = additional_filters.get('min_confidence', 60) if additional_filters else 60
    if entry_signal.get('confidence', 0) < min_confidence:
        logger.debug(f"Confidence {entry_signal.get('confidence', 0)} below minimum {min_confidence}")
        return False
    
    # Check if price is still valid
    current_price = data['close'].iloc[-1]
    entry_price = entry_signal['entry_price']
    max_deviation = additional_filters.get('max_deviation', 0.002) if additional_filters else 0.002
    
    if abs(current_price - entry_price) / entry_price > max_deviation:
        logger.debug(f"Price deviation too high: {abs(current_price - entry_price) / entry_price:.4f}")
        return False
    
    return True

# Additional helper functions for the strategy

def get_market_structure(data: pd.DataFrame, lookback: int = 50) -> str:
    """
    Determines current market structure (trending/ranging).
    
    Returns:
        'bullish', 'bearish', or 'ranging'
    """
    if len(data) < lookback:
        return 'ranging'
    
    recent_data = data.tail(lookback)
    
    # Calculate swing points
    highs = recent_data['high'].rolling(window=5).max()
    lows = recent_data['low'].rolling(window=5).min()
    
    # Check for higher highs and higher lows (bullish)
    if highs.iloc[-1] > highs.iloc[-10] and lows.iloc[-1] > lows.iloc[-10]:
        return 'bullish'
    
    # Check for lower highs and lower lows (bearish)
    elif highs.iloc[-1] < highs.iloc[-10] and lows.iloc[-1] < lows.iloc[-10]:
        return 'bearish'
    
    else:
        return 'ranging'

def check_volume_confirmation(data: pd.DataFrame, signal_index: int, 
                             lookback: int = 20) -> bool:
    """
    Checks if volume confirms the signal.
    
    Returns:
        True if volume is above average
    """
    if 'volume' not in data.columns or signal_index >= len(data):
        return True  # Skip volume check if not available
    
    avg_volume = data['volume'].iloc[max(0, signal_index - lookback):signal_index].mean()
    signal_volume = data['volume'].iloc[signal_index]
    
    return signal_volume > avg_volume * 1.2  # 20% above average

# Main strategy execution function
def execute_strategy(data: pd.DataFrame, config: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    """
    Main strategy execution function that combines all components.
    
    Args:
        data: DataFrame with OHLC data
        config: Strategy configuration parameters
    
    Returns:
        Entry signal or None
    """
    if config is None:
        config = {
            'fractal_period': 5,
            'sweep_threshold': 0.001,
            'risk_reward': 2.0,
            'min_confidence': 60
        }
    
    # Find market structure elements
    fractals = find_fractals(data, period=config.get('fractal_period', 5))
    sweeps = detect_sweep(data, fractals, threshold=config.get('sweep_threshold', 0.001))
    order_blocks = find_order_blocks(data)
    
    # Get the most recent sweep
    if sweeps:
        latest_sweep = sweeps[-1]
        
        # Compute entry based on sweep and order blocks
        entry_signal = compute_ob_entry(
            data, 
            latest_sweep, 
            order_blocks,
            risk_reward=config.get('risk_reward', 2.0)
        )
        
        # Validate entry
        if entry_signal and validate_entry_conditions(data, entry_signal, config):
            return entry_signal
    
    return None

logger.info("Strategy module loaded successfully with all required functions")


class LiquidityGrabStrategy:
    """
    Main strategy class for NY Liquidity Bot.
    Implements ICT concepts for liquidity grab detection and trading.
    """
    
    def __init__(self, config: Dict[str, Any] = None, exchange=None, analyzer=None):
        """
        Initialize the strategy with configuration.
        
        Args:
            config: Strategy configuration dictionary
            exchange: Exchange connection (optional)
            analyzer: Market analyzer (optional)
        """
        self.exchange = exchange
        self.analyzer = analyzer
        self.config = config or {
            'fractal_period': 5,
            'sweep_threshold': 0.001,
            'sweep_lookback': 20,
            'ob_lookback': 50,
            'min_imbalance': 0.5,
            'risk_reward': 2.0,
            'risk_percent': 0.01,  # 1% risk per trade
            'min_confidence': 60,
            'max_deviation': 0.002,
            'enable_fvg': True,
            'enable_volume_filter': True,
            'ny_session_only': True
        }
        
        self.fractals = {'highs': [], 'lows': []}
        self.sweeps = []
        self.order_blocks = []
        self.fvgs = []
        self.current_position = None
        self.trade_history = []
        
        logger.info(f"LiquidityGrabStrategy initialized with config: {self.config}")
    
    def update(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Update strategy with new data and generate signals.
        
        Args:
            data: DataFrame with OHLC data
        
        Returns:
            Trading signal dictionary or None
        """
        if len(data) < self.config['fractal_period'] * 2 + 1:
            logger.warning("Insufficient data for strategy update")
            return None
        
        # Update market structure components
        self.fractals = find_fractals(
            data, 
            period=self.config['fractal_period']
        )
        
        self.sweeps = detect_sweep(
            data, 
            self.fractals,
            threshold=self.config['sweep_threshold'],
            lookback=self.config['sweep_lookback']
        )
        
        self.order_blocks = find_order_blocks(
            data,
            lookback=self.config['ob_lookback'],
            min_imbalance=self.config['min_imbalance']
        )
        
        if self.config['enable_fvg']:
            self.fvgs = find_fvg(data)
        
        # Check for entry signal
        signal = self.generate_signal(data)
        
        return signal
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal based on current market conditions.
        
        Args:
            data: DataFrame with OHLC data
        
        Returns:
            Trading signal or None
        """
        # Check if we're in NY session if required
        if self.config['ny_session_only'] and not is_ny_session(pd.Timestamp.now()):
            return None
        
        # Don't generate new signal if we have an open position
        if self.current_position is not None:
            return None
        
        # Look for recent sweep
        if not self.sweeps:
            return None
        
        latest_sweep = self.sweeps[-1]
        
        # Ensure sweep is recent (within last 10 bars)
        if latest_sweep.index < len(data) - 10:
            return None
        
        # Generate entry based on sweep and order blocks
        entry = compute_ob_entry(
            data,
            latest_sweep,
            self.order_blocks,
            risk_reward=self.config['risk_reward']
        )
        
        if entry is None:
            return None
        
        # Apply additional filters
        if not self.validate_signal(data, entry):
            return None
        
        # Add volume confirmation if enabled
        if self.config['enable_volume_filter']:
            if not check_volume_confirmation(data, latest_sweep.index):
                logger.debug("Volume confirmation failed")
                return None
        
        # Calculate position size
        account_balance = self.config.get('account_balance', 10000)
        position_size = calculate_position_size(
            account_balance,
            self.config['risk_percent'],
            entry['entry_price'],
            entry['stop_loss']
        )
        
        # Add additional signal information
        entry['position_size'] = position_size
        entry['market_structure'] = get_market_structure(data)
        entry['signal_time'] = pd.Timestamp.now()
        entry['sweep_index'] = latest_sweep.index
        
        logger.info(f"Signal generated: {entry['type']} at {entry['entry_price']:.5f}")
        
        return entry
    
    def validate_signal(self, data: pd.DataFrame, signal: Dict[str, Any]) -> bool:
        """
        Validate trading signal against strategy rules.
        
        Args:
            data: DataFrame with OHLC data
            signal: Trading signal to validate
        
        Returns:
            True if signal is valid
        """
        # Check confidence threshold
        if signal.get('confidence', 0) < self.config['min_confidence']:
            logger.debug(f"Signal confidence {signal['confidence']:.1f}% below threshold")
            return False
        
        # Check price deviation
        current_price = data['close'].iloc[-1]
        max_deviation = self.config['max_deviation']
        
        price_deviation = abs(current_price - signal['entry_price']) / signal['entry_price']
        if price_deviation > max_deviation:
            logger.debug(f"Price deviation {price_deviation:.4f} exceeds maximum")
            return False
        
        # Check market structure alignment
        market_structure = get_market_structure(data)
        
        if signal['type'] == 'long' and market_structure == 'bearish':
            logger.debug("Long signal in bearish market structure")
            return False
        
        if signal['type'] == 'short' and market_structure == 'bullish':
            logger.debug("Short signal in bullish market structure")
            return False
        
        return True
    
    def execute_trade(self, signal: Dict[str, Any]) -> bool:
        """
        Execute trade based on signal.
        
        Args:
            signal: Trading signal
        
        Returns:
            True if trade executed successfully
        """
        if self.current_position is not None:
            logger.warning("Cannot execute trade - position already open")
            return False
        
        self.current_position = {
            'type': signal['type'],
            'entry_price': signal['entry_price'],
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'position_size': signal['position_size'],
            'entry_time': pd.Timestamp.now(),
            'signal': signal
        }
        
        logger.info(f"Trade executed: {signal['type']} {signal['position_size']:.2f} units at {signal['entry_price']:.5f}")
        
        return True
    
    def check_exit_conditions(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Check if current position should be closed.
        
        Args:
            data: DataFrame with OHLC data
        
        Returns:
            Exit signal or None
        """
        if self.current_position is None:
            return None
        
        current_price = data['close'].iloc[-1]
        position = self.current_position
        
        exit_signal = None
        
        # Check stop loss
        if position['type'] == 'long':
            if current_price <= position['stop_loss']:
                exit_signal = {
                    'reason': 'stop_loss',
                    'exit_price': position['stop_loss'],
                    'pnl': (position['stop_loss'] - position['entry_price']) * position['position_size']
                }
            elif current_price >= position['take_profit']:
                exit_signal = {
                    'reason': 'take_profit',
                    'exit_price': position['take_profit'],
                    'pnl': (position['take_profit'] - position['entry_price']) * position['position_size']
                }
        
        else:  # short position
            if current_price >= position['stop_loss']:
                exit_signal = {
                    'reason': 'stop_loss',
                    'exit_price': position['stop_loss'],
                    'pnl': (position['entry_price'] - position['stop_loss']) * position['position_size']
                }
            elif current_price <= position['take_profit']:
                exit_signal = {
                    'reason': 'take_profit',
                    'exit_price': position['take_profit'],
                    'pnl': (position['entry_price'] - position['take_profit']) * position['position_size']
                }
        
        if exit_signal:
            exit_signal['exit_time'] = pd.Timestamp.now()
            exit_signal['position'] = position.copy()
            
        return exit_signal
    
    def close_position(self, exit_signal: Dict[str, Any]) -> bool:
        """
        Close current position.
        
        Args:
            exit_signal: Exit signal with details
        
        Returns:
            True if position closed successfully
        """
        if self.current_position is None:
            logger.warning("No position to close")
            return False
        
        # Record trade in history
        trade_record = {
            'entry': self.current_position,
            'exit': exit_signal,
            'duration': exit_signal['exit_time'] - self.current_position['entry_time'],
            'pnl': exit_signal['pnl']
        }
        
        self.trade_history.append(trade_record)
        
        logger.info(f"Position closed: {exit_signal['reason']} at {exit_signal['exit_price']:.5f}, PnL: {exit_signal['pnl']:.2f}")
        
        self.current_position = None
        
        return True
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """
        Get strategy performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'average_pnl': 0,
                'best_trade': 0,
                'worst_trade': 0
            }
        
        pnls = [trade['pnl'] for trade in self.trade_history]
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        stats = {
            'total_trades': len(self.trade_history),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trade_history) * 100 if self.trade_history else 0,
            'total_pnl': sum(pnls),
            'average_pnl': sum(pnls) / len(pnls) if pnls else 0,
            'best_trade': max(pnls) if pnls else 0,
            'worst_trade': min(pnls) if pnls else 0,
            'current_position': self.current_position is not None
        }
        
        return stats
    
    def reset(self):
        """Reset strategy state."""
        self.fractals = {'highs': [], 'lows': []}
        self.sweeps = []
        self.order_blocks = []
        self.fvgs = []
        self.current_position = None
        self.trade_history = []
        logger.info("Strategy state reset")
    
    def save_state(self, filepath: str):
        """
        Save strategy state to file.
        
        Args:
            filepath: Path to save state
        """
        import json
        
        state = {
            'config': self.config,
            'trade_history': self.trade_history,
            'current_position': self.current_position,
            'stats': self.get_strategy_stats()
        }
        
        # Convert timestamps to strings for JSON serialization
        def convert_timestamps(obj):
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, pd.Timedelta):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_timestamps(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_timestamps(item) for item in obj]
            return obj
        
        state = convert_timestamps(state)
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Strategy state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """
        Load strategy state from file.
        
        Args:
            filepath: Path to load state from
        """
        import json
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.config = state.get('config', self.config)
        self.trade_history = state.get('trade_history', [])
        self.current_position = state.get('current_position', None)
        
        logger.info(f"Strategy state loaded from {filepath}")
    
    def reset_daily_stats(self):
        """Reset daily statistics and limits."""
        self.daily_pnl = 0
        self.daily_trades = 0
        self.daily_losses = 0
        self.last_reset_date = pd.Timestamp.now().date()
        logger.info("Daily statistics reset")
    
    def check_daily_limits(self) -> bool:
        """Check if daily loss limits have been reached."""
        # Reset if new day
        current_date = pd.Timestamp.now().date()
        if hasattr(self, 'last_reset_date') and current_date > self.last_reset_date:
            self.reset_daily_stats()
        
        # Check daily loss limit
        max_daily_loss = self.config.get('max_daily_risk', 5.0) / 100
        account_balance = self.config.get('account_balance', 10000)
        max_loss_amount = account_balance * max_daily_loss
        
        if hasattr(self, 'daily_pnl') and self.daily_pnl < -max_loss_amount:
            logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2f}")
            return False
        
        # Check max positions
        max_positions = self.config.get('max_positions', 3)
        if self.current_position is not None:
            # For simplicity, we're tracking single position
            # Could be extended for multiple positions
            return True
        
        return True
    
    def get_current_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get current trading signals from the strategy."""
        signals = []
        
        # Update strategy with latest data
        signal = self.update(data)
        
        if signal:
            signals.append({
                'symbol': signal.get('symbol', 'UNKNOWN'),
                'type': signal['type'],
                'entry_price': signal['entry_price'],
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'confidence': signal['confidence'],
                'timestamp': pd.Timestamp.now()
            })
        
        return signals
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive number)
        
        Returns:
            Optimal fraction of capital to risk
        """
        if avg_loss == 0:
            return 0
        
        # Kelly formula: f = (p * b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        kelly_fraction = (p * b - q) / b
        
        # Apply Kelly fraction with safety factor (usually 0.25 to 0.5 of full Kelly)
        safety_factor = 0.25
        safe_kelly = kelly_fraction * safety_factor
        
        # Cap at maximum risk per trade
        max_risk = self.config.get('risk_percent', 0.01)
        
        return min(max(safe_kelly, 0), max_risk)
    
    def update_performance_metrics(self, trade_result: Dict[str, Any]):
        """Update strategy performance metrics after a trade."""
        if not hasattr(self, 'performance_metrics'):
            self.performance_metrics = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'current_streak': 0,
                'best_streak': 0,
                'worst_streak': 0
            }
        
        # Update trade counts
        self.performance_metrics['total_trades'] += 1
        
        pnl = trade_result.get('pnl', 0)
        self.performance_metrics['total_pnl'] += pnl
        
        if pnl > 0:
            self.performance_metrics['winning_trades'] += 1
            # Update winning streak
            if self.performance_metrics['current_streak'] >= 0:
                self.performance_metrics['current_streak'] += 1
            else:
                self.performance_metrics['current_streak'] = 1
                
            # Track best streak
            if self.performance_metrics['current_streak'] > self.performance_metrics['best_streak']:
                self.performance_metrics['best_streak'] = self.performance_metrics['current_streak']
        else:
            self.performance_metrics['losing_trades'] += 1
            # Update losing streak
            if self.performance_metrics['current_streak'] <= 0:
                self.performance_metrics['current_streak'] -= 1
            else:
                self.performance_metrics['current_streak'] = -1
                
            # Track worst streak
            if self.performance_metrics['current_streak'] < self.performance_metrics['worst_streak']:
                self.performance_metrics['worst_streak'] = self.performance_metrics['current_streak']
        
        # Calculate win rate
        if self.performance_metrics['total_trades'] > 0:
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['winning_trades'] / 
                self.performance_metrics['total_trades'] * 100
            )
        
        # Update daily stats if they exist
        if hasattr(self, 'daily_pnl'):
            self.daily_pnl += pnl
        if hasattr(self, 'daily_trades'):
            self.daily_trades += 1
        
        logger.info(f"Performance updated: Win Rate: {self.performance_metrics['win_rate']:.1f}%, Total PnL: {self.performance_metrics['total_pnl']:.2f}")
    
    def should_trade(self, current_time: pd.Timestamp = None) -> bool:
        """
        Determine if trading should be allowed based on various conditions.
        
        Args:
            current_time: Current timestamp
        
        Returns:
            True if trading is allowed
        """
        if current_time is None:
            current_time = pd.Timestamp.now()
        
        # Check if within NY session if required
        if self.config.get('ny_session_only', True):
            if not is_ny_session(current_time):
                return False
        
        # Check daily limits
        if not self.check_daily_limits():
            return False
        
        # Check if we have enough historical performance
        if hasattr(self, 'performance_metrics'):
            min_win_rate = self.config.get('min_win_rate', 40)
            if self.performance_metrics['total_trades'] >= 20:  # Need sample size
                if self.performance_metrics['win_rate'] < min_win_rate:
                    logger.warning(f"Win rate {self.performance_metrics['win_rate']:.1f}% below minimum {min_win_rate}%")
                    return False
        
        # Check for current positions
        max_positions = self.config.get('max_positions', 3)
        if self.current_position is not None:
            # Simple check for single position
            # Could be extended for multiple positions
            return False
        
        return True
    
    def adjust_risk_by_performance(self) -> float:
        """
        Dynamically adjust risk based on recent performance.
        
        Returns:
            Adjusted risk percentage
        """
        base_risk = self.config.get('risk_percent', 0.01)
        
        if not hasattr(self, 'performance_metrics'):
            return base_risk
        
        # Reduce risk during losing streaks
        if self.performance_metrics.get('current_streak', 0) <= -3:
            return base_risk * 0.5  # Half risk after 3 losses
        
        # Increase risk during winning streaks (carefully)
        if self.performance_metrics.get('current_streak', 0) >= 5:
            return base_risk * 1.25  # 25% more risk after 5 wins
        
        # Use Kelly Criterion if we have enough data
        if self.performance_metrics['total_trades'] >= 30:
            win_rate = self.performance_metrics['win_rate'] / 100
            if self.performance_metrics['winning_trades'] > 0:
                avg_win = (self.performance_metrics['total_pnl'] / 
                          self.performance_metrics['winning_trades'])
            else:
                avg_win = 0
                
            if self.performance_metrics['losing_trades'] > 0:
                avg_loss = abs(self.performance_metrics['total_pnl'] / 
                              self.performance_metrics['losing_trades'])
            else:
                avg_loss = 1
            
            if avg_win > 0 and avg_loss > 0:
                kelly_risk = self.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
                return kelly_risk
        
        return base_risk