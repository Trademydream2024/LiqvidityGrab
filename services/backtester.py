"""
NY Liquidity Bot - Backtesting Module
Provides comprehensive backtesting capabilities for the liquidity grab strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
import json
import os

# Import strategy components
from .strategy import (
    LiquidityGrabStrategy,
    find_fractals,
    detect_sweep,
    compute_ob_entry,
    find_order_blocks,
    find_fvg,
    is_ny_session,
    calculate_position_size
)

logger = logging.getLogger(__name__)

@dataclass
class BacktestTrade:
    """Represents a single trade in backtest."""
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    trade_type: str  # 'long' or 'short'
    stop_loss: float
    take_profit: float
    pnl: float = 0.0
    pnl_percent: float = 0.0
    exit_reason: str = ""
    fees: float = 0.0
    slippage: float = 0.0
    
    def calculate_pnl(self, exit_price: float, fee_rate: float = 0.001):
        """Calculate PnL for the trade."""
        self.exit_price = exit_price
        
        # Calculate gross PnL
        if self.trade_type == 'long':
            gross_pnl = (exit_price - self.entry_price) * self.position_size
        else:  # short
            gross_pnl = (self.entry_price - exit_price) * self.position_size
        
        # Calculate fees (entry + exit)
        self.fees = (self.entry_price + exit_price) * self.position_size * fee_rate
        
        # Net PnL
        self.pnl = gross_pnl - self.fees
        self.pnl_percent = (self.pnl / (self.entry_price * self.position_size)) * 100
        
        return self.pnl

@dataclass
class BacktestResult:
    """Contains complete backtest results."""
    trades: List[BacktestTrade]
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    statistics: Dict[str, Any]
    signals: List[Dict[str, Any]]
    config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'trades': [self._trade_to_dict(t) for t in self.trades],
            'equity_curve': self.equity_curve.to_list(),
            'drawdown_curve': self.drawdown_curve.to_list(),
            'statistics': self.statistics,
            'signals': self.signals,
            'config': self.config
        }
    
    def _trade_to_dict(self, trade: BacktestTrade) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            'entry_time': trade.entry_time.isoformat() if trade.entry_time else None,
            'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'position_size': trade.position_size,
            'trade_type': trade.trade_type,
            'stop_loss': trade.stop_loss,
            'take_profit': trade.take_profit,
            'pnl': trade.pnl,
            'pnl_percent': trade.pnl_percent,
            'exit_reason': trade.exit_reason,
            'fees': trade.fees
        }

class Backtester:
    """
    Main backtesting engine for NY Liquidity Bot strategy.
    """
    
    def __init__(self, strategy: Optional[LiquidityGrabStrategy] = None, 
                 initial_capital: float = 10000.0,
                 fee_rate: float = 0.001,
                 slippage: float = 0.0001):
        """
        Initialize backtester.
        
        Args:
            strategy: Strategy instance to backtest
            initial_capital: Starting capital
            fee_rate: Trading fee rate (0.001 = 0.1%)
            slippage: Slippage rate
        """
        self.strategy = strategy or LiquidityGrabStrategy()
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        
        # Backtest state
        self.current_capital = initial_capital
        self.trades: List[BacktestTrade] = []
        self.open_trade: Optional[BacktestTrade] = None
        self.equity_curve = []
        self.drawdown_curve = []
        self.signals = []
        
        # Performance tracking
        self.max_equity = initial_capital
        self.max_drawdown = 0.0
        self.peak_equity = initial_capital
        
        logger.info(f"Backtester initialized with capital: ${initial_capital:,.2f}")
    
    def run(self, data: pd.DataFrame, 
            start_date: Optional[pd.Timestamp] = None,
            end_date: Optional[pd.Timestamp] = None,
            progress_callback: Optional[callable] = None) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            data: DataFrame with OHLC data
            start_date: Backtest start date
            end_date: Backtest end date
            progress_callback: Function to call with progress updates
        
        Returns:
            BacktestResult object
        """
        logger.info("Starting backtest...")
        
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        if len(data) < 100:
            logger.error("Insufficient data for backtest")
            return self._create_empty_result()
        
        # Reset state
        self.reset()
        
        # Set strategy account balance
        self.strategy.config['account_balance'] = self.current_capital
        
        # Process each bar
        for i in range(100, len(data)):
            # Get data slice up to current bar
            current_data = data.iloc[:i+1]
            current_bar = data.iloc[i]
            
            # Update progress
            if progress_callback and i % 100 == 0:
                progress = (i / len(data)) * 100
                progress_callback(progress)
            
            # Check for exit conditions first
            if self.open_trade:
                self._check_exit(current_bar, current_data.index[i])
            
            # Generate new signals only if no open position
            if not self.open_trade:
                signal = self.strategy.update(current_data)
                
                if signal:
                    self.signals.append(signal)
                    
                    # Validate signal and enter trade
                    if self._validate_signal(signal, current_bar):
                        self._enter_trade(signal, current_data.index[i])
            
            # Update equity curve
            self._update_equity(current_bar)
        
        # Close any remaining open trade
        if self.open_trade:
            last_bar = data.iloc[-1]
            self._force_exit(last_bar, data.index[-1], "end_of_data")
        
        # Calculate final statistics
        result = self._calculate_results(data)
        
        logger.info(f"Backtest completed. Total trades: {len(self.trades)}")
        
        return result
    
    def _validate_signal(self, signal: Dict[str, Any], current_bar: pd.Series) -> bool:
        """
        Validate trading signal before execution.
        
        Args:
            signal: Trading signal
            current_bar: Current price bar
        
        Returns:
            True if signal is valid
        """
        # Check if we have enough capital
        required_capital = signal['position_size'] * signal['entry_price']
        if required_capital > self.current_capital * 0.95:  # Use max 95% of capital
            logger.debug("Insufficient capital for trade")
            return False
        
        # Check if entry price is realistic
        if signal['type'] == 'long':
            if signal['entry_price'] < current_bar['low'] or signal['entry_price'] > current_bar['high'] * 1.01:
                logger.debug("Long entry price outside realistic range")
                return False
        else:  # short
            if signal['entry_price'] > current_bar['high'] or signal['entry_price'] < current_bar['low'] * 0.99:
                logger.debug("Short entry price outside realistic range")
                return False
        
        return True
    
    def _enter_trade(self, signal: Dict[str, Any], timestamp: pd.Timestamp):
        """
        Enter a new trade based on signal.
        
        Args:
            signal: Trading signal
            timestamp: Entry timestamp
        """
        # Apply slippage
        if signal['type'] == 'long':
            entry_price = signal['entry_price'] * (1 + self.slippage)
        else:
            entry_price = signal['entry_price'] * (1 - self.slippage)
        
        # Calculate position size based on available capital
        max_position_value = self.current_capital * 0.95
        max_position_size = max_position_value / entry_price
        position_size = min(signal['position_size'], max_position_size)
        
        # Create trade
        self.open_trade = BacktestTrade(
            entry_time=timestamp,
            exit_time=None,
            entry_price=entry_price,
            exit_price=None,
            position_size=position_size,
            trade_type=signal['type'],
            stop_loss=signal['stop_loss'],
            take_profit=signal['take_profit']
        )
        
        # Deduct from capital (for fees calculation)
        entry_fee = entry_price * position_size * self.fee_rate
        self.current_capital -= entry_fee
        
        logger.debug(f"Entered {signal['type']} trade at {entry_price:.5f}")
    
    def _check_exit(self, current_bar: pd.Series, timestamp: pd.Timestamp):
        """
        Check if current trade should be exited.
        
        Args:
            current_bar: Current price bar
            timestamp: Current timestamp
        """
        if not self.open_trade:
            return
        
        exit_price = None
        exit_reason = ""
        
        if self.open_trade.trade_type == 'long':
            # Check stop loss
            if current_bar['low'] <= self.open_trade.stop_loss:
                exit_price = self.open_trade.stop_loss
                exit_reason = "stop_loss"
            # Check take profit
            elif current_bar['high'] >= self.open_trade.take_profit:
                exit_price = self.open_trade.take_profit
                exit_reason = "take_profit"
        
        else:  # short
            # Check stop loss
            if current_bar['high'] >= self.open_trade.stop_loss:
                exit_price = self.open_trade.stop_loss
                exit_reason = "stop_loss"
            # Check take profit
            elif current_bar['low'] <= self.open_trade.take_profit:
                exit_price = self.open_trade.take_profit
                exit_reason = "take_profit"
        
        if exit_price:
            self._exit_trade(exit_price, timestamp, exit_reason)
    
    def _exit_trade(self, exit_price: float, timestamp: pd.Timestamp, reason: str):
        """
        Exit current trade.
        
        Args:
            exit_price: Exit price
            timestamp: Exit timestamp
            reason: Exit reason
        """
        if not self.open_trade:
            return
        
        # Apply slippage
        if reason == "stop_loss":
            if self.open_trade.trade_type == 'long':
                exit_price *= (1 - self.slippage)
            else:
                exit_price *= (1 + self.slippage)
        
        # Complete trade
        self.open_trade.exit_time = timestamp
        self.open_trade.exit_reason = reason
        
        # Calculate PnL
        pnl = self.open_trade.calculate_pnl(exit_price, self.fee_rate)
        
        # Update capital
        self.current_capital += self.open_trade.position_size * exit_price - (exit_price * self.open_trade.position_size * self.fee_rate)
        
        # Record trade
        self.trades.append(self.open_trade)
        
        logger.debug(f"Exited trade: {reason} at {exit_price:.5f}, PnL: {pnl:.2f}")
        
        # Clear open trade
        self.open_trade = None
    
    def _force_exit(self, current_bar: pd.Series, timestamp: pd.Timestamp, reason: str):
        """
        Force exit of open trade.
        
        Args:
            current_bar: Current price bar
            timestamp: Exit timestamp
            reason: Exit reason
        """
        if not self.open_trade:
            return
        
        exit_price = current_bar['close']
        self._exit_trade(exit_price, timestamp, reason)
    
    def _update_equity(self, current_bar: pd.Series):
        """
        Update equity curve with current position value.
        
        Args:
            current_bar: Current price bar
        """
        equity = self.current_capital
        
        # Add unrealized PnL if position is open
        if self.open_trade:
            current_price = current_bar['close']
            if self.open_trade.trade_type == 'long':
                unrealized_pnl = (current_price - self.open_trade.entry_price) * self.open_trade.position_size
            else:
                unrealized_pnl = (self.open_trade.entry_price - current_price) * self.open_trade.position_size
            
            equity += unrealized_pnl
        
        self.equity_curve.append(equity)
        
        # Update peak and drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        drawdown = (self.peak_equity - equity) / self.peak_equity * 100
        self.drawdown_curve.append(drawdown)
        
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
    
    def _calculate_results(self, data: pd.DataFrame) -> BacktestResult:
        """
        Calculate comprehensive backtest results.
        
        Args:
            data: Original data DataFrame
        
        Returns:
            BacktestResult object
        """
        if not self.trades:
            return self._create_empty_result()
        
        # Convert lists to Series
        equity_series = pd.Series(self.equity_curve, index=data.index[-len(self.equity_curve):])
        drawdown_series = pd.Series(self.drawdown_curve, index=data.index[-len(self.drawdown_curve):])
        
        # Calculate statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        total_pnl = sum(t.pnl for t in self.trades)
        
        # Calculate returns
        returns = equity_series.pct_change().dropna()
        
        # Sharpe ratio (assuming 252 trading days)
        sharpe_ratio = 0
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        
        # Win rate
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        
        # Average win/loss
        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Calculate trade duration statistics
        trade_durations = []
        for trade in self.trades:
            if trade.exit_time and trade.entry_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # in hours
                trade_durations.append(duration)
        
        avg_duration = sum(trade_durations) / len(trade_durations) if trade_durations else 0
        
        statistics = {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital * 100,
            'average_pnl': total_pnl / len(self.trades) if self.trades else 0,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'best_trade': max(t.pnl for t in self.trades) if self.trades else 0,
            'worst_trade': min(t.pnl for t in self.trades) if self.trades else 0,
            'avg_trade_duration': avg_duration,
            'total_fees': sum(t.fees for t in self.trades),
            'final_capital': self.current_capital,
            'initial_capital': self.initial_capital
        }
        
        return BacktestResult(
            trades=self.trades,
            equity_curve=equity_series,
            drawdown_curve=drawdown_series,
            statistics=statistics,
            signals=self.signals,
            config=self.strategy.config
        )
    
    def _create_empty_result(self) -> BacktestResult:
        """Create empty result when no trades executed."""
        return BacktestResult(
            trades=[],
            equity_curve=pd.Series([self.initial_capital]),
            drawdown_curve=pd.Series([0]),
            statistics={
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return': 0,
                'average_pnl': 0,
                'average_win': 0,
                'average_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'avg_trade_duration': 0,
                'total_fees': 0,
                'final_capital': self.initial_capital,
                'initial_capital': self.initial_capital
            },
            signals=[],
            config=self.strategy.config
        )
    
    def reset(self):
        """Reset backtester state."""
        self.current_capital = self.initial_capital
        self.trades = []
        self.open_trade = None
        self.equity_curve = []
        self.drawdown_curve = []
        self.signals = []
        self.max_equity = self.initial_capital
        self.max_drawdown = 0.0
        self.peak_equity = self.initial_capital
        self.strategy.reset()
        
        logger.debug("Backtester state reset")
    
    def save_results(self, filepath: str, results: BacktestResult):
        """
        Save backtest results to file.
        
        Args:
            filepath: Path to save results
            results: BacktestResult object
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert results to JSON-serializable format
        results_dict = results.to_dict()
        
        # Convert any remaining timestamps
        def convert_timestamps(obj):
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, pd.Series):
                return obj.to_list()
            elif isinstance(obj, dict):
                return {k: convert_timestamps(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_timestamps(item) for item in obj]
            return obj
        
        results_dict = convert_timestamps(results_dict)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def print_results(self, results: BacktestResult):
        """
        Print formatted backtest results.
        
        Args:
            results: BacktestResult object
        """
        stats = results.statistics
        
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
        print(f"\nCapital:")
        print(f"  Initial: ${stats['initial_capital']:,.2f}")
        print(f"  Final:   ${stats['final_capital']:,.2f}")
        print(f"  Return:  {stats['total_return']:.2f}%")
        
        print(f"\nTrades:")
        print(f"  Total:   {stats['total_trades']}")
        print(f"  Winners: {stats['winning_trades']} ({stats['win_rate']:.1f}%)")
        print(f"  Losers:  {stats['losing_trades']}")
        
        print(f"\nPerformance:")
        print(f"  Total PnL:      ${stats['total_pnl']:,.2f}")
        print(f"  Average PnL:    ${stats['average_pnl']:,.2f}")
        print(f"  Average Win:    ${stats['average_win']:,.2f}")
        print(f"  Average Loss:   ${stats['average_loss']:,.2f}")
        print(f"  Best Trade:     ${stats['best_trade']:,.2f}")
        print(f"  Worst Trade:    ${stats['worst_trade']:,.2f}")
        
        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown:   {stats['max_drawdown']:.2f}%")
        print(f"  Profit Factor:  {stats['profit_factor']:.2f}")
        print(f"  Sharpe Ratio:   {stats['sharpe_ratio']:.2f}")
        
        print(f"\nOther:")
        print(f"  Avg Duration:   {stats['avg_trade_duration']:.1f} hours")
        print(f"  Total Fees:     ${stats['total_fees']:,.2f}")
        
        print("="*60)


logger.info("Backtester module loaded successfully")