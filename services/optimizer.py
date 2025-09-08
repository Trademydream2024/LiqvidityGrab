"""
NY Liquidity Bot - Strategy Optimizer Module
Optimizes strategy parameters using various optimization techniques.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json
import os
from itertools import product
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Import necessary components
from .backtester import Backtester, BacktestResult
from .strategy import LiquidityGrabStrategy

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Contains optimization results."""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    optimization_time: float
    convergence_history: List[float]
    parameter_importance: Dict[str, float]

class Optimizer:
    """
    Strategy parameter optimizer for NY Liquidity Bot.
    Supports grid search, random search, and Bayesian optimization.
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 initial_capital: float = 10000.0,
                 optimization_metric: str = 'sharpe_ratio',
                 n_jobs: int = 1):
        """
        Initialize optimizer.
        
        Args:
            data: Historical data for optimization
            initial_capital: Starting capital for backtests
            optimization_metric: Metric to optimize ('sharpe_ratio', 'total_return', 'profit_factor')
            n_jobs: Number of parallel jobs (-1 for all CPU cores)
        """
        self.data = data
        self.initial_capital = initial_capital
        self.optimization_metric = optimization_metric
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        
        # Results storage
        self.results_history = []
        self.best_params = None
        self.best_score = -float('inf')
        
        logger.info(f"Optimizer initialized with {len(data)} data points, optimizing for {optimization_metric}")
    
    def grid_search(self, 
                   param_grid: Dict[str, List[Any]],
                   validation_split: float = 0.2,
                   verbose: bool = True) -> OptimizationResult:
        """
        Perform grid search optimization.
        
        Args:
            param_grid: Dictionary of parameters and their values to test
            validation_split: Fraction of data to use for validation
            verbose: Print progress
        
        Returns:
            OptimizationResult object
        """
        logger.info("Starting grid search optimization...")
        start_time = datetime.now()
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        total_combinations = len(param_combinations)
        logger.info(f"Testing {total_combinations} parameter combinations")
        
        # Split data
        split_index = int(len(self.data) * (1 - validation_split))
        train_data = self.data.iloc[:split_index]
        val_data = self.data.iloc[split_index:]
        
        # Test each combination
        results = []
        convergence_history = []
        
        for i, params_tuple in enumerate(param_combinations):
            params = dict(zip(param_names, params_tuple))
            
            if verbose and i % 10 == 0:
                print(f"Progress: {i}/{total_combinations} ({i/total_combinations*100:.1f}%)")
            
            # Run backtest with these parameters
            score = self._evaluate_parameters(params, train_data, val_data)
            
            results.append({
                'params': params,
                'score': score,
                'iteration': i
            })
            
            # Track best score
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                
            convergence_history.append(self.best_score)
        
        # Calculate parameter importance
        parameter_importance = self._calculate_parameter_importance(results)
        
        # Calculate optimization time
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Grid search completed in {optimization_time:.2f} seconds")
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best params: {self.best_params}")
        
        return OptimizationResult(
            best_params=self.best_params,
            best_score=self.best_score,
            all_results=results,
            optimization_time=optimization_time,
            convergence_history=convergence_history,
            parameter_importance=parameter_importance
        )
    
    def random_search(self,
                     param_distributions: Dict[str, Tuple[float, float]],
                     n_iterations: int = 100,
                     validation_split: float = 0.2,
                     verbose: bool = True) -> OptimizationResult:
        """
        Perform random search optimization.
        
        Args:
            param_distributions: Dictionary of parameters and their (min, max) ranges
            n_iterations: Number of random combinations to test
            validation_split: Fraction of data to use for validation
            verbose: Print progress
        
        Returns:
            OptimizationResult object
        """
        logger.info(f"Starting random search optimization with {n_iterations} iterations...")
        start_time = datetime.now()
        
        # Split data
        split_index = int(len(self.data) * (1 - validation_split))
        train_data = self.data.iloc[:split_index]
        val_data = self.data.iloc[split_index:]
        
        results = []
        convergence_history = []
        
        for i in range(n_iterations):
            # Generate random parameters
            params = {}
            for param_name, (min_val, max_val) in param_distributions.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    params[param_name] = np.random.uniform(min_val, max_val)
            
            if verbose and i % 10 == 0:
                print(f"Progress: {i}/{n_iterations} ({i/n_iterations*100:.1f}%)")
            
            # Evaluate parameters
            score = self._evaluate_parameters(params, train_data, val_data)
            
            results.append({
                'params': params,
                'score': score,
                'iteration': i
            })
            
            # Track best score
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
            
            convergence_history.append(self.best_score)
        
        # Calculate parameter importance
        parameter_importance = self._calculate_parameter_importance(results)
        
        # Calculate optimization time
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Random search completed in {optimization_time:.2f} seconds")
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best params: {self.best_params}")
        
        return OptimizationResult(
            best_params=self.best_params,
            best_score=self.best_score,
            all_results=results,
            optimization_time=optimization_time,
            convergence_history=convergence_history,
            parameter_importance=parameter_importance
        )
    
    def walk_forward_optimization(self,
                                 param_grid: Dict[str, List[Any]],
                                 window_size: int = 1000,
                                 step_size: int = 100,
                                 verbose: bool = True) -> Dict[str, Any]:
        """
        Perform walk-forward optimization.
        
        Args:
            param_grid: Dictionary of parameters and their values to test
            window_size: Size of training window in bars
            step_size: Step size for moving window
            verbose: Print progress
        
        Returns:
            Dictionary with walk-forward results
        """
        logger.info("Starting walk-forward optimization...")
        
        windows = []
        window_results = []
        
        # Generate windows
        for start_idx in range(0, len(self.data) - window_size, step_size):
            end_idx = start_idx + window_size
            
            if end_idx + step_size > len(self.data):
                break
            
            train_data = self.data.iloc[start_idx:end_idx]
            test_data = self.data.iloc[end_idx:end_idx + step_size]
            
            windows.append({
                'train_start': start_idx,
                'train_end': end_idx,
                'test_start': end_idx,
                'test_end': end_idx + step_size
            })
            
            # Optimize on training window
            window_optimizer = Optimizer(
                train_data,
                self.initial_capital,
                self.optimization_metric,
                self.n_jobs
            )
            
            result = window_optimizer.grid_search(param_grid, validation_split=0, verbose=False)
            
            # Test on out-of-sample data
            test_score = self._evaluate_parameters(result.best_params, test_data, None)
            
            window_results.append({
                'window': windows[-1],
                'best_params': result.best_params,
                'in_sample_score': result.best_score,
                'out_of_sample_score': test_score
            })
            
            if verbose:
                print(f"Window {len(window_results)}: IS={result.best_score:.4f}, OOS={test_score:.4f}")
        
        # Calculate overall statistics
        in_sample_scores = [r['in_sample_score'] for r in window_results]
        out_of_sample_scores = [r['out_of_sample_score'] for r in window_results]
        
        results = {
            'windows': windows,
            'window_results': window_results,
            'avg_in_sample_score': np.mean(in_sample_scores),
            'avg_out_of_sample_score': np.mean(out_of_sample_scores),
            'stability_ratio': np.mean(out_of_sample_scores) / np.mean(in_sample_scores) if np.mean(in_sample_scores) > 0 else 0,
            'consistency': 1 - (np.std(out_of_sample_scores) / np.mean(out_of_sample_scores)) if np.mean(out_of_sample_scores) > 0 else 0
        }
        
        logger.info(f"Walk-forward completed. Stability ratio: {results['stability_ratio']:.4f}")
        
        return results
    
    def _evaluate_parameters(self, 
                            params: Dict[str, Any],
                            train_data: pd.DataFrame,
                            val_data: Optional[pd.DataFrame] = None) -> float:
        """
        Evaluate a set of parameters using backtesting.
        
        Args:
            params: Strategy parameters to test
            train_data: Training data
            val_data: Validation data (optional)
        
        Returns:
            Score based on optimization metric
        """
        try:
            # Create strategy with given parameters
            strategy_config = {
                'fractal_period': params.get('fractal_period', 5),
                'sweep_threshold': params.get('sweep_threshold', 0.001),
                'sweep_lookback': params.get('sweep_lookback', 20),
                'ob_lookback': params.get('ob_lookback', 50),
                'min_imbalance': params.get('min_imbalance', 0.5),
                'risk_reward': params.get('risk_reward', 2.0),
                'risk_percent': params.get('risk_percent', 0.01),
                'min_confidence': params.get('min_confidence', 60),
                'max_deviation': params.get('max_deviation', 0.002),
                'enable_fvg': params.get('enable_fvg', True),
                'enable_volume_filter': params.get('enable_volume_filter', True),
                'ny_session_only': params.get('ny_session_only', True)
            }
            
            strategy = LiquidityGrabStrategy(strategy_config)
            
            # Run backtest on training data
            backtester = Backtester(strategy, self.initial_capital)
            train_result = backtester.run(train_data)
            
            # If validation data provided, test on it
            if val_data is not None and len(val_data) > 100:
                backtester.reset()
                val_result = backtester.run(val_data)
                
                # Combine scores (weighted average)
                train_score = self._get_metric_value(train_result.statistics)
                val_score = self._get_metric_value(val_result.statistics)
                
                # Weight validation more heavily to prevent overfitting
                score = 0.3 * train_score + 0.7 * val_score
            else:
                score = self._get_metric_value(train_result.statistics)
            
            # Penalize strategies with too few trades
            if train_result.statistics['total_trades'] < 10:
                score *= 0.5
            
            return score
            
        except Exception as e:
            logger.error(f"Error evaluating parameters: {e}")
            return -float('inf')
    
    def _get_metric_value(self, statistics: Dict[str, Any]) -> float:
        """
        Get the value of the optimization metric from statistics.
        
        Args:
            statistics: Backtest statistics
        
        Returns:
            Metric value
        """
        if self.optimization_metric == 'sharpe_ratio':
            return statistics.get('sharpe_ratio', 0)
        elif self.optimization_metric == 'total_return':
            return statistics.get('total_return', 0)
        elif self.optimization_metric == 'profit_factor':
            return statistics.get('profit_factor', 0)
        elif self.optimization_metric == 'win_rate':
            return statistics.get('win_rate', 0)
        elif self.optimization_metric == 'max_drawdown':
            # For drawdown, lower is better, so return negative
            return -statistics.get('max_drawdown', 100)
        else:
            # Custom metric: combination of multiple factors
            sharpe = statistics.get('sharpe_ratio', 0)
            profit_factor = statistics.get('profit_factor', 0)
            win_rate = statistics.get('win_rate', 0) / 100
            max_dd = statistics.get('max_drawdown', 100)
            
            # Weighted combination
            score = (sharpe * 0.3 + 
                    profit_factor * 0.3 + 
                    win_rate * 0.2 - 
                    max_dd * 0.002)
            
            return score
    
    def _calculate_parameter_importance(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate importance of each parameter based on results.
        
        Args:
            results: List of optimization results
        
        Returns:
            Dictionary of parameter importance scores
        """
        if not results:
            return {}
        
        # Get all parameter names
        param_names = list(results[0]['params'].keys())
        importance = {}
        
        for param_name in param_names:
            # Calculate correlation between parameter values and scores
            param_values = [r['params'][param_name] for r in results]
            scores = [r['score'] for r in results]
            
            # Convert to numeric if possible
            try:
                param_values = [float(v) for v in param_values]
                
                # Calculate correlation
                if len(set(param_values)) > 1:  # Check if there's variation
                    correlation = abs(np.corrcoef(param_values, scores)[0, 1])
                    importance[param_name] = correlation if not np.isnan(correlation) else 0
                else:
                    importance[param_name] = 0
            except:
                importance[param_name] = 0
        
        # Normalize importance scores
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance
    
    def optimize_robust(self,
                       param_distributions: Dict[str, Tuple[float, float]],
                       n_iterations: int = 50,
                       n_bootstrap: int = 10) -> OptimizationResult:
        """
        Perform robust optimization using bootstrap sampling.
        
        Args:
            param_distributions: Parameter ranges
            n_iterations: Number of parameter combinations to test
            n_bootstrap: Number of bootstrap samples
        
        Returns:
            OptimizationResult with robust parameters
        """
        logger.info(f"Starting robust optimization with {n_bootstrap} bootstrap samples...")
        
        all_results = []
        
        for bootstrap_idx in range(n_bootstrap):
            # Create bootstrap sample
            bootstrap_data = self.data.sample(n=len(self.data), replace=True).sort_index()
            
            # Create optimizer for this sample
            bootstrap_optimizer = Optimizer(
                bootstrap_data,
                self.initial_capital,
                self.optimization_metric,
                self.n_jobs
            )
            
            # Run optimization
            result = bootstrap_optimizer.random_search(
                param_distributions,
                n_iterations=n_iterations // n_bootstrap,
                verbose=False
            )
            
            all_results.extend(result.all_results)
        
        # Aggregate results
        param_scores = {}
        
        for result in all_results:
            param_key = str(result['params'])
            if param_key not in param_scores:
                param_scores[param_key] = []
            param_scores[param_key].append(result['score'])
        
        # Find most consistent parameters
        best_params = None
        best_score = -float('inf')
        
        for param_key, scores in param_scores.items():
            # Use mean score minus standard deviation for robustness
            robust_score = np.mean(scores) - np.std(scores)
            
            if robust_score > best_score:
                best_score = robust_score
                best_params = eval(param_key)
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_time=0,
            convergence_history=[],
            parameter_importance={}
        )
    
    def save_results(self, results: OptimizationResult, filepath: str):
        """
        Save optimization results to file.
        
        Args:
            results: OptimizationResult object
            filepath: Path to save results
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert to JSON-serializable format
        results_dict = {
            'best_params': results.best_params,
            'best_score': results.best_score,
            'optimization_time': results.optimization_time,
            'parameter_importance': results.parameter_importance,
            'n_iterations': len(results.all_results),
            'convergence_history': results.convergence_history[-100:],  # Last 100 for file size
            'top_10_results': sorted(results.all_results, 
                                    key=lambda x: x['score'], 
                                    reverse=True)[:10]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Optimization results saved to {filepath}")
    
    def print_results(self, results: OptimizationResult):
        """
        Print formatted optimization results.
        
        Args:
            results: OptimizationResult object
        """
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS")
        print("="*60)
        
        print(f"\nBest Parameters:")
        for param, value in results.best_params.items():
            if isinstance(value, float):
                print(f"  {param}: {value:.6f}")
            else:
                print(f"  {param}: {value}")
        
        print(f"\nBest Score: {results.best_score:.4f}")
        print(f"Optimization Time: {results.optimization_time:.2f} seconds")
        print(f"Total Iterations: {len(results.all_results)}")
        
        if results.parameter_importance:
            print("\nParameter Importance:")
            sorted_importance = sorted(results.parameter_importance.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True)
            for param, importance in sorted_importance:
                print(f"  {param}: {importance:.2%}")
        
        # Top 5 results
        print("\nTop 5 Configurations:")
        sorted_results = sorted(results.all_results, 
                              key=lambda x: x['score'], 
                              reverse=True)[:5]
        
        for i, result in enumerate(sorted_results, 1):
            print(f"\n  #{i} Score: {result['score']:.4f}")
            for param, value in result['params'].items():
                if isinstance(value, float):
                    print(f"    {param}: {value:.6f}")
                else:
                    print(f"    {param}: {value}")
        
        print("="*60)


# Standalone backtest function for compatibility
def backtest(strategy: LiquidityGrabStrategy, 
            data: pd.DataFrame,
            initial_capital: float = 10000.0) -> BacktestResult:
    """
    Run a simple backtest with given strategy and data.
    
    Args:
        strategy: Strategy instance
        data: Historical data
        initial_capital: Starting capital
    
    Returns:
        BacktestResult object
    """
    backtester = Backtester(strategy, initial_capital)
    return backtester.run(data)


logger.info("Optimizer module loaded successfully")