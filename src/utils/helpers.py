"""
Utility helper functions for the trading signal system.

Provides common mathematical operations, data transformations,
retry mechanisms, rate limiting, and financial calculations.
"""

import asyncio
import time
import math
import statistics
from datetime import datetime, timezone, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, AsyncGenerator
from collections import deque, defaultdict
import random

import numpy as np
import pandas as pd
from scipy import stats
from rich.progress import Progress, TaskID
from rich.table import Table
from rich.console import Console

T = TypeVar('T')


class ExponentialBackoffRetry:
    """Retry decorator with exponential backoff and jitter."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        exception_types: tuple = (Exception,)
    ):
        """Initialize retry configuration."""
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.exception_types = exception_types

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Apply retry decorator to function."""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(self.max_attempts):
                try:
                    return func(*args, **kwargs)
                except self.exception_types as e:
                    last_exception = e

                    if attempt == self.max_attempts - 1:
                        # Last attempt, re-raise the exception
                        raise e

                    # Calculate delay with exponential backoff and jitter
                    delay = self._calculate_delay(attempt)
                    time.sleep(delay)

            # This should never be reached
            raise last_exception

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(self.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except self.exception_types as e:
                    last_exception = e

                    if attempt == self.max_attempts - 1:
                        # Last attempt, re-raise the exception
                        raise e

                    # Calculate delay with exponential backoff and jitter
                    delay = self._calculate_delay(attempt)
                    await asyncio.sleep(delay)

            # This should never be reached
            raise last_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt."""
        # Exponential backoff
        delay = self.base_delay * (self.exponential_base ** attempt)

        # Cap the delay
        delay = min(delay, self.max_delay)

        # Add jitter to avoid thundering herd
        if self.jitter:
            # Add up to 25% random jitter
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)

        # Ensure non-negative delay
        return max(0, delay)


def retry_with_exponential_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exception_types: tuple = (Exception,)
):
    """Decorator factory for retry with exponential backoff."""
    return ExponentialBackoffRetry(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        exception_types=exception_types
    )


class TokenBucketRateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, capacity: int, refill_rate: float):
        """Initialize rate limiter.

        Args:
            capacity: Maximum number of tokens in bucket
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()

    def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Acquire tokens from bucket.

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait for tokens

        Returns:
            True if tokens acquired, False if timeout
        """
        start_time = time.time() if timeout else None

        while True:
            # Refill tokens based on elapsed time
            current_time = time.time()
            time_elapsed = current_time - self.last_refill
            tokens_to_add = time_elapsed * self.refill_rate

            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = current_time

            if self.tokens >= tokens:
                # Acquire tokens
                self.tokens -= tokens
                return True

            # Check timeout
            if timeout is not None:
                if time.time() - start_time >= timeout:
                    return False

            # Wait before next attempt
            time.sleep(0.01)  # 10ms

    def time_until_available(self, tokens: int = 1) -> float:
        """Calculate time until tokens are available."""
        if self.tokens >= tokens:
            return 0.0

        tokens_needed = tokens - self.tokens
        time_needed = tokens_needed / self.refill_rate
        return max(0, time_needed)

    def reset(self) -> None:
        """Reset the rate limiter to full capacity."""
        self.tokens = self.capacity
        self.last_refill = time.time()


def rate_limiter(calls: int, period_seconds: float):
    """Decorator factory for rate limiting function calls."""
    limiter = TokenBucketRateLimiter(calls, calls / period_seconds)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            limiter.acquire()
            return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            await asyncio.sleep(limiter.time_until_available())
            limiter.acquire()
            return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper

    return decorator


class CircularBuffer:
    """Thread-safe circular buffer for real-time data."""

    def __init__(self, max_size: int):
        """Initialize circular buffer.

        Args:
            max_size: Maximum number of items in buffer
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self._lock = asyncio.Lock() if asyncio.get_event_loop() else threading.Lock()

    async def async_append(self, item: T) -> None:
        """Append item to buffer (async version)."""
        async with self._lock:
            self.buffer.append(item)

    def append(self, item: T) -> None:
        """Append item to buffer (sync version)."""
        self.buffer.append(item)

    def extend(self, items: List[T]) -> None:
        """Extend buffer with multiple items."""
        self.buffer.extend(items)

    def get_latest(self, n: int) -> List[T]:
        """Get latest n items."""
        return list(self.buffer)[-n:]

    def get_latest_sync(self, n: int) -> List[T]:
        """Get latest n items (sync version)."""
        return list(self.buffer)[-n:]

    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self.buffer) == self.max_size

    def clear(self) -> None:
        """Clear buffer."""
        self.buffer.clear()

    def to_list(self) -> List[T]:
        """Convert buffer to list."""
        return list(self.buffer)


# Financial calculation functions
def calculate_returns(prices: np.ndarray, periods: int = 1) -> np.ndarray:
    """Calculate returns over specified periods.

    Args:
        prices: Array of prices
        periods: Number of periods for return calculation

    Returns:
        Array of returns
    """
    if len(prices) <= periods:
        return np.array([])

    return (prices[periods:] - prices[:-periods]) / prices[:-periods]


def calculate_log_returns(prices: np.ndarray, periods: int = 1) -> np.ndarray:
    """Calculate log returns over specified periods.

    Args:
        prices: Array of prices
        periods: Number of periods for log return calculation

    Returns:
        Array of log returns
    """
    if len(prices) <= periods:
        return np.array([])

    return np.log(prices[periods:] / prices[:-periods])


def calculate_volatility(returns: np.ndarray, annualize: bool = True) -> float:
    """Calculate volatility of returns.

    Args:
        returns: Array of returns
        annualize: Whether to annualize the volatility

    Returns:
        Volatility measure
    """
    if len(returns) == 0:
        return 0.0

    volatility = np.std(returns, ddof=1)

    if annualize:
        # Assuming daily returns, annualize by sqrt(252)
        volatility *= np.sqrt(252)

    return volatility


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualize: bool = True
) -> float:
    """Calculate Sharpe ratio.

    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        annualize: Whether to annualize

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate
    mean_excess_return = np.mean(excess_returns)
    volatility = calculate_volatility(excess_returns, annualize=False)

    if volatility == 0:
        return 0.0

    sharpe = mean_excess_return / volatility

    if annualize:
        # Annualize assuming daily returns
        sharpe *= np.sqrt(252)

    return sharpe


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualize: bool = True
) -> float:
    """Calculate Sortino ratio (downside deviation).

    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        annualize: Whether to annualize

    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate
    mean_excess_return = np.mean(excess_returns)

    # Calculate downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return float('inf') if mean_excess_return > 0 else 0.0

    downside_deviation = np.std(downside_returns, ddof=1)

    if downside_deviation == 0:
        return 0.0

    sortino = mean_excess_return / downside_deviation

    if annualize:
        sortino *= np.sqrt(252)

    return sortino


def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    """Calculate maximum drawdown and its duration.

    Args:
        equity_curve: Array of cumulative returns or portfolio values

    Returns:
        Tuple of (max_drawdown_pct, drawdown_start_idx, drawdown_end_idx)
    """
    if len(equity_curve) == 0:
        return 0.0, 0, 0

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown_idx = np.argmin(drawdown)

    # Find the peak before the max drawdown
    peak_idx = np.argmax(equity_curve[:max_drawdown_idx + 1])

    return drawdown[max_drawdown_idx], peak_idx, max_drawdown_idx


def calculate_calmar_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Calculate Calmar ratio (annual return / max drawdown).

    Args:
        returns: Array of returns
        periods_per_year: Number of periods per year

    Returns:
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0

    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + returns)
    total_return = cumulative_returns[-1] - 1

    # Annualize return
    annual_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1

    # Calculate max drawdown
    max_drawdown, _, _ = calculate_max_drawdown(cumulative_returns)
    max_drawdown = abs(max_drawdown)

    if max_drawdown == 0:
        return float('inf') if annual_return > 0 else 0.0

    return annual_return / max_drawdown


def calculate_information_ratio(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray
) -> float:
    """Calculate information ratio.

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Information ratio
    """
    if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) == 0:
        return 0.0

    excess_returns = portfolio_returns - benchmark_returns
    mean_excess = np.mean(excess_returns)
    tracking_error = np.std(excess_returns, ddof=1)

    return mean_excess / tracking_error if tracking_error != 0 else 0.0


def calculate_beta(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray
) -> float:
    """Calculate beta coefficient.

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Beta coefficient
    """
    if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
        return 0.0

    covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns, ddof=1)

    return covariance / benchmark_variance if benchmark_variance != 0 else 0.0


def calculate_alpha(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    risk_free_rate: float = 0.0
) -> float:
    """Calculate Jensen's alpha.

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns
        risk_free_rate: Risk-free rate

    Returns:
        Alpha measure
    """
    if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) == 0:
        return 0.0

    beta = calculate_beta(portfolio_returns, benchmark_returns)
    portfolio_mean = np.mean(portfolio_returns)
    benchmark_mean = np.mean(benchmark_returns)

    return portfolio_mean - (risk_free_rate + beta * (benchmark_mean - risk_free_rate))


def calculate_value_at_risk(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    method: str = "historical"
) -> float:
    """Calculate Value at Risk (VaR).

    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        method: Method to use ("historical", "parametric", "monte_carlo")

    Returns:
        Value at Risk (negative number for losses)
    """
    if len(returns) == 0:
        return 0.0

    if method == "historical":
        return np.percentile(returns, (1 - confidence_level) * 100)

    elif method == "parametric":
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        z_score = stats.norm.ppf(1 - confidence_level)
        return mean + z_score * std

    elif method == "monte_carlo":
        # Simple Monte Carlo simulation
        n_simulations = 10000
        simulated_returns = np.random.choice(returns, size=n_simulations, replace=True)
        return np.percentile(simulated_returns, (1 - confidence_level) * 100)

    else:
        raise ValueError(f"Unknown VaR method: {method}")


def calculate_conditional_value_at_risk(
    returns: np.ndarray,
    confidence_level: float = 0.95
) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall).

    Args:
        returns: Array of returns
        confidence_level: Confidence level

    Returns:
        Conditional Value at Risk (negative number for losses)
    """
    if len(returns) == 0:
        return 0.0

    var = calculate_value_at_risk(returns, confidence_level, "historical")
    tail_returns = returns[returns <= var]

    return np.mean(tail_returns) if len(tail_returns) > 0 else var


def calculate_risk_adjusted_metrics(returns: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive risk-adjusted performance metrics.

    Args:
        returns: Array of returns

    Returns:
        Dictionary of risk-adjusted metrics
    """
    if len(returns) == 0:
        return {
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown': 0.0,
            'var_95': 0.0,
            'cvar_95': 0.0,
            'volatility': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0
        }

    metrics = {
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'sortino_ratio': calculate_sortino_ratio(returns),
        'calmar_ratio': calculate_calmar_ratio(returns),
        'volatility': calculate_volatility(returns),
        'var_95': calculate_value_at_risk(returns, 0.95),
        'cvar_95': calculate_conditional_value_at_risk(returns, 0.95),
    }

    # Add drawdown
    cumulative_returns = np.cumprod(1 + returns)
    max_dd, _, _ = calculate_max_drawdown(cumulative_returns)
    metrics['max_drawdown'] = abs(max_dd)

    # Add higher moments
    metrics['skewness'] = stats.skew(returns)
    metrics['kurtosis'] = stats.kurtosis(returns)

    return metrics


def normalize_timestamp(timestamp: Union[str, datetime, int, float]) -> datetime:
    """Normalize timestamp to datetime object.

    Args:
        timestamp: Timestamp in various formats

    Returns:
        Normalized datetime object
    """
    if isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)

    elif isinstance(timestamp, str):
        try:
            # Try parsing as ISO format
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.astimezone(timezone.utc)
        except ValueError:
            # Try common formats
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%f',
            ]
            for fmt in formats:
                try:
                    dt = datetime.strptime(timestamp, fmt)
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

            raise ValueError(f"Unable to parse timestamp: {timestamp}")

    elif isinstance(timestamp, (int, float)):
        # Assume milliseconds since epoch
        return datetime.fromtimestamp(timestamp / 1000, timezone.utc)

    else:
        raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")


def format_trading_pair(pair: str) -> str:
    """Format trading pair to standard format."""
    if not pair:
        return ""

    # Remove whitespace and convert to uppercase
    pair = pair.strip().upper()

    # Ensure consistent format (BASE/QUOTE)
    if '/' not in pair:
        # Try to guess the quote currency
        common_quotes = ['USDT', 'USD', 'BTC', 'ETH', 'BNB', 'USDC']
        for quote in common_quotes:
            if pair.endswith(quote):
                base = pair[:-len(quote)]
                return f"{base}/{quote}"

        # If no common quote found, assume last 4 characters are quote
        if len(pair) >= 6:
            base = pair[:-4]
            quote = pair[-4:]
            return f"{base}/{quote}"

        return pair

    return pair


def parse_timeframe_to_seconds(timeframe: str) -> int:
    """Parse timeframe string to seconds.

    Args:
        timeframe: Timeframe string (e.g., "1m", "5m", "1h", "4h", "1d")

    Returns:
        Number of seconds in the timeframe
    """
    timeframe = timeframe.strip().lower()

    # Map of timeframe units to seconds
    unit_mapping = {
        's': 1,
        'm': 60,
        'h': 3600,
        'd': 86400,
        'w': 604800,
        'M': 2592000,  # 30 days
    }

    # Extract number and unit
    import re
    match = re.match(r'^(\d+)([smhdwM])$', timeframe)
    if not match:
        raise ValueError(f"Invalid timeframe format: {timeframe}")

    number, unit = match.groups()
    seconds = int(number) * unit_mapping[unit]

    return seconds


def chunk_dataframe(df: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
    """Split DataFrame into chunks for processing.

    Args:
        df: DataFrame to chunk
        chunk_size: Size of each chunk

    Returns:
        List of DataFrame chunks
    """
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size].copy()
        chunks.append(chunk)
    return chunks


def display_progress_with_rich(
    iterable,
    description: str = "Processing",
    total: Optional[int] = None
) -> Any:
    """Display progress using Rich library."""
    console = Console()
    with Progress(console=console) as progress:
        task = progress.add_task(description, total=total)
        for item in iterable:
            yield item
            progress.advance(task)


def create_summary_table(data: Dict[str, Any], title: str = "Summary") -> Table:
    """Create a formatted Rich table for data summary."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", style="green", width=20)

    for key, value in data.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        elif isinstance(value, int):
            table.add_row(key, str(value))
        else:
            table.add_row(key, str(value))

    return table


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers with default fallback."""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def is_outlier(value: float, data: List[float], threshold: float = 3.0) -> bool:
    """Check if value is an outlier using z-score."""
    if len(data) < 2:
        return False

    mean = statistics.mean(data)
    std = statistics.stdev(data)
    if std == 0:
        return False

    z_score = abs((value - mean) / std)
    return z_score > threshold


def remove_outliers(data: List[float], threshold: float = 3.0) -> List[float]:
    """Remove outliers from data using z-score."""
    if len(data) < 3:
        return data.copy()

    mean = statistics.mean(data)
    std = statistics.stdev(data)
    if std == 0:
        return data.copy()

    return [x for x in data if abs((x - mean) / std) <= threshold]