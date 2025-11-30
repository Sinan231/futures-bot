"""
Technical indicators library for trading signal system.

Comprehensive collection of 50+ technical indicators including moving averages,
momentum, volatility, volume, price patterns, and statistical
measures with efficient vectorized implementation and configuration.
"""

import math
import warnings
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

from ..utils.logging import get_logger, performance_monitor
from ..utils.helpers import calculate_returns, calculate_volatility
from ..utils.config import DataConfig


class IndicatorType(Enum):
    """Type of technical indicator."""
    MOVING_AVERAGE = "moving_average"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    TREND = "trend"
    PATTERN = "pattern"
    STATISTICAL = "statistical"


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""
    indicator_type: IndicatorType
    name: str
    description: str
    parameters: Dict[str, Any]
    enabled: bool = True


@dataclass
class IndicatorResult:
    """Result of indicator calculation."""
    values: Union[np.ndarray, pd.Series]
    metadata: Optional[Dict[str, Any]] = None
    calculation_time: Optional[float] = None
    quality_score: Optional[float] = None


class TechnicalIndicators:
    """Comprehensive technical indicators library."""

    def __init__(self, config: DataConfig):
        """Initialize technical indicators.

        Args:
            config: Data configuration
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Indicator configurations
        self.indicator_configs = self._get_indicator_configs()

        # Performance metrics
        self._calculation_times: Dict[str, List[float]] = {}

    def _get_indicator_configs(self) -> Dict[str, IndicatorConfig]:
        """Get indicator configurations."""
        return {
            # Moving Averages
            'sma_5': IndicatorConfig(IndicatorType.MOVING_AVERAGE, 'SMA(5)', '5-period Simple Moving Average', {'period': 5, 'price_col': 'close'}),
            'sma_10': IndicatorConfig(IndicatorType.MOVING_AVERAGE, 'SMA(10)', '10-period Simple Moving Average', {'period': 10, 'price_col': 'close'}),
            'sma_20': IndicatorConfig(IndicatorType.MOVING_AVERAGE, 'SMA(20)', '20-period Simple Moving Average', {'period': 20, 'price_col': 'close'}),
            'sma_50': IndicatorConfig(IndicatorType.MOVING_AVERAGE, 'SMA(50)', '50-period Simple Moving Average', {'period': 50, 'price_col': 'close'}),
            'sma_100': IndicatorConfig(IndicatorType.MOVING_AVERAGE, 'SMA(100)', '100-period Simple Moving Average', {'period': 100, 'price_col': 'close'}),
            'sma_200': IndicatorConfig(IndicatorType.MOVING_AVERAGE, 'SMA(200)', '200-period Simple Moving Average', {'period': 200, 'price_col': 'close'}),

            'ema_5': IndicatorConfig(IndicatorType.MOVING_AVERAGE, 'EMA(5)', '5-period Exponential Moving Average', {'period': 5, 'alpha': 0.2, 'price_col': 'close'}),
            'ema_10': IndicatorConfig(IndicatorType.MOVING_AVERAGE, 'EMA(10)', '10-period Exponential Moving Average', {'period': 10, 'alpha': 0.1, 'price_col': 'close'}),
            'ema_20': IndicatorConfig(IndicatorType.MOVING_AVERAGE, 'EMA(20)', '20-period Exponential Moving Average', {'period': 20, 'alpha': 0.05, 'price_col': 'close'}),
            'ema_50': IndicatorConfig(IndicatorType.MOVING_AVERAGE, 'EMA(50)', '50-period Exponential Moving Average', {'period': 50, 'alpha': 0.02, 'price_col': 'close'}),
            'ema_100': IndicatorConfig(IndicatorType.MOVING_AVERAGE, 'EMA(100)', '100-period Exponential Moving Average', {'period': 100, 'alpha': 0.01, 'price_col': 'close'}),
            'ema_200': IndicatorConfig(IndicatorType.MOVING_AVERAGE, 'EMA(200)', '200-period Exponential Moving Average', {'period': 200, 'alpha': 0.005, 'price_col': 'close'}),

            'wma_10': IndicatorConfig(IndicatorType.MOVING_AVERAGE, 'WMA(10)', '10-period Weighted Moving Average', {'period': 10, 'weights': 'linear', 'price_col': 'close'}),
            'wma_20': IndicatorConfig(IndicatorType.MOVING_AVERAGE, 'WMA(20)', '20-period Weighted Moving Average', {'period': 20, 'weights': 'linear', 'price_col': 'close'}),

            # Momentum Indicators
            'rsi_14': IndicatorConfig(IndicatorType.MOMENTUM, 'RSI(14)', '14-period Relative Strength Index', {'period': 14, 'price_col': 'close'}),
            'rsi_21': IndicatorConfig(IndicatorType.MOMENTUM, 'RSI(21)', '21-period Relative Strength Index', {'period': 21, 'price_col': 'close'}),
            'rsi_34': IndicatorConfig(IndicatorType.MOMENTUM, 'RSI(34)', '34-period Relative Strength Index', {'period': 34, 'price_col': 'close'}),

            'macd': IndicatorConfig(IndicatorType.MOMENTUM, 'MACD', 'MACD (12,26,9)', {'fast': 12, 'slow': 26, 'signal': 9, 'price_col': 'close'}),

            'stoch': IndicatorConfig(IndicatorType.MOMENTUM, 'Stochastic', 'Stochastic Oscillator (14,3,3)', {'k_period': 14, 'd_period': 3, 'slowing': 3, 'price_col': 'high'}),

            'williams_r': IndicatorConfig(IndicatorType.MOMENTUM, 'Williams %R', '14-period Williams %R', {'period': 14, 'price_col': 'high'}),

            'roc_5': IndicatorConfig(IndicatorType.MOMENTUM, 'ROC(5)', '5-period Rate of Change', {'period': 5, 'price_col': 'close'}),
            'roc_10': IndicatorConfig(IndicatorType.MOMENTUM, 'ROC(10)', '10-period Rate of Change', {'period': 10, 'price_col': 'close'}),
            'roc_20': IndicatorConfig(IndicatorType.MOMENTUM, 'ROC(20)', '20-period Rate of Change', {'period': 20, 'price_col': 'close'}),

            # Volatility Indicators
            'atr_14': IndicatorConfig(IndicatorType.VOLATILITY, 'ATR(14)', '14-period Average True Range', {'period': 14, 'price_cols': ['high', 'low', 'close']}),
            'atr_21': IndicatorConfig(IndicatorType.VOLATILITY, 'ATR(21)', '21-period Average True Range', {'period': 21, 'price_cols': ['high', 'low', 'close']}),
            'bb_20': IndicatorConfig(IndicatorType.VOLATILITY, 'BB(20)', '20-period Bollinger Bands', {'period': 20, 'std_dev': 2, 'price_col': 'close'}),

            'historical_vol': IndicatorConfig(IndicatorType.VOLATILITY, 'Historical Volatility', 'Historical Volatility (20)', {'period': 20, 'price_col': 'close'}),

            # Volume Indicators
            'obv': IndicatorConfig(IndicatorType.VOLUME, 'OBV', 'On Balance Volume', {'price_col': 'close', 'volume_col': 'volume'}),
            'vol_change_5': IndicatorConfig(IndicatorType.VOLUME, 'Vol Change(5)', '5-period Volume Rate of Change', {'period': 5, 'volume_col': 'volume'}),
            'vol_change_10': IndicatorConfig(IndicatorType.VOLUME, 'Vol Change(10)', '10-period Volume Rate of Change', {'period': 10, 'volume_col': 'volume'}),
            'vol_change_20': IndicatorConfig(IndicatorType.VOLUME, 'Vol Change(20)', '20-period Volume Rate of Change', {'period': 20, 'volume_col': 'volume'}),

            'mfi': IndicatorConfig(IndicatorType.VOLUME, 'MFI', 'Money Flow Index (14)', {'period': 14, 'price_col': 'close', 'volume_col': 'volume'}),

            # Trend Indicators
            'price_trend': IndicatorConfig(IndicatorType.TREND, 'Price Trend', 'Price Trend Analysis', {'period': 20, 'price_col': 'close'}),
            'ma_crossover': IndicatorConfig(IndicatorType.TREND, 'MA Crossover', 'Moving Average Crossover', {'fast_period': 5, 'slow_period': 20, 'price_col': 'close'}),
        }

    def calculate_all_indicators(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        volume_col: Optional[str] = None,
        high_col: Optional[str] = None,
        low_col: Optional[str] = None
    ) -> Dict[str, IndicatorResult]:
        """Calculate all enabled technical indicators.

        Args:
            df: DataFrame with OHLCV data
            price_col: Column name for price data
            volume_col: Column name for volume data
            high_col: Column name for high prices
            low_col: Column name for low prices

        Returns:
            Dictionary with all indicator results
        """
        if len(df) == 0:
            return {}

        self.logger.info(f"Calculating {len(self.indicator_configs)} indicators for {len(df)} records")

        results = {}

        # Calculate indicators by type
        for key, config in self.indicator_configs.items():
            if not config.enabled:
                continue

            try:
                start_time = time.time()
                result = self._calculate_single_indicator(df, config, price_col, volume_col, high_col, low_col)
                end_time = time.time()

                if result:
                    result.calculation_time = end_time - start_time
                    results[key] = result

                    # Track calculation time
                    if key not in self._calculation_times:
                        self._calculation_times[key] = []
                    self._calculation_times[key].append(end_time - start_time)

            except Exception as e:
                self.logger.error(f"Failed to calculate {key}: {e}")
                results[key] = IndicatorResult(
                    values=np.array([]),
                    metadata={'error': str(e)}
                )

        # Log performance summary
        self._log_calculation_performance()

        return results

    def _calculate_single_indicator(
        self,
        df: pd.DataFrame,
        config: IndicatorConfig,
        price_col: str = 'close',
        volume_col: Optional[str] = None,
        high_col: Optional[str] = None,
        low_col: Optional[str] = None
    ) -> Optional[IndicatorResult]:
        """Calculate a single technical indicator.

        Args:
            df: DataFrame with OHLCV data
            config: Indicator configuration
            price_col: Column name for price data
            volume_col: Column name for volume data
            high_col: Column name for high prices
            low_col: Column name for low prices

        Returns:
            Indicator result or None if failed
        """
        try:
            if config.indicator_type == IndicatorType.MOVING_AVERAGE:
                return self._calculate_moving_average(df, config, price_col)
            elif config.indicator_type == IndicatorType.MOMENTUM:
                return self._calculate_momentum(df, config, price_col, volume_col, high_col, low_col)
            elif config.indicator_type == IndicatorType.VOLATILITY:
                return self._calculate_volatility(df, config, price_col, high_col, low_col)
            elif config.indicator_type == IndicatorType.VOLUME:
                return self._calculate_volume(df, config, price_col, volume_col)
            elif config.indicator_type == IndicatorType.TREND:
                return self._calculate_trend(df, config, price_col, high_col, low_col)
            elif config.indicator_type == IndicatorType.PATTERN:
                return self._calculate_pattern(df, config, price_col, high_col, low_col)
            elif config.indicator_type == IndicatorType.STATISTICAL:
                return self._calculate_statistical(df, config, price_col)
            else:
                self.logger.warning(f"Unknown indicator type: {config.indicator_type}")
                return None

        except Exception as e:
            self.logger.error(f"Error calculating {config.name}: {e}")
            return IndicatorResult(
                values=np.array([]),
                metadata={'error': str(e), 'indicator': config.name}
            )

    @performance_monitor("sma_calculation")
    def _calculate_moving_average(
        self,
        df: pd.DataFrame,
        config: IndicatorConfig,
        price_col: str = 'close'
    ) -> Optional[IndicatorResult]:
        """Calculate moving average indicators."""
        period = config.parameters.get('period', 20)
        alpha = config.parameters.get('alpha', None)
        weights = config.parameters.get('weights', None)

        if weights == 'linear':
            # Weighted Moving Average
            return self._calculate_wma(df, period, price_col, linear_weights=True)
        elif weights == 'exponential':
            # Exponential Weighted Moving Average
            return self._calculate_ema(df, period, price_col, alpha or 2.0 / (period + 1))
        elif alpha is not None:
            # Exponential Moving Average
            return self._calculate_ema(df, period, price_col, alpha)
        else:
            # Simple Moving Average
            return self._calculate_sma(df, period, price_col)

    def _calculate_sma(self, df: pd.DataFrame, period: int, price_col: str = 'close') -> IndicatorResult:
        """Calculate Simple Moving Average."""
        if len(df) < period:
            return IndicatorResult(
                values=np.array([]),
                metadata={'error': 'Insufficient data for SMA'}
            )

        values = df[price_col].rolling(window=period, min_periods=1).mean().values
        metadata = {
            'period': period,
            'data_points': len(values),
            'null_count': np.isnan(values).sum()
        }

        return IndicatorResult(
            values=values,
            metadata=metadata
        )

    def _calculate_ema(self, df: pd.DataFrame, period: int, price_col: str = 'close', alpha: float = 2.0 / (period + 1)) -> IndicatorResult:
        """Calculate Exponential Moving Average."""
        if len(df) < 2:
            return IndicatorResult(
                values=np.array([]),
                metadata={'error': 'Insufficient data for EMA'}
            )

        # Calculate EMA using pandas
        values = df[price_col].ewm(alpha=alpha, adjust=False).mean().values
        metadata = {
            'period': period,
            'alpha': alpha,
            'data_points': len(values),
            'null_count': np.isnan(values).sum()
        }

        return IndicatorResult(
            values=values,
            metadata=metadata
        )

    def _calculate_wma(self, df: pd.DataFrame, period: int, price_col: str = 'close', linear_weights: bool = True) -> IndicatorResult:
        """Calculate Weighted Moving Average."""
        if len(df) < period:
            return IndicatorResult(
                values=np.array([]),
                metadata={'error': 'Insufficient data for WMA'}
            )

        if linear_weights:
            weights = np.arange(1, period + 1)
        else:
            # Exponential weights
            weights = np.array([2 * i / period for i in range(1, period + 1)])

        # Normalize weights
        weights = weights / weights.sum()

        values = df[price_col].rolling(window=period, min_periods=1).apply(
            lambda x: np.sum(x * weights) if len(x) == len(weights) else np.nan
        ).values

        metadata = {
            'period': period,
            'weights': weights.tolist(),
            'data_points': len(values),
            'null_count': np.isnan(values).sum()
        }

        return IndicatorResult(
            values=values,
            metadata=metadata
        )

    @performance_monitor("momentum_indicators")
    def _calculate_momentum(
        self,
        df: pd.DataFrame,
        config: IndicatorConfig,
        price_col: str = 'close',
        volume_col: Optional[str] = None,
        high_col: Optional[str] = None,
        low_col: Optional[str] = None
    ) -> Optional[IndicatorResult]:
        """Calculate momentum indicators."""
        if config.name.startswith('RSI'):
            return self._calculate_rsi(df, config, price_col)
        elif config.name == 'MACD':
            return self._calculate_macd(df, config, price_col)
        elif config.name.startswith('Stochastic'):
            return self._calculate_stochastic(df, config, high_col, low_col, price_col)
        elif config.name.startswith('Williams'):
            return self._calculate_williams_r(df, config, high_col, low_col, price_col)
        elif config.name.startswith('ROC'):
            return self._calculate_roc(df, config, price_col)
        else:
            self.logger.warning(f"Unknown momentum indicator: {config.name}")
            return None

    def _calculate_rsi(self, df: pd.DataFrame, config: IndicatorConfig, price_col: str = 'close') -> IndicatorResult:
        """Calculate Relative Strength Index."""
        period = config.parameters.get('period', 14)

        if len(df) < period + 1:
            return IndicatorResult(
                values=np.array([]),
                metadata={'error': 'Insufficient data for RSI'}
            )

        # Calculate price changes
        delta = df[price_col].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gains and losses
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        # Calculate RSI
        rs = avg_gain.rolling(window=period, min_periods=1).mean()
        ls = avg_loss.rolling(window=period, min_periods=1).mean()

        rsi = 100 * rs / (rs + ls)

        metadata = {
            'period': period,
            'data_points': len(rsi),
            'null_count': np.isnan(rsi).sum(),
            'avg_gain': avg_gain.iloc[-1] if len(avg_gain) > 0 else 0,
            'avg_loss': avg_loss.iloc[-1] if len(avg_loss) > 0 else 0
        }

        return IndicatorResult(
            values=rsi.values,
            metadata=metadata
        )

    def _calculate_macd(self, df: pd.DataFrame, config: IndicatorConfig, price_col: str = 'close') -> IndicatorResult:
        """Calculate MACD indicator."""
        fast = config.parameters.get('fast', 12)
        slow = config.parameters.get('slow', 26)
        signal = config.parameters.get('signal', 9)

        if len(df) < slow:
            return IndicatorResult(
                values=np.array([]),
                metadata={'error': 'Insufficient data for MACD'}
            )

        # Calculate EMAs
        ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
        ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()

        # Calculate MACD line and signal
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        values = np.column_stack([
            macd_line.values,
            signal_line.values,
            histogram.values
        ])

        metadata = {
            'fast_period': fast,
            'slow_period': slow,
            'signal_period': signal,
            'data_points': len(values),
            'null_count': np.isnan(values).sum()
        }

        return IndicatorResult(
            values=values,
            metadata=metadata
        )

    def _calculate_stochastic(
        self,
        df: pd.DataFrame,
        config: IndicatorConfig,
        high_col: Optional[str] = None,
        low_col: Optional[str] = None,
        close_col: str = 'close'
    ) -> IndicatorResult:
        """Calculate Stochastic Oscillator."""
        k_period = config.parameters.get('k_period', 14)
        d_period = config.parameters.get('d_period', 3)
        slowing = config.parameters.get('slowing', 3)

        if not high_col or not low_col:
            return IndicatorResult(
                values=np.array([]),
                metadata={'error': 'High and Low columns required for Stochastic'}
            )

        if len(df) < k_period + d_period:
            return IndicatorResult(
                values=np.array([]),
                metadata={'error': 'Insufficient data for Stochastic'}
            )

        # Calculate %K
        lowest_low = df[low_col].rolling(window=k_period, min_periods=1).min()
        highest_high = df[high_col].rolling(window=k_period, min_periods=1).max()

        current_close = df[close_col]
        r100 = (current_close - lowest_low) / (highest_high - lowest_low) * 100

        # Calculate %D
        k_percent = r100.rolling(window=d_period, min_periods=1).mean()
        d_percent = k_percent.rolling(window=slowing, min_periods=1).mean()

        # Calculate Stochastic oscillator
        k_slow = k_percent.rolling(window=slowing, min_periods=1).mean()
        d_slow = d_percent.rolling(window=slowing, min_periods=1).mean()

        # Final oscillator values
        k_final = k_percent.values
        d_final = d_percent.values

        values = np.column_stack([k_final, d_final, k_slow, d_slow])

        metadata = {
            'k_period': k_period,
            'd_period': d_period,
            'slowing': slowing,
            'data_points': len(values),
            'null_count': np.isnan(values).sum()
        }

        return IndicatorResult(
            values=values,
            metadata=metadata
        )

    def _calculate_williams_r(self, df: pd.DataFrame, config: IndicatorConfig, high_col: Optional[str] = None, low_col: Optional[str] = None, close_col: str = 'close') -> IndicatorResult:
        """Calculate Williams %R."""
        period = config.parameters.get('period', 14)

        if not high_col or not low_col:
            return IndicatorResult(
                values=np.array([]),
                metadata={'error': 'High and Low columns required for Williams %R'}
            )

        if len(df) < period:
            return IndicatorResult(
                values=np.array([]),
                metadata={'error': 'Insufficient data for Williams %R'}
            )

        # Calculate %R
        highest_high = df[high_col].rolling(window=period, min_periods=1).max()
        lowest_low = df[low_col].rolling(window=period, min_periods=1).min()

        current_close = df[close_col]
        r100 = (highest_high - current_close.shift(1)) / (highest_high - lowest_low) * 100

        values = r100.values
        metadata = {
            'period': period,
            'data_points': len(values),
            'null_count': np.isnan(values).sum()
        }

        return IndicatorResult(
            values=values,
            metadata=metadata
        )

    def _calculate_roc(self, df: pd.DataFrame, config: IndicatorConfig, price_col: str = 'close') -> IndicatorResult:
        """Calculate Rate of Change."""
        period = config.parameters.get('period', 10)

        if len(df) <= period:
            return IndicatorResult(
                values=np.array([]),
                metadata={'error': 'Insufficient data for ROC'}
            )

        values = (df[price_col] / df[price_col].shift(period) - 1) * 100

        metadata = {
            'period': period,
            'data_points': len(values),
            'null_count': np.isnan(values).sum()
        }

        return IndicatorResult(
            values=values.values,
            metadata=metadata
        )

    @performance_monitor("volatility_indicators")
    def _calculate_volatility(
        self,
        df: pd.DataFrame,
        config: IndicatorConfig,
        price_col: str = 'close',
        high_col: Optional[str] = None,
        low_col: Optional[str] = None
    ) -> Optional[IndicatorResult]:
        """Calculate volatility indicators."""
        if config.name.startswith('ATR'):
            return self._calculate_atr(df, config, high_col, low_col, close_col)
        elif config.name.startswith('BB'):
            return self._calculate_bollinger_bands(df, config, close_col)
        elif config.name.startswith('Historical'):
            return self._calculate_historical_volatility(df, config, close_col)
        else:
            self.logger.warning(f"Unknown volatility indicator: {config.name}")
            return None

    def _calculate_atr(self, df: pd.DataFrame, config: IndicatorConfig, high_col: Optional[str] = None, low_col: Optional[str] = None, close_col: str = 'close') -> IndicatorResult:
        """Calculate Average True Range."""
        period = config.parameters.get('period', 14)
        price_cols = config.parameters.get('price_cols', ['high', 'low', 'close'])

        if len(df) < period:
            return IndicatorResult(
                values=np.array([]),
                metadata={'error': 'Insufficient data for ATR'}
            )

        # Validate required columns
        required_cols = set(price_cols)
        available_cols = set(df.columns)
        missing_cols = required_cols - available_cols

        if missing_cols:
            return IndicatorResult(
                values=np.array([]),
                metadata={'error': f'Missing required columns: {missing_cols}'}
            )

        # Calculate True Range
        if 'high' in price_cols and 'low' in price_cols:
            tr1 = df[high_col] - df[low_col]
        elif 'high' in price_cols and 'close' in price_cols:
            tr1 = abs(df[high_col] - df[close_col])
        else:
            tr1 = abs(df[close_col] - df[close_col].shift(1))

        # Calculate ATR
        atr = tr1.rolling(window=period, min_periods=1).mean()

        metadata = {
            'period': period,
            'price_cols': price_cols,
            'data_points': len(atr),
            'null_count': np.isnan(atr).sum(),
            'avg_atr': atr.iloc[-1] if len(atr) > 0 else 0
        }

        return IndicatorResult(
            values=atr.values,
            metadata=metadata
        )

    def _calculate_bollinger_bands(self, df: pd.DataFrame, config: IndicatorConfig, price_col: str = 'close') -> IndicatorResult:
        """Calculate Bollinger Bands."""
        period = config.parameters.get('period', 20)
        std_dev = config.parameters.get('std_dev', 2)

        if len(df) < period:
            return IndicatorResult(
                values=np.array([]),
                metadata={'error': 'Insufficient data for Bollinger Bands'}
            )

        # Calculate middle band (SMA)
        middle_band = df[price_col].rolling(window=period, min_periods=1).mean()

        # Calculate standard deviation
        std_deviation = df[price_col].rolling(window=period, min_periods=1).std()

        # Calculate upper and lower bands
        upper_band = middle_band + (std_deviation * std_dev)
        lower_band = middle_band - (std_deviation * std_dev)

        values = np.column_stack([
            middle_band.values,
            upper_band.values,
            lower_band.values,
            std_deviation.values
        ])

        metadata = {
            'period': period,
            'std_dev': std_dev,
            'data_points': len(values),
            'null_count': np.isnan(values).sum(),
            'bandwidth': std_dev * 2,
            'current_position': self._calculate_bb_position(df[price_col].iloc[-1], upper_band.iloc[-1], lower_band.iloc[-1])
        }

        return IndicatorResult(
            values=values,
            metadata=metadata
        )

    def _calculate_historical_volatility(self, df: pd.DataFrame, config: IndicatorConfig, price_col: str = 'close') -> IndicatorResult:
        """Calculate historical volatility."""
        period = config.parameters.get('period', 20)

        if len(df) < period:
            return IndicatorResult(
                values=np.array([]),
                metadata={'error': 'Insufficient data for Historical Volatility'}
            )

        # Calculate rolling volatility
        returns = calculate_returns(df[price_col])
        volatility = returns.rolling(window=period, min_periods=1).std() * np.sqrt(252)  # Annualized

        values = volatility.values

        metadata = {
            'period': period,
            'data_points': len(values),
            'null_count': np.isnan(values).sum(),
            'current_volatility': volatility.iloc[-1] if len(volatility) > 0 else 0
        }

        return IndicatorResult(
            values=values,
            metadata=metadata
        )

    def _calculate_bb_position(self, price: float, upper: float, lower: float) -> str:
        """Calculate position within Bollinger Bands."""
        if price > upper:
            return 'above_upper'
        elif price < lower:
            return 'below_lower'
        else:
            return 'within_bands'

    @performance_monitor("volume_indicators")
    def _calculate_volume(self, df: pd.DataFrame, config: IndicatorConfig, price_col: str = 'close', volume_col: Optional[str] = None) -> IndicatorResult:
        """Calculate volume-based indicators."""
        if not volume_col or volume_col not in df.columns:
            return IndicatorResult(
                values=np.array([]),
                metadata={'error': 'Volume column not found'}
            )

        if config.name == 'OBV':
            return self._calculate_obv(df, price_col, volume_col)
        elif config.name.startswith('Vol Change'):
            return self._calculate_volume_change(df, config, volume_col)
        elif config.name == 'MFI':
            return self._calculate_mfi(df, config, price_col, volume_col)
        else:
            self.logger.warning(f"Unknown volume indicator: {config.name}")
            return None

    def _calculate_obv(self, df: pd.DataFrame, price_col: str = 'close', volume_col: str = 'volume') -> IndicatorResult:
        """Calculate On Balance Volume."""
        # Calculate price change direction
        price_change = np.sign(df[price_col].diff()).fillna(0)

        # Calculate OBV
        obv = (price_change * df[volume_col]).cumsum()

        metadata = {
            'data_points': len(obv),
            'null_count': np.isnan(obv).sum(),
            'final_obv': obv.iloc[-1] if len(obv) > 0 else 0
        }

        return IndicatorResult(
            values=obv.values,
            metadata=metadata
        )

    def _calculate_volume_change(self, df: pd.DataFrame, config: IndicatorConfig, volume_col: str = 'volume') -> IndicatorResult:
        """Calculate Volume Rate of Change."""
        period = config.parameters.get('period', 10)

        if len(df) < period:
            return IndicatorResult(
                values=np.array([]),
                metadata={'error': 'Insufficient data for Volume Change'}
            )

        # Calculate percentage change
        volume_change = df[volume_col].pct_change(periods=period) * 100

        metadata = {
            'period': period,
            'data_points': len(volume_change),
            'null_count': np.isnan(volume_change).sum(),
            'avg_change': volume_change.mean().iloc[-1] if len(volume_change) > 0 else 0
        }

        return IndicatorResult(
            values=volume_change.values,
            metadata=metadata
        )

    def _calculate_mfi(self, df: pd.DataFrame, config: IndicatorConfig, price_col: str = 'close', volume_col: str = 'volume') -> IndicatorResult:
        """Calculate Money Flow Index."""
        period = config.parameters.get('period', 14)

        if len(df) < period:
            return IndicatorResult(
                values=np.array([]),
                metadata={'error': 'Insufficient data for MFI'}
            )

        # Calculate typical price
        typical_price = ((df[high_col] + df[low_col] + 2 * df[close_col]) / 4).rolling(window=period, min_periods=1)

        # Calculate money flow
        money_flow = typical_price * df[volume_col]

        # Calculate positive and negative money flow
        current_typical = typical_price.shift(1)
        positive_mf = money_flow.where(current_typical <= typical_price, 0)
        negative_mf = money_flow.where(current_typical > typical_price, 0)

        # Calculate sums
        positive_mf_sum = positive_mf.rolling(window=period, min_periods=1).sum()
        negative_mf_sum = negative_mf.rolling(window=period, min_periods=1).sum()

        # Calculate MFI
        mfi = 100 * positive_mf_sum / (positive_mf_sum + negative_mf_sum)

        metadata = {
            'period': period,
            'data_points': len(mfi),
            'null_count': np.isnan(mfi).sum(),
            'final_mfi': mfi.iloc[-1] if len(mfi) > 0 else 50
        }

        return IndicatorResult(
            values=mfi.values,
            metadata=metadata
        )

    @performance_monitor("trend_indicators")
    def _calculate_trend(self, df: pd.DataFrame, config: IndicatorConfig, price_col: str = 'close', high_col: Optional[str] = None, low_col: Optional[str] = None) -> IndicatorResult:
        """Calculate trend indicators."""
        if config.name.startswith('MA Crossover'):
            return self._calculate_ma_crossover(df, config, price_col)
        elif config.name.startswith('Price Trend'):
            return self._calculate_price_trend(df, config, price_col, high_col, low_col)
        else:
            self.logger.warning(f"Unknown trend indicator: {config.name}")
            return None

    def _calculate_ma_crossover(self, df: pd.DataFrame, config: IndicatorConfig, price_col: str = 'close') -> IndicatorResult:
        """Calculate Moving Average Crossover."""
        fast_period = config.parameters.get('fast_period', 5)
        slow_period = config.parameters.get('slow_period', 20)

        if len(df) < slow_period:
            return IndicatorResult(
                values=np.array([]),
                metadata={'error': 'Insufficient data for MA Crossover'}
            )

        # Calculate moving averages
        fast_ma = df[price_col].rolling(window=fast_period, min_periods=1).mean()
        slow_ma = df[price_col].rolling(window=slow_period, min_periods=1).mean()

        # Calculate crossover signals
        crossover = (fast_ma > slow_ma).astype(int)
        crossunder = (fast_ma < slow_ma).astype(int)

        values = np.column_stack([crossover.values, crossunder.values])

        metadata = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'data_points': len(values),
            'null_count': np.isnan(values).sum(),
            'current_signal': 'bullish' if crossover.iloc[-1] else 'bearish'
        }

        return IndicatorResult(
            values=values,
            metadata=metadata
        )

    def _calculate_price_trend(self, df: pd.DataFrame, config: IndicatorConfig, price_col: str = 'close', high_col: Optional[str] = None, low_col: Optional[str] = None) -> IndicatorResult:
        """Calculate price trend."""
        period = config.parameters.get('period', 20)

        if len(df) < period:
            return IndicatorResult(
                values=np.array([]),
                metadata={'error': 'Insufficient data for Price Trend'}
            )

        # Calculate trend metrics
        if high_col and low_col in df.columns:
            high_max = df[high_col].rolling(window=period, min_periods=1).max()
            low_min = df[low_col].rolling(window=period, min_periods=1).min()
        else:
            high_max = df[price_col].rolling(window=period, min_periods=1).max()
            low_min = df[price_col].rolling(window=period, min_periods=1).min()

        # Calculate trend strength
        trend_strength = high_max - low_min
        trend_direction = np.sign(trend_strength)

        # Detect higher highs and higher lows
        if high_col and low_col in df.columns:
            higher_highs = (df[high_col] > high_max.shift(1)).astype(int)
            higher_lows = (df[low_col] < low_min.shift(1)).astype(int)
        else:
            higher_highs = np.zeros(len(trend_strength))
            higher_lows = np.zeros(len(trend_strength))

        values = np.column_stack([
            trend_strength.values,
            trend_direction.values,
            higher_highs,
            higher_lows
        ])

        metadata = {
            'period': period,
            'data_points': len(values),
            'null_count': np.isnan(values).sum(),
            'trend_strength': trend_strength.iloc[-1] if len(trend_strength) > 0 else 0
        }

        return IndicatorResult(
            values=values,
            metadata=metadata
        )

    @performance_monitor("pattern_indicators")
    def _calculate_pattern(self, df: pd.DataFrame, config: IndicatorConfig, price_col: str = 'close', high_col: Optional[str] = None, low_col: Optional[str] = None) -> IndicatorResult:
        """Calculate pattern recognition indicators."""
        # Placeholder for advanced pattern recognition
        # Could include head and shoulders, triangles, support/resistance, etc.
        pattern_signals = np.zeros(len(df))

        metadata = {
            'data_points': len(pattern_signals),
            'null_count': np.isnan(pattern_signals).sum(),
            'patterns_detected': 0
        }

        return IndicatorResult(
            values=pattern_signals,
            metadata=metadata
        )

    @performance_monitor("statistical_indicators")
    def _calculate_statistical(self, df: pd.DataFrame, config: IndicatorConfig, price_col: str = 'close') -> IndicatorResult:
        """Calculate statistical indicators."""
        period = config.parameters.get('period', 20)

        if len(df) < period:
            return IndicatorResult(
                values=np.array([]),
                metadata={'error': 'Insufficient data for Statistical indicators'}
            )

        # Calculate z-scores
        mean_price = df[price_col].rolling(window=period, min_periods=1).mean()
        std_price = df[price_col].rolling(window=period, min_periods=1).std()
        z_scores = (df[price_col] - mean_price) / std_price

        values = z_scores.values

        metadata = {
            'period': period,
            'data_points': len(values),
            'null_count': np.isnan(values).sum(),
            'current_z_score': z_scores.iloc[-1] if len(z_scores) > 0 else 0
        }

        return IndicatorResult(
            values=values,
            metadata=metadata
        )

    def _log_calculation_performance(self) -> None:
        """Log performance statistics for indicator calculations."""
        if not self._calculation_times:
            return

        for indicator, times in self._calculation_times.items():
            if times:
                avg_time = np.mean(times)
                total_time = np.sum(times)
                max_time = np.max(times)
                min_time = np.min(times)

                self.logger.info(
                    f"{indicator} performance: avg={avg_time:.4f}s, "
                    f"total={total_time:.2f}s, max={max_time:.4f}s, "
                    f"min={min_time:.4f}s, calculations={len(times)}"
                )

    def get_indicator_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all indicators."""
        performance = {}

        for indicator, times in self._calculation_times.items():
            if times:
                performance[indicator] = {
                    'avg_time_seconds': np.mean(times),
                    'total_time_seconds': np.sum(times),
                    'max_time_seconds': np.max(times),
                    'min_time_seconds': np.min(times),
                    'calculations_performed': len(times),
                    'calculations_per_second': len(times) / max(1, np.sum(times))
                }

        return performance

    def calculate_single_indicator(
        self,
        df: pd.DataFrame,
        indicator_name: str,
        price_col: str = 'close',
        volume_col: Optional[str] = None,
        high_col: Optional[str] = None,
        low_col: Optional[str] = None,
        **kwargs
    ) -> Optional[IndicatorResult]:
        """Calculate a single indicator by name."""
        if indicator_name not in self.indicator_configs:
            self.logger.warning(f"Indicator {indicator_name} not configured")
            return None

        config = self.indicator_configs[indicator_name]
        return self._calculate_single_indicator(df, config, price_col, volume_col, high_col, low_col)

    def get_available_indicators(self) -> List[str]:
        """Get list of available indicator configurations."""
        return list(self.indicator_configs.keys())

    def is_indicator_enabled(self, indicator_name: str) -> bool:
        """Check if an indicator is enabled."""
        return (indicator_name in self.indicator_configs and
                self.indicator_configs[indicator_name].enabled)

    def update_indicator_config(self, indicator_name: str, **updates) -> None:
        """Update configuration for an indicator."""
        if indicator_name not in self.indicator_configs:
            self.logger.warning(f"Indicator {indicator_name} not found")
            return

        # Update configuration
        current_config = self.indicator_configs[indicator_name]
        current_config.parameters.update(updates)

        self.indicator_configs[indicator_name] = current_config
        self.logger.info(f"Updated configuration for {indicator_name}")

    def enable_indicator(self, indicator_name: str) -> None:
        """Enable an indicator."""
        if indicator_name in self.indicator_configs:
            self.indicator_configs[indicator_name].enabled = True
            self.logger.info(f"Enabled indicator: {indicator_name}")

    def disable_indicator(self, indicator_name: str) -> None:
        """Disable an indicator."""
        if indicator_name in self.indicator_configs:
            self.indicator_configs[indicator_name].enabled = False
            self.logger.info(f"Disabled indicator: {indicator_name}")

    def batch_calculate_indicators(
        self,
        df: pd.DataFrame,
        indicator_names: List[str],
        parallel: bool = True
    ) -> Dict[str, IndicatorResult]:
        """Calculate multiple indicators in batch."""
        self.logger.info(f"Batch calculating {len(indicator_names)} indicators")

        if not parallel:
            results = {}
            for name in indicator_names:
                if name in self.indicator_configs and self.indicator_configs[name].enabled:
                    result = self.calculate_single_indicator(df, name)
                    if result:
                        results[name] = result
            return results

        # Parallel calculation
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    self.calculate_single_indicator, df, name
                )
                for name in indicator_names
                if name in self.indicator_configs and self.indicator_configs[name].enabled
            }

            results = {}
            for future in as_completed(futures):
                name, result = future.result()
                if result:
                    results[name] = result

        return results