"""
Main feature engineering pipeline for trading signal system.

Orchestrates data merging, feature creation, technical indicators,
market microstructure analysis, and feature validation with efficient
processing for both historical and real-time data.
"""

import warnings
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import concurrent.futures

import pandas as pd
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc

from ..utils.logging import get_logger, performance_monitor, log_data_quality
from ..utils.helpers import calculate_data_quality_score, parse_timeframe_to_seconds
from ..utils.config import DataConfig
from .indicators import TechnicalIndicators
from .microstructure import OrderBookFeatures, TradeMicrostructureFeatures
from .time_features import TimeBasedFeatures, CyclicalFeatures, SessionFeatures
from .scaling import FeatureScaler


class FeatureType(Enum):
    """Type of features in the pipeline."""
    OHLCV = "ohlcv"
    ORDER_BOOK = "order_book"
    TRADE_MICROSTRUCTURE = "trade_microstructure"
    TECHNICAL_INDICATORS = "technical_indicators"
    TIME_FEATURES = "time_features"
    CYCLICAL_FEATURES = "cyclical_features"
    MARKET_SESSION = "market_session"
    DERIVED_FEATURES = "derived_features"
    RISK_FEATURES = "risk_features"


@dataclass
class FeatureSchema:
    """Definition of a feature in the pipeline."""
    name: str
    feature_type: FeatureType
    description: str
    source_columns: List[str]
    data_type: str = "float64"
    required: bool = True
    aggregation_method: Optional[str] = None
    time_window: Optional[int] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering pipeline."""
    enabled_features: Set[FeatureType]
    timeframes: List[str]
    technical_indicators: Dict[str, Any]
    order_book_levels: int = 10
    microstructure_features: bool = True
    time_features: bool = True
    cyclical_features: bool = True
    session_features: bool = True
    derived_features: bool = True
    risk_features: bool = True
    quality_threshold: float = 0.7
    parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 10000
    memory_limit_gb: float = 4.0


class DataMerger:
    """Handles merging of multiple data sources into unified timeline."""

    def __init__(self, config: DataConfig):
        """Initialize data merger.

        Args:
            config: Data configuration
        """
        self.config = config
        self.logger = get_logger(__name__)

    def merge_datasets(
        self,
        klines_data: Optional[pd.DataFrame] = None,
        trades_data: Optional[pd.DataFrame] = None,
        depth_data: Optional[pd.DataFrame] = None,
        book_ticker_data: Optional[pd.DataFrame] = None,
        mark_price_data: Optional[pd.DataFrame] = None,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        symbol: str = "BTCUSDT",
        timeframe: Optional[str] = None
    ) -> pd.DataFrame:
        """Merge multiple data sources into unified DataFrame.

        Args:
            klines_data: Kline/candlestick data
            trades_data: Trade data
            depth_data: Order book depth data
            book_ticker_data: Book ticker data
            mark_price_data: Mark price data
            funding_rate_data: Funding rate data
            open_interest_data: Open interest data
            symbol: Trading symbol
            timeframe: Timeframe (for klines)

        Returns:
            Merged DataFrame with aligned timestamps
        """
        self.logger.info(f"Merging datasets for {symbol} {timeframe or ''}")

        # Start with primary data (klines if available)
        if klines_data is not None and len(klines_data) > 0:
            merged_df = klines_data.copy()
            self.logger.debug(f"Base dataset: {len(merged_df)} klines")
        elif trades_data is not None and len(trades_data) > 0:
            merged_df = trades_data.copy()
            # Create OHLC from trades if no klines
            if timeframe:
                merged_df = self._create_ohlcv_from_trades(merged_df, timeframe)
                self.logger.debug(f"Created OHLC from {len(trades_data)} trades")
        else:
            self.logger.warning("No primary data source available")
            return pd.DataFrame()

        # Determine primary timestamp column
        timestamp_col = self._find_timestamp_column(merged_df)
        if not timestamp_col:
            self.logger.error("No timestamp column found in primary data")
            return pd.DataFrame()

        # Merge secondary datasets
        datasets_to_merge = [
            ('depth', depth_data),
            ('book_ticker', book_ticker_data),
            ('mark_price', mark_price_data),
            ('funding_rate', funding_rate_data),
            ('open_interest', open_interest_data)
        ]

        for name, dataset in datasets_to_merge:
            if dataset is not None and len(dataset) > 0:
                merged_df = self._merge_single_dataset(
                    merged_df, dataset, name, timestamp_col
                )
                self.logger.debug(f"Merged {name} data: {len(dataset)} records")

        # Validate merge quality
        merge_quality = self._assess_merge_quality(merged_df)
        self.logger.info(f"Merge quality score: {merge_quality:.3f}")

        # Sort by timestamp
        if timestamp_col in merged_df.columns:
            merged_df = merged_df.sort_values(timestamp_col).reset_index(drop=True)

        # Add metadata
        merged_df['symbol'] = symbol
        merged_df['timeframe'] = timeframe
        merged_df['merge_timestamp'] = datetime.now(timezone.utc)

        return merged_df

    def _create_ohlcv_from_trades(
        self,
        trades_df: pd.DataFrame,
        timeframe: str
    ) -> pd.DataFrame:
        """Create OHLCV data from trades for specified timeframe."""
        if len(trades_df) == 0:
            return pd.DataFrame()

        # Ensure timestamp column
        timestamp_col = self._find_timestamp_column(trades_df)
        if not timestamp_col:
            return pd.DataFrame()

        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(trades_df[timestamp_col]):
            trades_df[timestamp_col] = pd.to_datetime(trades_df[timestamp_col], utc=True)

        # Sort by timestamp
        trades_df = trades_df.sort_values(timestamp_col)

        # Group by timeframe
        timeframe_seconds = parse_timeframe_to_seconds(timeframe)
        trades_df['time_group'] = trades_df[timestamp_col].dt.floor(f'{timeframe_seconds}S')

        # Create OHLCV
        ohlcv = trades_df.groupby('time_group').agg({
            'open': ('price', 'first'),
            'high': ('price', 'max'),
            'low': ('price', 'min'),
            'close': ('price', 'last'),
            'volume': ('qty', 'sum'),
            'trade_count': ('price', 'count'),
            'buy_volume': ('qty', lambda x: x[trades_df.loc[x.index, 'is_buyer_maker'] == False].sum()]),
            'sell_volume': ('qty', lambda x: x[trades_df.loc[x.index, 'is_buyer_maker'] == True].sum()]),
            'vwap': ('price', lambda x: (x * trades_df.loc[x.index, 'qty']).sum() / trades_df.loc[x.index, 'qty'].sum()),
            'timestamp': (timestamp_col, 'first')
        }).reset_index(drop=True)

        # Rename timestamp column
        ohlcv = ohlcv.rename(columns={'timestamp': 'open_time'})

        # Calculate close_time (end of period)
        ohlcv['close_time'] = ohlcv['open_time'] + pd.Timedelta(seconds=timeframe_seconds)

        # Add derived features
        ohlcv['price_range'] = ohlcv['high'] - ohlcv['low']
        ohlcv['price_change'] = ohlcv['close'] - ohlcv['open']
        ohlcv['price_change_pct'] = (ohlcv['price_change'] / ohlcv['open']) * 100
        ohlcv['buy_sell_ratio'] = ohlcv['buy_volume'] / (ohlcv['buy_volume'] + ohlcv['sell_volume'])
        ohlcv['avg_trade_size'] = ohlcv['volume'] / ohlcv['trade_count']

        return ohlcv

    def _merge_single_dataset(
        self,
        primary_df: pd.DataFrame,
        secondary_df: pd.DataFrame,
        dataset_name: str,
        timestamp_col: str
    ) -> pd.DataFrame:
        """Merge a single dataset into the primary DataFrame."""
        # Find timestamp column in secondary dataset
        sec_timestamp_col = self._find_timestamp_column(secondary_df)
        if not sec_timestamp_col:
            self.logger.warning(f"No timestamp column in {dataset_name} data")
            return primary_df

        # Convert timestamps if needed
        if not pd.api.types.is_datetime64_any_dtype(primary_df[timestamp_col]):
            primary_df[timestamp_col] = pd.to_datetime(primary_df[timestamp_col], utc=True)
        if not pd.api.types.is_datetime64_any_dtype(secondary_df[sec_timestamp_col]):
            secondary_df[sec_timestamp_col] = pd.to_datetime(secondary_df[sec_timestamp_col], utc=True)

        # Merge with appropriate strategy based on data frequency
        if dataset_name in ['mark_price', 'funding_rate', 'open_interest']:
            # Low frequency data - forward fill
            return self._merge_forward_fill(primary_df, secondary_df, timestamp_col, sec_timestamp_col, dataset_name)
        else:
            # High frequency data - nearest neighbor
            return self._merge_nearest_neighbor(primary_df, secondary_df, timestamp_col, sec_timestamp_col, dataset_name)

    def _merge_forward_fill(
        self,
        primary_df: pd.DataFrame,
        secondary_df: pd.DataFrame,
        primary_timestamp_col: str,
        secondary_timestamp_col: str,
        dataset_name: str
    ) -> pd.DataFrame:
        """Merge using forward fill strategy."""
        # Prepare for merge
        primary_sorted = primary_df.sort_values(primary_timestamp_col)
        secondary_sorted = secondary_df.sort_values(secondary_timestamp_col)

        # Create suffix for column names
        suffix = f"_{dataset_name}"
        secondary_renamed = secondary_sorted.rename(columns={
            col: f"{col}{suffix}" for col in secondary_sorted.columns
            if col != secondary_timestamp_col
        })

        # Merge
        merged = pd.merge_asof(
            primary_sorted,
            secondary_renamed,
            left_on=primary_timestamp_col,
            right_on=secondary_timestamp_col,
            direction='forward',
            tolerance=pd.Timedelta(minutes=1)  # Allow 1 minute tolerance
        )

        return merged

    def _merge_nearest_neighbor(
        self,
        primary_df: pd.DataFrame,
        secondary_df: pd.DataFrame,
        primary_timestamp_col: str,
        secondary_timestamp_col: str,
        dataset_name: str
    ) -> pd.DataFrame:
        """Merge using nearest neighbor strategy."""
        # Prepare for merge
        primary_sorted = primary_df.sort_values(primary_timestamp_col)
        secondary_sorted = secondary_df.sort_values(secondary_timestamp_col)

        # Create suffix for column names
        suffix = f"_{dataset_name}"
        secondary_renamed = secondary_sorted.rename(columns={
            col: f"{col}{suffix}" for col in secondary_sorted.columns
            if col != secondary_timestamp_col
        })

        # Merge with nearest neighbor
        merged = pd.merge_asof(
            primary_sorted,
            secondary_renamed,
            left_on=primary_timestamp_col,
            right_on=secondary_timestamp_col,
            direction='nearest',
            tolerance=pd.Timedelta(seconds=30)  # 30 second tolerance
        )

        return merged

    def _find_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the primary timestamp column in DataFrame."""
        timestamp_candidates = ['timestamp', 'open_time', 'time', 'event_time', 'T', 'E']
        for candidate in timestamp_candidates:
            if candidate in df.columns:
                return candidate
        return None

    def _assess_merge_quality(self, df: pd.DataFrame) -> float:
        """Assess the quality of merged data."""
        if len(df) == 0:
            return 0.0

        quality_factors = {}

        # Timestamp continuity
        timestamp_col = self._find_timestamp_column(df)
        if timestamp_col and timestamp_col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)

            # Check for gaps
            if len(df) > 1:
                df_sorted = df.sort_values(timestamp_col)
                time_diffs = df_sorted[timestamp_col].diff().dropna()
                median_diff = time_diffs.median()

                # Large gaps reduce quality
                large_gaps = time_diffs > median_diff * 3
                gap_ratio = large_gaps.sum() / len(time_diffs)
                quality_factors['timestamp_continuity'] = max(0, 1 - gap_ratio)

        # Data completeness
        missing_ratios = df.isnull().sum() / len(df)
        avg_missing_ratio = missing_ratios.mean()
        quality_factors['completeness'] = max(0, 1 - avg_missing_ratio)

        # Consistency checks
        consistency_score = 1.0

        # OHLC consistency
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            ohlc_consistent = (
                (df['high'] >= df['low']).all() and
                (df['high'] >= df['open']).all() and
                (df['high'] >= df['close']).all() and
                (df['low'] <= df['open']).all() and
                (df['low'] <= df['close']).all() and
                (df[['open', 'high', 'low', 'close']] > 0).all().all()
            )
            consistency_score *= 0.9 if ohlc_consistent else 0.5

        # Price consistency
        price_columns = [col for col in df.columns if 'price' in col.lower()]
        if price_columns:
            for col in price_columns:
                if col in df.columns:
                    price_consistent = (df[col] > 0).all()
                    consistency_score *= 0.9 if price_consistent else 0.6

        # Volume consistency
        volume_columns = [col for col in df.columns if 'volume' in col.lower() or col == 'qty']
        if volume_columns:
            for col in volume_columns:
                if col in df.columns:
                    volume_consistent = (df[col] >= 0).all()
                    consistency_score *= 0.9 if volume_consistent else 0.6

        quality_factors['consistency'] = consistency_score

        # Calculate overall quality score
        weights = {
            'timestamp_continuity': 0.3,
            'completeness': 0.3,
            'consistency': 0.4
        }

        overall_score = sum(
            quality_factors[factor] * weights[factor]
            for factor in weights
            if factor in quality_factors
        )

        return max(0, min(1, overall_score))


class FeatureBuilder:
    """Builds individual features from data."""

    def __init__(self, config: DataConfig):
        """Initialize feature builder.

        Args:
            config: Data configuration
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Initialize feature calculators
        self.technical_indicators = TechnicalIndicators()
        self.time_features = TimeBasedFeatures()
        self.cyclical_features = CyclicalFeatures()
        self.session_features = SessionFeatures()

    def build_base_features(
        self,
        df: pd.DataFrame,
        feature_config: FeatureConfig
    ) -> pd.DataFrame:
        """Build base features from raw data."""
        if len(df) == 0:
            return df.copy()

        feature_df = df.copy()

        # OHLCV features
        if FeatureType.OHLCV in feature_config.enabled_features:
            feature_df = self._build_ohlcv_features(feature_df)

        # Time-based features
        if FeatureType.TIME_FEATURES in feature_config.enabled_features:
            feature_df = self._build_time_features(feature_df)

        # Cyclical features
        if FeatureType.CYCLICAL_FEATURES in feature_config.enabled_features:
            feature_df = self._build_cyclical_features(feature_df)

        # Market session features
        if FeatureType.MARKET_SESSION in feature_config.enabled_features:
            feature_df = self._build_session_features(feature_df)

        return feature_df

    def _build_ohlcv_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build OHLCV-derived features."""
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            self.logger.warning("OHLC columns not found for OHLCV features")
            return df

        # Basic price features
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['weighted_close'] = (df['high'] + df['low'] + 2 * df['close'] + df['open']) / 5
        df['price_range'] = df['high'] - df['low']
        df['price_range_pct'] = (df['price_range'] / df['close']) * 100
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['shadow_ratio'] = (df['upper_shadow'] + df['lower_shadow']) / df['price_range']

        # Volume features
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_std'] = df['volume'].rolling(window=20).std()
            df['volume_change'] = df['volume'].pct_change()
            df['volume_rank'] = df['volume'].rolling(window=100).rank(pct=True)
            df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

        # Price changes
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['abs_returns'] = df['returns'].abs()

        # Price position within range
        df['close_position'] = (df['close'] - df['low']) / df['price_range']
        df['open_position'] = (df['open'] - df['low']) / df['price_range']

        return df

    def _build_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build time-based features."""
        timestamp_col = self._find_timestamp_column(df)
        if not timestamp_col or timestamp_col not in df.columns:
            self.logger.warning("No timestamp column for time features")
            return df

        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)

        # Use TimeBasedFeatures utility
        df = self.time_features.add_time_features(df, timestamp_col)
        return df

    def _build_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build cyclical time features."""
        timestamp_col = self._find_timestamp_column(df)
        if not timestamp_col or timestamp_col not in df.columns:
            self.logger.warning("No timestamp column for cyclical features")
            return df

        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)

        # Use CyclicalFeatures utility
        df = self.cyclical_features.add_cyclical_features(df, timestamp_col)
        return df

    def _build_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build market session features."""
        timestamp_col = self._find_timestamp_column(df)
        if not timestamp_col or timestamp_col not in df.columns:
            self.logger.warning("No timestamp column for session features")
            return df

        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)

        # Use SessionFeatures utility
        df = self.session_features.add_session_features(df, timestamp_col)
        return df

    def _find_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the primary timestamp column."""
        timestamp_candidates = ['timestamp', 'open_time', 'time', 'event_time', 'T', 'E']
        for candidate in timestamp_candidates:
            if candidate in df.columns:
                return candidate
        return None


class FeatureEngineeringPipeline:
    """Main pipeline for comprehensive feature engineering."""

    def __init__(
        self,
        config: DataConfig,
        feature_config: Optional[FeatureConfig] = None
    ):
        """Initialize feature engineering pipeline.

        Args:
            config: Data configuration
            feature_config: Feature engineering configuration
        """
        self.config = config
        self.feature_config = feature_config or self._get_default_feature_config()
        self.logger = get_logger(__name__)

        # Initialize components
        self.data_merger = DataMerger(config)
        self.feature_builder = FeatureBuilder(config)
        self.feature_scaler = FeatureScaler()

        # Feature schemas
        self.feature_schemas = self._define_feature_schemas()

    @performance_monitor("feature_engineering")
    def create_features(
        self,
        klines_data: Optional[pd.DataFrame] = None,
        trades_data: Optional[pd.DataFrame] = None,
        depth_data: Optional[pd.DataFrame] = None,
        book_ticker_data: Optional[pd.DataFrame] = None,
        mark_price_data: Optional[pd.DataFrame] = None,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        symbol: str = "BTCUSDT",
        timeframe: Optional[str] = None,
        include_technical_indicators: bool = True
    ) -> pd.DataFrame:
        """Create comprehensive features from multiple data sources.

        Args:
            klines_data: Kline data
            trades_data: Trade data
            depth_data: Order book depth data
            book_ticker_data: Book ticker data
            mark_price_data: Mark price data
            funding_rate_data: Funding rate data
            open_interest_data: Open interest data
            symbol: Trading symbol
            timeframe: Timeframe
            include_technical_indicators: Whether to compute technical indicators

        Returns:
            DataFrame with comprehensive features
        """
        self.logger.info(f"Creating features for {symbol} {timeframe or ''}")

        # Step 1: Merge datasets
        merged_df = self.data_merger.merge_datasets(
            klines_data=klines_data,
            trades_data=trades_data,
            depth_data=depth_data,
            book_ticker_data=book_ticker_data,
            mark_price_data=mark_price_data,
            funding_rate_data=funding_rate_data,
            open_interest_data=open_interest_data,
            symbol=symbol,
            timeframe=timeframe
        )

        if len(merged_df) == 0:
            self.logger.warning("No data available for feature creation")
            return pd.DataFrame()

        # Step 2: Build base features
        features_df = self.feature_builder.build_base_features(merged_df, self.feature_config)

        # Step 3: Add technical indicators
        if include_technical_indicators and FeatureType.TECHNICAL_INDICATORS in self.feature_config.enabled_features:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    features_df = self.technical_indicators.calculate_all_indicators(
                        features_df,
                        self.feature_config.technical_indicators
                    )
                    self.logger.debug(f"Added {len(self.feature_config.technical_indicators)} technical indicators")
            except Exception as e:
                self.logger.error(f"Technical indicators failed: {e}")

        # Step 4: Add order book features
        if FeatureType.ORDER_BOOK in self.feature_config.enabled_features and depth_data is not None:
            try:
                features_df = self._add_order_book_features(features_df, depth_data)
                self.logger.debug("Added order book features")
            except Exception as e:
                self.logger.error(f"Order book features failed: {e}")

        # Step 5: Add microstructure features
        if FeatureType.TRADE_MICROSTRUCTURE in self.feature_config.enabled_features and trades_data is not None:
            try:
                features_df = self._add_microstructure_features(features_df, trades_data)
                self.logger.debug("Added microstructure features")
            except Exception as e:
                self.logger.error(f"Microstructure features failed: {e}")

        # Step 6: Add derived features
        if FeatureType.DERIVED_FEATURES in self.feature_config.enabled_features:
            features_df = self._add_derived_features(features_df)

        # Step 7: Add risk features
        if FeatureType.RISK_FEATURES in self.feature_config.enabled_features:
            features_df = self._add_risk_features(features_df)

        # Step 8: Validate features
        self._validate_features(features_df)

        # Step 9: Sort by timestamp
        timestamp_col = self._find_timestamp_column(features_df)
        if timestamp_col and timestamp_col in features_df.columns:
            features_df = features_df.sort_values(timestamp_col).reset_index(drop=True)

        # Log feature creation summary
        self._log_feature_summary(features_df)

        return features_df

    def _add_order_book_features(
        self,
        features_df: pd.DataFrame,
        depth_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Add order book features to feature DataFrame."""
        if len(depth_data) == 0:
            return features_df

        # Process order book features
        order_book_features = OrderBookFeatures(
            levels=self.feature_config.order_book_levels
        )

        try:
            # Create order book metrics for each timestamp
            book_metrics = order_book_features.calculate_features(depth_data)

            # Merge with existing features
            if book_metrics is not None and len(book_metrics) > 0:
                timestamp_col = self._find_timestamp_column(features_df)
                if timestamp_col:
                    merged_features = pd.merge_asof(
                        features_df,
                        book_metrics,
                        left_on=timestamp_col,
                        right_on='timestamp',
                        direction='nearest',
                        tolerance=pd.Timedelta(seconds=60)
                    )
                    return merged_features
        except Exception as e:
            self.logger.error(f"Order book feature merge failed: {e}")

        return features_df

    def _add_microstructure_features(
        self,
        features_df: pd.DataFrame,
        trades_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Add trade microstructure features."""
        if len(trades_data) == 0:
            return features_df

        try:
            # Calculate microstructure features
            microstructure_features = TradeMicrostructureFeatures()
            micro_metrics = microstructure_features.calculate_features(trades_data)

            # Merge with existing features
            if micro_metrics is not None and len(micro_metrics) > 0:
                timestamp_col = self._find_timestamp_column(features_df)
                if timestamp_col:
                    merged_features = pd.merge_asof(
                        features_df,
                        micro_metrics,
                        left_on=timestamp_col,
                        right_on='timestamp',
                        direction='nearest',
                        tolerance=pd.Timedelta(seconds=30)
                    )
                    return merged_features
        except Exception as e:
            self.logger.error(f"Microstructure feature merge failed: {e}")

        return features_df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features from existing features."""
        feature_df = df.copy()

        # Interaction features
        if 'returns' in feature_df.columns and 'volume' in feature_df.columns:
            feature_df['volume_weighted_return'] = feature_df['returns'] * feature_df['volume']

        if 'close_position' in feature_df.columns and 'volume' in feature_df.columns:
            feature_df['position_volume'] = feature_df['close_position'] * feature_df['volume']

        # Volatility features
        if 'returns' in feature_df.columns:
            feature_df['realized_vol'] = feature_df['returns'].rolling(window=20).std()
            feature_df['vol_rank'] = feature_df['realized_vol'].rolling(window=100).rank(pct=True)

        # Momentum features
        if 'returns' in feature_df.columns:
            feature_df['momentum_5'] = feature_df['returns'].rolling(window=5).mean()
            feature_df['momentum_20'] = feature_df['returns'].rolling(window=20).mean()

        # Mean reversion features
        if 'returns' in feature_df.columns:
            feature_df['mean_reversion'] = -feature_df['returns'].rolling(window=10).mean()

        return feature_df

    def _add_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add risk-related features."""
        feature_df = df.copy()

        # Volatility-based risk
        if 'realized_vol' in feature_df.columns:
            feature_df['volatility_regime'] = pd.cut(
                feature_df['realized_vol'],
                bins=[0, feature_df['realized_vol'].quantile(0.25),
                      feature_df['realized_vol'].quantile(0.75), float('inf')],
                labels=['Low', 'Medium', 'High']
            )

        # Price level risk
        if 'close' in feature_df.columns:
            feature_df['price_deviation'] = feature_df['close'].rolling(window=20).apply(
                lambda x: abs(x.iloc[-1] - x.mean()) / x.mean() if len(x) > 0 else 0
            )

        # Volume anomaly detection
        if 'volume' in feature_df.columns:
            feature_df['volume_anomaly'] = abs(
                feature_df['volume'] - feature_df['volume'].rolling(window=20).mean()
            ) / feature_df['volume'].rolling(window=20).std()

        return feature_df

    def _validate_features(self, df: pd.DataFrame) -> None:
        """Validate created features for quality issues."""
        if len(df) == 0:
            return

        # Check for infinite values
        infinite_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
        if infinite_counts > 0:
            self.logger.warning(f"Found {infinite_counts} infinite values in features")

        # Check for extreme outliers (>10 standard deviations)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_counts = 0
        for col in numeric_columns:
            if col in df.columns and len(df[col].dropna()) > 10:
                mean_val = df[col].mean()
                std_val = df[col].std()
                extreme_outliers = abs(df[col] - mean_val) > 10 * std_val
                outlier_counts += extreme_outliers.sum()

        if outlier_counts > 0:
            self.logger.warning(f"Found {outlier_counts} extreme outliers in features")

        # Calculate feature quality score
        quality_score = calculate_data_quality_score(df)
        self.logger.info(f"Feature quality score: {quality_score:.3f}")

    def _find_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the primary timestamp column."""
        timestamp_candidates = ['timestamp', 'open_time', 'time', 'event_time', 'T', 'E']
        for candidate in timestamp_candidates:
            if candidate in df.columns:
                return candidate
        return None

    def _log_feature_summary(self, df: pd.DataFrame) -> None:
        """Log summary of created features."""
        if len(df) == 0:
            self.logger.warning("No features created")
            return

        feature_count = len(df.columns)
        numeric_features = len(df.select_dtypes(include=[np.number]).columns)
        total_records = len(df)

        self.logger.info(
            f"Feature creation summary: {total_records:,} records, "
            f"{feature_count} features, {numeric_features} numeric"
        )

        # Log data quality
        log_data_quality(
            data_source="feature_engineering",
            record_count=total_records,
            quality_score=calculate_data_quality_score(df),
            issues=[]
        )

    def _get_default_feature_config(self) -> FeatureConfig:
        """Get default feature configuration."""
        return FeatureConfig(
            enabled_features={
                FeatureType.OHLCV,
                FeatureType.TECHNICAL_INDICATORS,
                FeatureType.TIME_FEATURES,
                FeatureType.CYCLICAL_FEATURES,
                FeatureType.DERIVED_FEATURES,
                FeatureType.RISK_FEATURES
            },
            timeframes=['1m', '5m', '1h', '4h'],
            technical_indicators={
                'sma_periods': [5, 10, 20, 50, 100, 200],
                'ema_periods': [5, 10, 20, 50, 100, 200],
                'rsi_periods': [14, 21],
                'macd_params': {'fast': 12, 'slow': 26, 'signal': 9},
                'bb_period': 20,
                'bb_std': 2,
                'atr_periods': [14, 21]
            },
            order_book_levels=10,
            microstructure_features=True,
            time_features=True,
            cyclical_features=True,
            session_features=True,
            derived_features=True,
            risk_features=True,
            quality_threshold=0.7,
            parallel_processing=True,
            max_workers=4,
            chunk_size=10000,
            memory_limit_gb=4.0
        )

    def _define_feature_schemas(self) -> List[FeatureSchema]:
        """Define schemas for all features that can be created."""
        schemas = []

        # OHLCV base features
        ohlcv_features = [
            FeatureSchema('open', FeatureType.OHLCV, 'Opening price', ['open']),
            FeatureSchema('high', FeatureType.OHLCV, 'Highest price', ['high']),
            FeatureSchema('low', FeatureType.OHLCV, 'Lowest price', ['low']),
            FeatureSchema('close', FeatureType.OHLCV, 'Closing price', ['close']),
            FeatureSchema('volume', FeatureType.OHLCV, 'Trading volume', ['volume']),
        ]

        # Technical indicators
        technical_features = [
            FeatureSchema('sma_5', FeatureType.TECHNICAL_INDICATORS, '5-period Simple Moving Average', ['close'],
                        aggregation_method='mean', time_window=5),
            FeatureSchema('sma_20', FeatureType.TECHNICAL_INDICATORS, '20-period Simple Moving Average', ['close'],
                        aggregation_method='mean', time_window=20),
            FeatureSchema('rsi_14', FeatureType.TECHNICAL_INDICATORS, '14-period RSI', ['close'],
                        time_window=14),
            FeatureSchema('macd', FeatureType.TECHNICAL_INDICATORS, 'MACD', ['close']),
            FeatureSchema('atr', FeatureType.TECHNICAL_INDICATORS, 'Average True Range', ['high', 'low', 'close'],
                        time_window=14),
        ]

        # Time features
        time_features = [
            FeatureSchema('hour', FeatureType.TIME_FEATURES, 'Hour of day (0-23)', []),
            FeatureSchema('day_of_week', FeatureType.TIME_FEATURES, 'Day of week (0-6)', []),
            FeatureSchema('month', FeatureType.TIME_FEATURES, 'Month (1-12)', []),
        ]

        # Order book features
        order_book_features = [
            FeatureSchema('bid_ask_spread', FeatureType.ORDER_BOOK, 'Bid-ask spread', ['bid', 'ask']),
            FeatureSchema('order_book_imbalance', FeatureType.ORDER_BOOK, 'Order book imbalance', ['bids', 'asks']),
        ]

        schemas.extend(ohlcv_features)
        schemas.extend(technical_features)
        schemas.extend(time_features)
        schemas.extend(order_book_features)

        return schemas

    def get_feature_schema(self, feature_name: str) -> Optional[FeatureSchema]:
        """Get schema for a specific feature."""
        for schema in self.feature_schemas:
            if schema.name == feature_name:
                return schema
        return None

    def validate_feature_schema(self, schema: FeatureSchema) -> bool:
        """Validate a feature schema definition."""
        if not schema.name:
            return False

        if not schema.feature_type:
            return False

        if not schema.source_columns:
            return False

        # Validate aggregation method
        if schema.aggregation_method and schema.aggregation_method not in [
            'mean', 'median', 'std', 'sum', 'min', 'max', 'count'
        ]:
            return False

        return True