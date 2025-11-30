"""
Feature engineering pipeline for trading signal system.

Provides comprehensive feature creation including technical indicators,
order book features, market microstructure, time-based features,
and feature scaling with persistence for real-time and historical data.
"""

from .engineering import (
    FeatureEngineeringPipeline,
    DataMerger,
    FeatureBuilder,
    FeatureSchema,
)

from .indicators import (
    TechnicalIndicators,
    MomentumIndicators,
    VolatilityIndicators,
    VolumeIndicators,
    TrendIndicators,
    PricePatterns,
)

from .microstructure import (
    OrderBookFeatures,
    TradeMicrostructureFeatures,
    MarketImpactFeatures,
    LiquidityFeatures,
)

from .time_features import (
    TimeBasedFeatures,
    CyclicalFeatures,
    SessionFeatures,
    MarketHoursFeatures,
)

from .scaling import (
    FeatureScaler,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    ScalingPersistence,
)

__all__ = [
    # Core feature engineering
    "FeatureEngineeringPipeline",
    "DataMerger",
    "FeatureBuilder",
    "FeatureSchema",
    # Technical indicators
    "TechnicalIndicators",
    "MomentumIndicators",
    "VolatilityIndicators",
    "VolumeIndicators",
    "TrendIndicators",
    "PricePatterns",
    # Market microstructure
    "OrderBookFeatures",
    "TradeMicrostructureFeatures",
    "MarketImpactFeatures",
    "LiquidityFeatures",
    # Time features
    "TimeBasedFeatures",
    "CyclicalFeatures",
    "SessionFeatures",
    "MarketHoursFeatures",
    # Feature scaling
    "FeatureScaler",
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "ScalingPersistence",
]