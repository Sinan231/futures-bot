"""
Data layer for the trading signal system.

Provides comprehensive data fetching, real-time streaming, storage,
and validation for Binance market data including historical data,
WebSocket streams, and data quality management.
"""

from .binance_client import (
    BinanceClient,
    BinanceAPIError,
    BinanceRateLimitError,
)

from .historical_fetcher import (
    HistoricalDataFetcher,
    DataDownloadResult,
    DataSource,
)

from .realtime_streams import (
    RealTimeStreamManager,
    WebSocketConnection,
    StreamType,
)

from .storage import (
    DataStorageManager,
    DataValidator,
    DataQualityMetrics,
)

from .validators import (
    DataValidator,
    ValidationError,
)

__all__ = [
    # Binance client
    "BinanceClient",
    "BinanceAPIError",
    "BinanceRateLimitError",
    # Historical data
    "HistoricalDataFetcher",
    "DataDownloadResult",
    "DataSource",
    # Real-time streams
    "RealTimeStreamManager",
    "WebSocketConnection",
    "StreamType",
    # Storage and validation
    "DataStorageManager",
    "DataValidator",
    "DataQualityMetrics",
    "ValidationError",
]