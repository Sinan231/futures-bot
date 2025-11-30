"""
Utility modules for the trading signal system.

This package provides common utilities for logging, configuration,
security, data validation, and helper functions used across
all components of the trading system.
"""

from .logging import (
    get_logger,
    setup_logging,
    get_structured_logger,
    log_performance,
    log_signal,
)

from .config import (
    get_config,
    reload_config,
    validate_config,
)

from .security import (
    generate_hmac_signature,
    verify_hmac_signature,
    encrypt_sensitive_data,
    decrypt_sensitive_data,
    generate_jwt_token,
    verify_jwt_token,
)

from .helpers import (
    retry_with_exponential_backoff,
    rate_limiter,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_volatility,
    normalize_timestamp,
    format_trading_pair,
    parse_timeframe_to_seconds,
)

__all__ = [
    # Logging
    "get_logger",
    "setup_logging",
    "get_structured_logger",
    "log_performance",
    "log_signal",
    # Configuration
    "get_config",
    "reload_config",
    "validate_config",
    # Security
    "generate_hmac_signature",
    "verify_hmac_signature",
    "encrypt_sensitive_data",
    "decrypt_sensitive_data",
    "generate_jwt_token",
    "verify_jwt_token",
    # Helpers
    "retry_with_exponential_backoff",
    "rate_limiter",
    "calculate_returns",
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
    "calculate_volatility",
    "normalize_timestamp",
    "format_trading_pair",
    "parse_timeframe_to_seconds",
]