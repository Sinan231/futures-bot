"""
Futures Bot - Production-Ready Trading Signal System

A comprehensive trading signal system that fetches market data from Binance,
creates feature-rich datasets, trains machine learning models, and emits
real-time trading signals with proper risk management.
"""

__version__ = "1.0.0"
__author__ = "Trading System Team"
__email__ = "trading@example.com"
__description__ = "Production-Ready Trading Signal System"

from .utils import (
    get_logger,
    get_config,
    setup_logging,
)

__all__ = [
    "get_logger",
    "get_config",
    "setup_logging",
    "__version__",
]