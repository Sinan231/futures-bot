"""
Configuration settings and management for the trading signal system.

This module provides Pydantic-based configuration classes for type safety,
validation, and environment variable loading.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class DataConfig(BaseSettings):
    """Configuration for data fetching and storage."""

    # Binance API settings
    binance_api_key: str = Field(..., env="BINANCE_API_KEY")
    binance_api_secret: str = Field(..., env="BINANCE_API_SECRET")
    binance_testnet: bool = Field(False, env="BINANCE_TESTNET")

    # Data source settings
    default_pair: str = Field("BTCUSDT", env="DEFAULT_PAIR")
    timeframes: List[str] = Field(default_factory=lambda: ["1m", "5m", "1h", "4h", "1d"])
    history_months: int = Field(7, env="HISTORY_MONTHS")
    prediction_horizon_hours: int = Field(4, env="PREDICTION_HORIZON_HOURS")

    # Storage settings
    data_storage_path: str = Field("./data", env="DATA_STORAGE_PATH")
    artifacts_storage_path: str = Field("./artifacts", env="ARTIFACTS_STORAGE_PATH")
    logs_path: str = Field("./logs", env="LOGS_PATH")

    # Rate limiting
    binance_rate_limit: int = Field(1200, env="BINANCE_RATE_LIMIT_REQUESTS_PER_MINUTE")
    websocket_reconnect_delay: int = Field(5, env="WEBSOCKET_RECONNECT_DELAY_SECONDS")
    max_reconnect_attempts: int = Field(10, env="MAX_RECONNECT_ATTEMPTS")

    @validator("timeframes", pre=True)
    def parse_timeframes(cls, v):
        if isinstance(v, str):
            return [tf.strip() for tf in v.split(",")]
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class RiskConfig(BaseSettings):
    """Configuration for risk management and position sizing."""

    # Trading parameters
    min_confidence_threshold: float = Field(0.80, env="MIN_CONFIDENCE_THRESHOLD")
    max_leverage: int = Field(20, env="MAX_LEVERAGE")
    recommended_leverage: int = Field(10, env="RECOMMENDED_LEVERAGE")
    risk_per_trade_percent: float = Field(0.5, env="RISK_PER_TRADE_PERCENT")
    max_portfolio_exposure_percent: float = Field(10.0, env="MAX_PORTFOLIO_EXPOSURE_PERCENT")

    # Signal parameters
    cooldown_seconds: int = Field(3600, env="COOLDOWN_SECONDS")
    atr_stop_loss_multiplier: float = Field(1.5, env="ATR_STOP_LOSS_MULTIPLIER")
    profit_threshold_percent: float = Field(2.0, env="PROFIT_THRESHOLD_PERCENT")
    stop_loss_threshold_percent: float = Field(1.0, env="STOP_LOSS_THRESHOLD_PERCENT")

    # Position limits
    max_positions_per_pair: int = Field(1)
    max_correlated_positions: int = Field(5)
    max_drawdown_limit: float = Field(20.0, env="MAX_DRAWDOWN_PERCENT")

    @validator("min_confidence_threshold", "risk_per_trade_percent", "max_portfolio_exposure_percent")
    def validate_percentages(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Percentage values must be between 0 and 1")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class ModelConfig(BaseSettings):
    """Configuration for model training and evaluation."""

    # Model selection criteria
    min_precision: float = Field(0.60, env="MIN_PRECISION")
    min_sharpe_ratio: float = Field(1.2, env="MIN_SHARPE_RATIO")
    max_drawdown_percent: float = Field(20.0, env="MAX_DRAWDOWN_PERCENT")
    calibration_quality_threshold: float = Field(0.05, env="CALIBRATION_QUALITY_THRESHOLD")

    # Training parameters
    random_seed: int = Field(42)
    train_val_test_split: List[float] = Field(default_factory=lambda: [0.7, 0.15, 0.15])
    max_trials: int = Field(100)
    early_stopping_patience: int = Field(10)
    cv_folds: int = Field(5)

    # Feature parameters
    feature_selection_threshold: float = Field(0.01)
    max_features: int = Field(200)
    correlation_threshold: float = Field(0.95)

    # Model hyperparameters (can be overridden in YAML config)
    model_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    @validator("train_val_test_split")
    def validate_split(cls, v):
        if len(v) != 3 or abs(sum(v) - 1.0) > 1e-6:
            raise ValueError("train_val_test_split must have 3 values summing to 1.0")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class MonitoringConfig(BaseSettings):
    """Configuration for monitoring and alerting."""

    # Performance monitoring
    performance_check_interval: int = Field(3600, env="PERFORMANCE_CHECK_INTERVAL_SECONDS")
    drift_detection_window_days: int = Field(7, env="DRIFT_DETECTION_WINDOW_DAYS")

    # Email alerts
    email_smtp_host: Optional[str] = Field(None, env="ALERT_EMAIL_SMTP_HOST")
    email_smtp_port: int = Field(587, env="ALERT_EMAIL_SMTP_PORT")
    email_username: Optional[str] = Field(None, env="ALERT_EMAIL_USERNAME")
    email_password: Optional[str] = Field(None, env="ALERT_EMAIL_PASSWORD")
    email_from: Optional[str] = Field(None, env="ALERT_EMAIL_FROM")
    email_to: Optional[str] = Field(None, env="ALERT_EMAIL_TO")

    # Slack alerts
    slack_webhook_url: Optional[str] = Field(None, env="SLACK_WEBHOOK_URL")

    # Monitoring thresholds
    performance_degradation_threshold: float = Field(0.20)  # 20% performance drop
    confidence_shift_threshold: float = Field(0.10)  # 10% confidence distribution shift
    data_freshness_threshold_minutes: int = Field(5)

    # Health checks
    health_check_interval: int = Field(30, env="HEALTH_CHECK_INTERVAL_SECONDS")
    max_memory_usage_gb: float = Field(8.0, env="MEMORY_LIMIT_GB")
    max_cpu_usage_percent: float = Field(80.0)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class TradingConfig(BaseSettings):
    """Main configuration class that combines all sub-configurations."""

    # Security
    hmac_secret_key: str = Field(..., env="HMAC_SECRET_KEY")
    jwt_secret_key: str = Field(..., env="JWT_SECRET_KEY")

    # System
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    max_workers: int = Field(4, env="MAX_WORKERS")

    # Database (if using external storage)
    database_url: str = Field("sqlite:///./trading_system.db", env="DATABASE_URL")

    # Sub-configurations
    data: DataConfig = Field(default_factory=DataConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @validator("hmac_secret_key", "jwt_secret_key")
    def validate_secret_keys(cls, v):
        if len(v) < 32:
            raise ValueError("Secret keys must be at least 32 characters long")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def load_config_from_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    return config_data or {}


def get_config(config_file: Optional[Union[str, Path]] = None) -> TradingConfig:
    """Get complete configuration with optional file override."""
    if config_file:
        config_data = load_config_from_file(config_file)
        return TradingConfig(**config_data)

    return TradingConfig()


def validate_config(config: TradingConfig) -> bool:
    """Validate configuration and return True if valid."""
    try:
        # Validate that critical paths exist or can be created
        for path_name in ["data_storage_path", "artifacts_storage_path", "logs_path"]:
            path = Path(getattr(config.data, path_name))
            path.mkdir(parents=True, exist_ok=True)

        # Validate that API keys are provided
        if not config.data.binance_api_key or not config.data.binance_api_secret:
            raise ValueError("Binance API credentials must be provided")

        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False


# Global configuration instance
_config_cache = None

def get_cached_config() -> TradingConfig:
    """Get cached configuration instance."""
    global _config_cache
    if _config_cache is None:
        _config_cache = get_config()
    return _config_cache

def reload_config() -> TradingConfig:
    """Reload configuration from files and environment."""
    global _config_cache
    _config_cache = get_config()
    return _config_cache