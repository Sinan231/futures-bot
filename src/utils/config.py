"""
Configuration management utilities for the trading signal system.

Provides configuration loading, validation, caching, and environment
variable management with type safety and validation.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union, Type, TypeVar, get_type_hints
from functools import lru_cache
import threading

from ..config import (
    TradingConfig,
    DataConfig,
    RiskConfig,
    ModelConfig,
    MonitoringConfig,
)

T = TypeVar('T')


class ConfigManager:
    """Thread-safe configuration manager with caching and validation."""

    def __init__(self):
        """Initialize configuration manager."""
        self._config_lock = threading.RLock()
        self._config_cache: Optional[TradingConfig] = None
        self._last_modified: Optional[float] = None

    def load_config(
        self,
        config_file: Optional[Union[str, Path]] = None,
        force_reload: bool = False
    ) -> TradingConfig:
        """Load configuration from file and environment variables."""
        with self._config_lock:
            # Check cache validity
            if not force_reload and self._config_cache is not None:
                if config_file is None or not self._is_config_modified(config_file):
                    return self._config_cache

            # Load configuration data
            config_data = self._load_config_data(config_file)

            # Validate and create configuration object
            config = TradingConfig(**config_data)

            # Validate configuration
            self._validate_full_config(config)

            # Cache the configuration
            self._config_cache = config
            self._last_modified = self._get_config_file_modified_time(config_file)

            return config

    def _load_config_data(self, config_file: Optional[Union[str, Path]]) -> Dict[str, Any]:
        """Load configuration data from YAML file and environment variables."""
        config_data = {}

        # Load from file if provided
        if config_file:
            config_file = Path(config_file)
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        file_config = yaml.safe_load(f) or {}
                        config_data.update(file_config)
                except yaml.YAMLError as e:
                    raise ValueError(f"Invalid YAML configuration file: {e}")
                except Exception as e:
                    raise ValueError(f"Error loading configuration file: {e}")

        # Override with environment variables
        env_overrides = self._get_env_overrides()
        if env_overrides:
            config_data.update(env_overrides)

        return config_data

    def _get_env_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides from environment variables."""
        overrides = {}

        # Define environment variable mappings
        env_mappings = {
            'DEBUG': 'debug',
            'LOG_LEVEL': 'log_level',
            'MAX_WORKERS': 'max_workers',
            'DATABASE_URL': 'database_url',
            'DEFAULT_PAIR': 'data.default_pair',
            'TIMEFRAMES': 'data.timeframes',
            'HISTORY_MONTHS': 'data.history_months',
            'MIN_CONFIDENCE_THRESHOLD': 'risk.min_confidence_threshold',
            'MAX_LEVERAGE': 'risk.max_leverage',
            'RISK_PER_TRADE_PERCENT': 'risk.risk_per_trade_percent',
            'MIN_PRECISION': 'model.min_precision',
            'MIN_SHARPE_RATIO': 'model.min_sharpe_ratio',
            'RANDOM_SEED': 'model.random_seed',
            'PERFORMANCE_CHECK_INTERVAL_SECONDS': 'monitoring.performance_check_interval',
            'DRIFT_DETECTION_WINDOW_DAYS': 'monitoring.drift_detection_window_days',
        }

        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Set nested configuration values
                self._set_nested_value(overrides, config_path, self._parse_env_value(value))

        return overrides

    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any) -> None:
        """Set nested configuration value from dot notation path."""
        keys = path.split('.')
        current = data

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Handle booleans
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # Handle numbers
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # Handle JSON values
        if value.startswith(('{', '[')):
            try:
                import json
                return json.loads(value)
            except json.JSONDecodeError:
                pass

        # Handle comma-separated lists
        if ',' in value:
            return [item.strip() for item in value.split(',')]

        # Default to string
        return value

    def _is_config_modified(self, config_file: Optional[Union[str, Path]]) -> bool:
        """Check if configuration file has been modified since last load."""
        if config_file is None:
            return False

        current_modified = self._get_config_file_modified_time(config_file)
        return current_modified != self._last_modified

    def _get_config_file_modified_time(self, config_file: Optional[Union[str, Path]]) -> Optional[float]:
        """Get modification time of configuration file."""
        if config_file is None:
            return None

        try:
            config_path = Path(config_file)
            return config_path.stat().st_mtime if config_path.exists() else None
        except OSError:
            return None

    def _validate_full_config(self, config: TradingConfig) -> None:
        """Validate complete configuration."""
        # Validate API credentials
        if not config.data.binance_api_key or not config.data.binance_api_secret:
            raise ValueError("Binance API credentials are required")

        # Validate secret keys length
        if len(config.hmac_secret_key) < 32:
            raise ValueError("HMAC secret key must be at least 32 characters long")

        if len(config.jwt_secret_key) < 32:
            raise ValueError("JWT secret key must be at least 32 characters long")

        # Validate risk parameters
        if not 0 < config.risk.min_confidence_threshold <= 1:
            raise ValueError("Minimum confidence threshold must be between 0 and 1")

        if config.risk.max_leverage <= 0:
            raise ValueError("Maximum leverage must be greater than 0")

        if not 0 < config.risk.risk_per_trade_percent <= 1:
            raise ValueError("Risk per trade must be between 0 and 1")

        # Validate model parameters
        if not 0 < config.model.min_precision <= 1:
            raise ValueError("Minimum precision must be between 0 and 1")

        if config.model.min_sharpe_ratio <= 0:
            raise ValueError("Minimum Sharpe ratio must be greater than 0")

        # Validate data parameters
        if config.data.history_months <= 0:
            raise ValueError("History months must be greater than 0")

        if not config.data.timeframes:
            raise ValueError("At least one timeframe must be specified")

        # Create required directories
        self._ensure_directories(config)

    def _ensure_directories(self, config: TradingConfig) -> None:
        """Ensure required directories exist."""
        directories = [
            config.data.data_storage_path,
            config.data.artifacts_storage_path,
            config.data.logs_path,
            Path(config.data.data_storage_path) / "raw",
            Path(config.data.data_storage_path) / "processed",
            Path(config.data.data_storage_path) / "features",
            Path(config.data.data_storage_path) / "signals",
            Path(config.data.artifacts_storage_path) / "models",
            Path(config.data.artifacts_storage_path) / "scalers",
            Path(config.data.artifacts_storage_path) / "features",
            Path(config.data.artifacts_storage_path) / "reports",
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def save_config(self, config: TradingConfig, config_file: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with self._config_lock:
            config_file = Path(config_file)

            # Convert config to dictionary
            config_dict = config.model_dump(exclude_none=True, exclude_unset=True)

            # Save to file
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)

            # Update cache
            self._config_cache = config
            self._last_modified = config_file.stat().st_mtime

    def get_sub_config(self, config_type: Type[T]) -> T:
        """Get a specific sub-configuration."""
        full_config = self.load_config()

        config_type_name = config_type.__name__.lower().replace('config', '')

        if config_type == DataConfig:
            return full_config.data
        elif config_type == RiskConfig:
            return full_config.risk
        elif config_type == ModelConfig:
            return full_config.model
        elif config_type == MonitoringConfig:
            return full_config.monitoring
        else:
            raise ValueError(f"Unknown configuration type: {config_type}")

    def reload_config(self) -> TradingConfig:
        """Force reload configuration from disk and environment."""
        return self.load_config(force_reload=True)

    def clear_cache(self) -> None:
        """Clear configuration cache."""
        with self._config_lock:
            self._config_cache = None
            self._last_modified = None


# Global configuration manager instance
_config_manager = ConfigManager()


@lru_cache(maxsize=1)
def get_config(config_file: Optional[Union[str, Path]] = None) -> TradingConfig:
    """Get trading configuration with caching."""
    return _config_manager.load_config(config_file)


def reload_config() -> TradingConfig:
    """Reload configuration from disk and environment variables."""
    return _config_manager.reload_config()


def validate_config(config: Optional[TradingConfig] = None) -> bool:
    """Validate configuration and return True if valid."""
    try:
        if config is None:
            config = get_config()

        _config_manager._validate_full_config(config)
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False


def save_config(config: TradingConfig, config_file: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    _config_manager.save_config(config, config_file)


def get_data_config() -> DataConfig:
    """Get data configuration."""
    return _config_manager.get_sub_config(DataConfig)


def get_risk_config() -> RiskConfig:
    """Get risk configuration."""
    return _config_manager.get_sub_config(RiskConfig)


def get_model_config() -> ModelConfig:
    """Get model configuration."""
    return _config_manager.get_sub_config(ModelConfig)


def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration."""
    return _config_manager.get_sub_config(MonitoringConfig)


class ConfigValidator:
    """Advanced configuration validator with detailed error reporting."""

    def __init__(self):
        """Initialize validator."""
        self.errors = []
        self.warnings = []

    def validate(self, config: TradingConfig) -> tuple[bool, list[str], list[str]]:
        """Validate configuration and return (is_valid, errors, warnings)."""
        self.errors.clear()
        self.warnings.clear()

        self._validate_credentials(config)
        self._validate_risk_parameters(config)
        self._validate_model_parameters(config)
        self._validate_data_parameters(config)
        self._validate_monitoring_parameters(config)

        return len(self.errors) == 0, self.errors.copy(), self.warnings.copy()

    def _validate_credentials(self, config: TradingConfig) -> None:
        """Validate API credentials and security settings."""
        if not config.data.binance_api_key:
            self.errors.append("Binance API key is required")

        if not config.data.binance_api_secret:
            self.errors.append("Binance API secret is required")

        if len(config.hmac_secret_key) < 32:
            self.errors.append("HMAC secret key must be at least 32 characters")

        if len(config.jwt_secret_key) < 32:
            self.errors.append("JWT secret key must be at least 32 characters")

    def _validate_risk_parameters(self, config: TradingConfig) -> None:
        """Validate risk management parameters."""
        if not 0 < config.risk.min_confidence_threshold <= 1:
            self.errors.append("Minimum confidence threshold must be between 0 and 1")

        if config.risk.max_leverage <= 0:
            self.errors.append("Maximum leverage must be greater than 0")

        if config.risk.recommended_leverage > config.risk.max_leverage:
            self.warnings.append("Recommended leverage exceeds maximum leverage")

        if not 0 < config.risk.risk_per_trade_percent <= 1:
            self.errors.append("Risk per trade must be between 0 and 1%")

        if config.risk.max_portfolio_exposure_percent > 1:
            self.errors.append("Maximum portfolio exposure must be less than or equal to 100%")

        if config.risk.atr_stop_loss_multiplier <= 0:
            self.errors.append("ATR stop loss multiplier must be greater than 0")

    def _validate_model_parameters(self, config: TradingConfig) -> None:
        """Validate model training parameters."""
        if not 0 < config.model.min_precision <= 1:
            self.errors.append("Minimum precision must be between 0 and 1")

        if config.model.min_sharpe_ratio <= 0:
            self.errors.append("Minimum Sharpe ratio must be greater than 0")

        if config.model.max_drawdown_percent <= 0:
            self.errors.append("Maximum drawdown must be greater than 0")

        if abs(sum(config.model.train_val_test_split) - 1.0) > 1e-6:
            self.errors.append("Train/validation/test split must sum to 1.0")

        if any(s <= 0 for s in config.model.train_val_test_split):
            self.errors.append("All train/validation/test split values must be positive")

        if config.model.cv_folds < 2:
            self.errors.append("Cross-validation folds must be at least 2")

    def _validate_data_parameters(self, config: TradingConfig) -> None:
        """Validate data fetching and storage parameters."""
        if config.data.history_months <= 0:
            self.errors.append("History months must be greater than 0")

        if not config.data.timeframes:
            self.errors.append("At least one timeframe must be specified")

        valid_timeframes = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}
        invalid_timeframes = set(config.data.timeframes) - valid_timeframes
        if invalid_timeframes:
            self.errors.append(f"Invalid timeframes: {invalid_timeframes}")

        if config.data.binance_rate_limit <= 0:
            self.errors.append("Binance rate limit must be greater than 0")

        if config.data.websocket_reconnect_delay <= 0:
            self.errors.append("WebSocket reconnect delay must be greater than 0")

        if config.data.max_reconnect_attempts <= 0:
            self.errors.append("Maximum reconnect attempts must be greater than 0")

    def _validate_monitoring_parameters(self, config: TradingConfig) -> None:
        """Validate monitoring and alerting parameters."""
        if config.monitoring.performance_check_interval <= 0:
            self.errors.append("Performance check interval must be greater than 0")

        if config.monitoring.drift_detection_window_days <= 0:
            self.errors.append("Drift detection window must be greater than 0")

        if not 0 < config.monitoring.performance_degradation_threshold <= 1:
            self.errors.append("Performance degradation threshold must be between 0 and 1")

        if not 0 < config.monitoring.confidence_shift_threshold <= 1:
            self.errors.append("Confidence shift threshold must be between 0 and 1")

        if config.monitoring.health_check_interval <= 0:
            self.errors.append("Health check interval must be greater than 0")

        if config.monitoring.max_memory_usage_gb <= 0:
            self.errors.append("Maximum memory usage must be greater than 0")

        if not 0 < config.monitoring.max_cpu_usage_percent <= 100:
            self.errors.append("Maximum CPU usage must be between 0 and 100")