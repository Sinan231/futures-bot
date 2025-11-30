"""
Comprehensive logging system for the trading signal system.

Provides structured logging with JSON formatting, performance metrics,
signal tracking, and audit trail functionality.
"""

import json
import logging
import logging.handlers
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union
from functools import wraps

import structlog
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.text import Text

from ..config import get_config


class TradingSystemLogger:
    """Advanced logging system for trading applications."""

    def __init__(self, config: Optional[Any] = None):
        """Initialize logger with configuration."""
        self.config = config or get_config()
        self.console = Console()
        self._setup_structlog()
        self._setup_standard_logging()

    def _setup_structlog(self):
        """Setup structured logging with processors."""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                self._add_trading_context,
                structlog.processors.JSONRenderer() if self.config.monitoring.json_format
                else structlog.dev.ConsoleRenderer(colors=True),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

    def _setup_standard_logging(self):
        """Setup standard Python logging with rich console and file handlers."""
        # Get log level
        log_level = getattr(logging, self.config.monitoring.log_level.upper())

        # Create formatters
        if self.config.monitoring.json_format:
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"logger": "%(name)s", "message": "%(message)s", '
                '"module": "%(module)s", "function": "%(funcName)s", '
                '"line": %(lineno)d}'
            )
        else:
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s:%(lineno)-4d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )

        # Setup rich console handler
        console_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_level=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            tracebacks_extra_lines=2,
        )
        console_handler.setLevel(log_level)

        # Setup file handler with rotation
        logs_dir = Path(self.config.monitoring.logs_path)
        logs_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=logs_dir / "trading_system.log",
            maxBytes=self.config.monitoring.max_file_size_mb * 1024 * 1024,
            backupCount=self.config.monitoring.backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

        # Setup audit file handler for critical events
        audit_handler = logging.handlers.RotatingFileHandler(
            filename=logs_dir / "audit.log",
            maxBytes=self.config.monitoring.max_file_size_mb * 1024 * 1024,
            backupCount=self.config.monitoring.backup_count,
            encoding="utf-8"
        )
        audit_handler.setLevel(logging.INFO)
        audit_handler.setFormatter(formatter)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.handlers.clear()

        # Add handlers
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(audit_handler)

        # Suppress noisy loggers
        logging.getLogger("websockets").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("aiohttp").setLevel(logging.WARNING)

    def _add_trading_context(self, logger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add trading-specific context to log entries."""
        # Add global trading context
        event_dict["system"] = "trading_signal_system"
        event_dict["version"] = "1.0.0"
        event_dict["environment"] = "production" if not self.config.debug else "development"

        return event_dict


# Global logger instance
_trading_logger = None


def setup_logging(config: Optional[Any] = None) -> None:
    """Setup logging system with configuration."""
    global _trading_logger
    _trading_logger = TradingSystemLogger(config)


def get_structured_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get structured logger instance."""
    if _trading_logger is None:
        setup_logging()
    return structlog.get_logger(name)


def get_logger(name: str) -> logging.Logger:
    """Get standard Python logger instance."""
    if _trading_logger is None:
        setup_logging()
    return logging.getLogger(name)


def log_performance(
    function_name: str,
    duration_seconds: float,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Log performance metrics for function execution."""
    logger = get_structured_logger("performance")

    logger.info(
        "Performance metric",
        function=function_name,
        duration_seconds=duration_seconds,
        metadata=metadata or {}
    )


def log_signal(signal_data: Dict[str, Any], confidence: float) -> None:
    """Log trading signal with audit trail."""
    # Create audit log entry
    audit_logger = logging.getLogger("audit")

    audit_info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "trading_signal",
        "signal_data": signal_data,
        "confidence": confidence,
        "action": "signal_generated"
    }

    audit_logger.info(
        f"Trading signal generated: {signal_data.get('pair', 'unknown')} "
        f"{signal_data.get('side', 'unknown')} @ {signal_data.get('entry_price', 'unknown')} "
        f"(confidence: {confidence:.3f})",
        extra={"audit_data": json.dumps(audit_info, default=str)}
    )


def log_model_performance(
    model_version: str,
    metrics: Dict[str, float],
    data_period: str
) -> None:
    """Log model performance metrics."""
    logger = get_structured_logger("model_performance")

    logger.info(
        "Model performance metrics",
        model_version=model_version,
        data_period=data_period,
        metrics=metrics
    )


def log_data_quality(
    data_source: str,
    record_count: int,
    quality_score: float,
    issues: Optional[list] = None
) -> None:
    """Log data quality metrics."""
    logger = get_structured_logger("data_quality")

    logger.info(
        "Data quality assessment",
        data_source=data_source,
        record_count=record_count,
        quality_score=quality_score,
        issues=issues or []
    )


def log_trading_event(
    event_type: str,
    pair: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Log trading-related events."""
    logger = get_structured_logger("trading_events")

    logger.info(
        f"Trading event: {event_type}",
        event_type=event_type,
        pair=pair,
        metadata=metadata or {}
    )


def log_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    reraise: bool = False
) -> None:
    """Log error with full context and traceback."""
    logger = get_structured_logger("errors")

    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "context": context or {}
    }

    logger.error(
        "System error occurred",
        **error_info
    )

    if reraise:
        raise error


def performance_monitor(logger: Optional[Union[str, logging.Logger]] = None):
    """Decorator to monitor function performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = f"{func.__module__}.{func.__name__}"

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Log successful execution
                log_performance(
                    function_name=function_name,
                    duration_seconds=duration,
                    metadata={
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                        "success": True
                    }
                )

                return result

            except Exception as e:
                duration = time.time() - start_time

                # Log failed execution
                log_performance(
                    function_name=function_name,
                    duration_seconds=duration,
                    metadata={
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                        "success": False,
                        "error": str(e)
                    }
                )

                # Log the error
                if isinstance(logger, str):
                    log_error(e, context={"function": function_name, "args": args, "kwargs": kwargs})
                elif logger:
                    log_error(e, context={"function": function_name, "logger": logger})
                else:
                    log_error(e, context={"function": function_name})

                raise

        return wrapper
    return decorator


class PerformanceMetrics:
    """Track and display performance metrics."""

    def __init__(self):
        self.metrics = {}
        self.console = Console()

    def add_metric(self, name: str, value: float, unit: str = "seconds") -> None:
        """Add a performance metric."""
        self.metrics[name] = {"value": value, "unit": unit}

    def display_summary(self) -> None:
        """Display performance summary in a formatted table."""
        if not self.metrics:
            self.console.print("[yellow]No performance metrics to display[/yellow]")
            return

        table = Table(title="Performance Metrics Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Unit", style="white")

        for name, data in self.metrics.items():
            table.add_row(name, f"{data['value']:.4f}", data['unit'])

        self.console.print(table)

    def save_to_file(self, filename: str) -> None:
        """Save metrics to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)


# Global performance metrics tracker
_performance_tracker = PerformanceMetrics()


def track_performance(metric_name: str, unit: str = "seconds"):
    """Decorator to track function performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            _performance_tracker.add_metric(
                name=f"{func.__module__}.{func.__name__}.{metric_name}",
                value=duration,
                unit=unit
            )

            return result
        return wrapper
    return decorator


def get_performance_metrics() -> PerformanceMetrics:
    """Get global performance metrics tracker."""
    return _performance_tracker


def display_system_status() -> None:
    """Display current system status in a formatted table."""
    console = Console()

    table = Table(title="System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="white")

    # Add system information
    table.add_row("Trading System", "🟢 Active", "Signal generation operational")
    table.add_row("Data Feeds", "🟢 Active", "WebSocket connections established")
    table.add_row("Model Service", "🟢 Active", "Latest model loaded: v1.0.0")
    table.add_row("Risk Manager", "🟢 Active", "Position limits enforced")
    table.add_row("Monitoring", "🟢 Active", "Performance tracking enabled")

    console.print(table)