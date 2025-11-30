"""
Data validation and quality assurance for trading signal system.

Provides comprehensive data validation, quality checks, anomaly detection,
and data integrity verification for all market data types including
OHLCV, trades, order book depth, and real-time streams.
"""

import statistics
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd

from ..utils.logging import get_logger, log_data_quality
from ..utils.helpers import calculate_data_quality_score, is_outlier, remove_outliers
from .historical_fetcher import DataSource


class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationRule(Enum):
    """Type of validation rules."""
    REQUIRED_FIELDS = "required_fields"
    DATA_TYPES = "data_types"
    VALUE_RANGES = "value_ranges"
    TEMPORAL_ORDER = "temporal_order"
    DUPLICATE_DATA = "duplicate_data"
    MISSING_DATA = "missing_data"
    OUTLIER_DETECTION = "outlier_detection"
    CONSISTENCY_CHECKS = "consistency_checks"
    BUSINESS_RULES = "business_rules"
    TIMESTAMP_FORMAT = "timestamp_format"
    DATA_FRESHNESS = "data_freshness"


@dataclass
class ValidationError:
    """Represents a data validation error."""
    rule: ValidationRule
    level: ValidationLevel
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    expected: Optional[Any] = None
    row_index: Optional[int] = None
    column: Optional[str] = None
    count: int = 1
    percentage: float = 0.0


@dataclass
class ValidationResult:
    """Result of data validation operation."""
    is_valid: bool
    total_records: int
    valid_records: int
    errors: List[ValidationError]
    warnings: List[ValidationError]
    quality_score: float
    validation_time: datetime
    data_source: DataSource
    symbol: str
    timeframe: Optional[str]
    fixable_issues: List[str]
    critical_issues: List[str]


class DataValidator:
    """Comprehensive data validator for trading market data."""

    def __init__(
        self,
        strict_mode: bool = False,
        outlier_threshold: float = 3.0,
        max_missing_ratio: float = 0.05,
        freshness_threshold_minutes: int = 5
    ):
        """Initialize data validator.

        Args:
            strict_mode: Whether to be strict with validation rules
            outlier_threshold: Standard deviation threshold for outlier detection
            max_missing_ratio: Maximum allowed ratio of missing data
            freshness_threshold_minutes: Maximum allowed data age in minutes
        """
        self.strict_mode = strict_mode
        self.outlier_threshold = outlier_threshold
        self.max_missing_ratio = max_missing_ratio
        self.freshness_threshold = timedelta(minutes=freshness_threshold_minutes)
        self.logger = get_logger(__name__)

        # Validation rules configuration
        self.required_fields = self._get_required_fields_config()
        self.value_ranges = self._get_value_ranges_config()
        self.business_rules = self._get_business_rules_config()

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        data_source: DataSource,
        symbol: str,
        timeframe: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Comprehensive validation of a DataFrame.

        Args:
            df: DataFrame to validate
            data_source: Type of data source
            symbol: Trading symbol
            timeframe: Timeframe (for kline data)
            context: Additional validation context

        Returns:
            Detailed validation result
        """
        start_time = datetime.now(timezone.utc)

        if len(df) == 0:
            return ValidationResult(
                is_valid=False,
                total_records=0,
                valid_records=0,
                errors=[
                    ValidationError(
                        rule=ValidationRule.REQUIRED_FIELDS,
                        level=ValidationLevel.CRITICAL,
                        message="Empty DataFrame",
                        count=1,
                        percentage=100.0
                    )
                ],
                warnings=[],
                quality_score=0.0,
                validation_time=start_time,
                data_source=data_source,
                symbol=symbol,
                timeframe=timeframe,
                fixable_issues=[],
                critical_issues=["Empty DataFrame"]
            )

        all_errors = []
        all_warnings = []

        # Run validation rules
        self._validate_required_fields(df, data_source, all_errors, all_warnings)
        self._validate_data_types(df, data_source, all_errors, all_warnings)
        self._validate_value_ranges(df, data_source, all_errors, all_warnings)
        self._validate_temporal_order(df, data_source, all_errors, all_warnings)
        self._validate_duplicates(df, data_source, all_errors, all_warnings)
        self._validate_missing_data(df, data_source, all_errors, all_warnings)
        self._validate_outliers(df, data_source, all_errors, all_warnings)
        self._validate_consistency(df, data_source, all_errors, all_warnings)
        self._validate_business_rules(df, data_source, all_errors, all_warnings, context)
        self._validate_timestamp_format(df, data_source, all_errors, all_warnings)
        self._validate_data_freshness(df, data_source, all_errors, all_warnings)

        # Categorize by severity
        errors = [e for e in all_errors if e.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]]
        warnings = [e for e in all_errors if e.level == ValidationLevel.WARNING] + \
                   [e for e in all_errors if e.level == ValidationLevel.INFO]

        # Calculate validation results
        total_errors = sum(e.count for e in errors)
        total_warnings = sum(e.count for e in warnings)
        valid_records = max(0, len(df) - total_errors)

        quality_score = self._calculate_validation_quality_score(
            len(df), valid_records, total_errors, total_warnings
        )

        # Identify fixable and critical issues
        fixable_issues = list(set(
            e.message for e in errors + warnings
            if e.rule in [
                ValidationRule.DUPLICATE_DATA,
                ValidationRule.MISSING_DATA,
                ValidationRule.OUTLIER_DETECTION
            ]
        ))

        critical_issues = list(set(
            e.message for e in errors
            if e.level == ValidationLevel.CRITICAL
        ))

        result = ValidationResult(
            is_valid=len(errors) == 0,
            total_records=len(df),
            valid_records=valid_records,
            errors=errors,
            warnings=warnings,
            quality_score=quality_score,
            validation_time=start_time,
            data_source=data_source,
            symbol=symbol,
            timeframe=timeframe,
            fixable_issues=fixable_issues,
            critical_issues=critical_issues
        )

        # Log validation result
        self._log_validation_result(result)

        return result

    def _validate_required_fields(
        self,
        df: pd.DataFrame,
        data_source: DataSource,
        errors: List[ValidationError],
        warnings: List[ValidationError]
    ) -> None:
        """Validate that required fields are present."""
        required = self.required_fields.get(data_source, [])
        missing_fields = [field for field in required if field not in df.columns]

        if missing_fields:
            errors.append(ValidationError(
                rule=ValidationRule.REQUIRED_FIELDS,
                level=ValidationLevel.CRITICAL,
                message=f"Missing required fields: {missing_fields}",
                count=len(missing_fields),
                percentage=100.0
            ))

    def _validate_data_types(
        self,
        df: pd.DataFrame,
        data_source: DataSource,
        errors: List[ValidationError],
        warnings: List[ValidationError]
    ) -> None:
        """Validate data types and formats."""
        # Expected data types for common fields
        expected_types = {
            'timestamp': ['datetime64[ns]', 'datetime64[us]', 'datetime64[s]', 'object'],
            'open_time': ['datetime64[ns]', 'datetime64[us]', 'datetime64[s]', 'object'],
            'close_time': ['datetime64[ns]', 'datetime64[us]', 'datetime64[s]', 'object'],
            'time': ['datetime64[ns]', 'datetime64[us]', 'datetime64[s]', 'object'],
            'open': ['float64', 'int64', 'float32', 'int32'],
            'high': ['float64', 'int64', 'float32', 'int32'],
            'low': ['float64', 'int64', 'float32', 'int32'],
            'close': ['float64', 'int64', 'float32', 'int32'],
            'volume': ['float64', 'int64', 'float32', 'int32'],
            'price': ['float64', 'int64', 'float32', 'int32'],
            'qty': ['float64', 'int64', 'float32', 'int32'],
            'quote_qty': ['float64', 'int64', 'float32', 'int32'],
            'mark_price': ['float64', 'float32'],
            'index_price': ['float64', 'float32'],
            'funding_rate': ['float64', 'float32'],
            'open_interest': ['float64', 'int64', 'float32', 'int32'],
        }

        for column in df.columns:
            if column.lower() in expected_types:
                expected = expected_types[column.lower()]
                actual_dtype = str(df[column].dtype)

                if actual_dtype not in expected:
                    # Try to convert
                    try:
                        if 'datetime' in expected[0]:
                            df[column] = pd.to_datetime(df[column], utc=True, errors='coerce')
                        elif 'float' in expected[0] or 'int' in expected[0]:
                            df[column] = pd.to_numeric(df[column], errors='coerce')

                        # Check if conversion succeeded
                        null_count = df[column].isnull().sum()
                        if null_count > len(df) * 0.1:  # More than 10% became null
                            warnings.append(ValidationError(
                                rule=ValidationRule.DATA_TYPES,
                                level=ValidationLevel.WARNING,
                                message=f"Type conversion issues in {column}: {actual_dtype} -> expected",
                                field=column,
                                count=null_count,
                                percentage=null_count / len(df) * 100
                            ))
                    except Exception as e:
                        errors.append(ValidationError(
                            rule=ValidationRule.DATA_TYPES,
                            level=ValidationLevel.ERROR,
                            message=f"Cannot convert {column}: {str(e)}",
                            field=column,
                            value=actual_dtype,
                            expected=expected[0],
                            count=len(df),
                            percentage=100.0
                        ))

    def _validate_value_ranges(
        self,
        df: pd.DataFrame,
        data_source: DataSource,
        errors: List[ValidationError],
        warnings: List[ValidationError]
    ) -> None:
        """Validate that values are within expected ranges."""
        ranges = self.value_ranges.get(data_source, {})

        for column, (min_val, max_val) in ranges.items():
            if column not in df.columns:
                continue

            # Count violations
            violations = df[(df[column] < min_val) | (df[column] > max_val)]
            violation_count = len(violations)

            if violation_count > 0:
                severity = ValidationLevel.ERROR if self.strict_mode else ValidationLevel.WARNING
                errors.append(ValidationError(
                    rule=ValidationRule.VALUE_RANGES,
                    level=severity,
                    message=f"Values out of range in {column}: {violation_count} records",
                    field=column,
                    expected=f"{min_val} to {max_val}",
                    count=violation_count,
                    percentage=violation_count / len(df) * 100
                ))

    def _validate_temporal_order(
        self,
        df: pd.DataFrame,
        data_source: DataSource,
        errors: List[ValidationError],
        warnings: List[ValidationError]
    ) -> None:
        """Validate temporal ordering of timestamps."""
        # Find timestamp column
        timestamp_col = self._find_timestamp_column(df)
        if not timestamp_col:
            errors.append(ValidationError(
                rule=ValidationRule.TEMPORAL_ORDER,
                level=ValidationLevel.ERROR,
                message="No timestamp column found",
                count=len(df),
                percentage=100.0
            ))
            return

        # Check if timestamps are monotonically increasing
        try:
            if timestamp_col in df.columns:
                df_sorted = df.sort_values(timestamp_col)
                timestamp_col_sorted = df_sorted[timestamp_col]

                # Detect out-of-order timestamps
                out_of_order = timestamp_col_sorted.diff().dropna() < pd.Timedelta(seconds=0)
                out_of_order_count = out_of_order.sum()

                if out_of_order_count > 0:
                    errors.append(ValidationError(
                        rule=ValidationRule.TEMPORAL_ORDER,
                        level=ValidationLevel.ERROR,
                        message=f"Out-of-order timestamps: {out_of_order_count} records",
                        field=timestamp_col,
                        count=out_of_order_count,
                        percentage=out_of_order_count / len(df) * 100
                    ))

                # Check for duplicate timestamps
                duplicate_timestamps = timestamp_col_sorted.duplicated().sum()
                if duplicate_timestamps > 0:
                    warnings.append(ValidationError(
                        rule=ValidationRule.TEMPORAL_ORDER,
                        level=ValidationLevel.WARNING,
                        message=f"Duplicate timestamps: {duplicate_timestamps} records",
                        field=timestamp_col,
                        count=duplicate_timestamps,
                        percentage=duplicate_timestamps / len(df) * 100
                    ))

        except Exception as e:
            errors.append(ValidationError(
                rule=ValidationRule.TEMPORAL_ORDER,
                level=ValidationLevel.ERROR,
                message=f"Timestamp validation failed: {str(e)}",
                count=len(df),
                percentage=100.0
            ))

    def _validate_duplicates(
        self,
        df: pd.DataFrame,
        data_source: DataSource,
        errors: List[ValidationError],
        warnings: List[ValidationError]
    ) -> None:
        """Validate for duplicate records."""
        if len(df) == 0:
            return

        # Find columns that should be unique for this data source
        unique_columns = self._get_unique_columns(data_source)
        if not unique_columns:
            return

        # Check for duplicates
        duplicate_mask = df.duplicated(subset=unique_columns, keep=False)
        duplicate_count = duplicate_mask.sum()

        if duplicate_count > 0:
            severity = ValidationLevel.ERROR if self.strict_mode else ValidationLevel.WARNING
            errors.append(ValidationError(
                rule=ValidationRule.DUPLICATE_DATA,
                level=severity,
                message=f"Duplicate records: {duplicate_count} records",
                count=duplicate_count,
                percentage=duplicate_count / len(df) * 100
            ))

    def _validate_missing_data(
        self,
        df: pd.DataFrame,
        data_source: DataSource,
        errors: List[ValidationError],
        warnings: List[ValidationError]
    ) -> None:
        """Validate for missing data patterns."""
        if len(df) == 0:
            return

        # Calculate missing data ratios
        missing_ratios = df.isnull().sum() / len(df)
        problematic_columns = []

        for column, missing_ratio in missing_ratios.items():
            if missing_ratio > self.max_missing_ratio:
                problematic_columns.append((column, missing_ratio))

        if problematic_columns:
            for column, missing_ratio in problematic_columns:
                errors.append(ValidationError(
                    rule=ValidationRule.MISSING_DATA,
                    level=ValidationLevel.ERROR,
                    message=f"High missing data in {column}: {missing_ratio:.2%}",
                    field=column,
                    count=int(missing_ratio * len(df)),
                    percentage=missing_ratio * 100
                ))

        # Check for consecutive missing data (time series specific)
        timestamp_col = self._find_timestamp_column(df)
        if timestamp_col and timestamp_col in df.columns and len(df) > 1:
            df_sorted = df.sort_values(timestamp_col)

            # Expected time intervals based on data source
            expected_intervals = {
                DataSource.KLINES: "1m",  # Default to 1 minute
                DataSource.MARK_PRICE: 1800,  # 30 minutes
                DataSource.FUNDING_RATE: 28800,  # 8 hours
            }

            if data_source in expected_intervals:
                expected_interval = pd.Timedelta(seconds=expected_intervals[data_source])
                time_diffs = df_sorted[timestamp_col].diff().dropna()

                # Find gaps larger than expected
                large_gaps = time_diffs > expected_interval * 2  # More than 2x expected
                gap_count = large_gaps.sum()

                if gap_count > 0:
                    warnings.append(ValidationError(
                        rule=ValidationRule.MISSING_DATA,
                        level=ValidationLevel.WARNING,
                        message=f"Time gaps detected: {gap_count} gaps",
                        field=timestamp_col,
                        count=gap_count,
                        percentage=gap_count / len(df) * 100
                    ))

    def _validate_outliers(
        self,
        df: pd.DataFrame,
        data_source: DataSource,
        errors: List[ValidationError],
        warnings: List[ValidationError]
    ) -> None:
        """Validate for statistical outliers."""
        if len(df) == 0:
            return

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}

        for column in numeric_columns:
            if column not in df.columns:
                continue

            series = df[column].dropna()
            if len(series) < 10:  # Need minimum data for outlier detection
                continue

            # Z-score outlier detection
            z_scores = np.abs((series - series.mean()) / series.std())
            outliers = z_scores > self.outlier_threshold
            outlier_count = outliers.sum()

            if outlier_count > 0:
                outlier_counts[column] = outlier_count

        # Report outliers
        for column, outlier_count in outlier_counts.items():
            severity = ValidationLevel.WARNING if not self.strict_mode else ValidationLevel.ERROR
            errors.append(ValidationError(
                rule=ValidationRule.OUTLIER_DETECTION,
                level=severity,
                message=f"Outliers detected in {column}: {outlier_count} records",
                field=column,
                count=outlier_count,
                percentage=outlier_count / len(df) * 100
            ))

    def _validate_consistency(
        self,
        df: pd.DataFrame,
        data_source: DataSource,
        errors: List[ValidationError],
        warnings: List[ValidationError]
    ) -> None:
        """Validate data consistency and business logic."""
        if len(df) == 0:
            return

        # OHLC consistency checks
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # High should be >= Open, High, Low, Close
            high_violations = df['high'] < df[['open', 'high', 'low', 'close']].max(axis=1)
            high_violation_count = high_violations.sum()

            # Low should be <= Open, High, Low, Close
            low_violations = df['low'] > df[['open', 'high', 'low', 'close']].min(axis=1)
            low_violation_count = low_violations.sum()

            # Negative values for OHLC
            negative_violations = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()

            if high_violation_count > 0:
                errors.append(ValidationError(
                    rule=ValidationRule.CONSISTENCY_CHECKS,
                    level=ValidationLevel.ERROR,
                    message=f"High price violations: {high_violation_count} records",
                    count=high_violation_count,
                    percentage=high_violation_count / len(df) * 100
                ))

            if low_violation_count > 0:
                errors.append(ValidationError(
                    rule=ValidationRule.CONSISTENCY_CHECKS,
                    level=ValidationLevel.ERROR,
                    message=f"Low price violations: {low_violation_count} records",
                    count=low_violation_count,
                    percentage=low_violation_count / len(df) * 100
                ))

            if negative_violations > 0:
                errors.append(ValidationError(
                    rule=ValidationRule.CONSISTENCY_CHECKS,
                    level=ValidationLevel.ERROR,
                    message=f"Negative OHLC values: {negative_violations} records",
                    count=negative_violations,
                    percentage=negative_violations / len(df) * 100
                ))

        # Volume consistency
        if 'volume' in df.columns:
            negative_volume = (df['volume'] < 0).sum()
            if negative_volume > 0:
                errors.append(ValidationError(
                    rule=ValidationRule.CONSISTENCY_CHECKS,
                    level=ValidationLevel.ERROR,
                    message=f"Negative volume values: {negative_volume} records",
                    field='volume',
                    count=negative_volume,
                    percentage=negative_volume / len(df) * 100
                ))

        # Price consistency for trade data
        if data_source == DataSource.AGG_TRADES and 'price' in df.columns:
            # Prices should be positive
            negative_prices = (df['price'] <= 0).sum()
            if negative_prices > 0:
                errors.append(ValidationError(
                    rule=ValidationRule.CONSISTENCY_CHECKS,
                    level=ValidationLevel.ERROR,
                    message=f"Invalid trade prices: {negative_prices} records",
                    field='price',
                    count=negative_prices,
                    percentage=negative_prices / len(df) * 100
                ))

    def _validate_business_rules(
        self,
        df: pd.DataFrame,
        data_source: DataSource,
        errors: List[ValidationError],
        warnings: List[ValidationError],
        context: Optional[Dict[str, Any]]
    ) -> None:
        """Validate business-specific rules."""
        rules = self.business_rules.get(data_source, [])

        for rule_name, rule_config in rules.items():
            if rule_name == 'min_price_movement' and all(col in df.columns for col in ['open', 'close']):
                # Check for minimum price movements
                price_change = abs(df['close'] - df['open'])
                min_movement = rule_config.get('min_movement', 0.00001)
                no_movement = (price_change < min_movement).sum()

                if no_movement > len(df) * 0.95:  # 95% have no movement
                    warnings.append(ValidationError(
                        rule=ValidationRule.BUSINESS_RULES,
                        level=ValidationLevel.WARNING,
                        message=f"Very low price movement: {no_movement} records",
                        count=no_movement,
                        percentage=no_movement / len(df) * 100
                    ))

            elif rule_name == 'max_price_change' and all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                # Check for extreme price changes
                max_change_pct = rule_config.get('max_change_pct', 50.0)
                high_change = ((df['high'] - df['open']) / df['open'] * 100).abs()
                low_change = ((df['low'] - df['open']) / df['open'] * 100).abs()

                extreme_changes = ((high_change > max_change_pct) | (low_change > max_change_pct)).sum()
                if extreme_changes > 0:
                    warnings.append(ValidationError(
                        rule=ValidationRule.BUSINESS_RULES,
                        level=ValidationLevel.WARNING,
                        message=f"Extreme price changes: {extreme_changes} records",
                        count=extreme_changes,
                        percentage=extreme_changes / len(df) * 100
                    ))

    def _validate_timestamp_format(
        self,
        df: pd.DataFrame,
        data_source: DataSource,
        errors: List[ValidationError],
        warnings: List[ValidationError]
    ) -> None:
        """Validate timestamp formats and timezones."""
        timestamp_columns = [col for col in df.columns if 'time' in col.lower()]

        for col in timestamp_columns:
            if col not in df.columns:
                continue

            try:
                # Try to convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')

                # Check for NaT values
                nat_count = df[col].isna().sum()
                if nat_count > 0:
                    warnings.append(ValidationError(
                        rule=ValidationRule.TIMESTAMP_FORMAT,
                        level=ValidationLevel.WARNING,
                        message=f"Invalid timestamp format in {col}: {nat_count} records",
                        field=col,
                        count=nat_count,
                        percentage=nat_count / len(df) * 100
                    ))

                # Check timezone
                if hasattr(df[col].dtype, 'tz') and df[col].dt.tz is None:
                    warnings.append(ValidationError(
                        rule=ValidationRule.TIMESTAMP_FORMAT,
                        level=ValidationLevel.INFO,
                        message=f"No timezone specified for {col}",
                        field=col
                    ))

            except Exception as e:
                errors.append(ValidationError(
                    rule=ValidationRule.TIMESTAMP_FORMAT,
                    level=ValidationLevel.ERROR,
                    message=f"Timestamp format error in {col}: {str(e)}",
                    field=col,
                    count=len(df),
                    percentage=100.0
                ))

    def _validate_data_freshness(
        self,
        df: pd.DataFrame,
        data_source: DataSource,
        errors: List[ValidationError],
        warnings: List[ValidationError]
    ) -> None:
        """Validate data freshness."""
        timestamp_col = self._find_timestamp_column(df)
        if not timestamp_col or timestamp_col not in df.columns:
            return

        try:
            # Get latest timestamp
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)

            latest_timestamp = df[timestamp_col].max()
            if pd.isna(latest_timestamp):
                errors.append(ValidationError(
                    rule=ValidationRule.DATA_FRESHNESS,
                    level=ValidationLevel.ERROR,
                    message="No valid timestamps in data",
                    count=len(df),
                    percentage=100.0
                ))
                return

            # Check freshness
            now = datetime.now(timezone.utc)
            age = now - latest_timestamp

            if age > self.freshness_threshold:
                warnings.append(ValidationError(
                    rule=ValidationRule.DATA_FRESHNESS,
                    level=ValidationLevel.WARNING,
                    message=f"Data is stale: {age}",
                    field=timestamp_col,
                    value=latest_timestamp,
                    expected=f"Within {self.freshness_threshold}",
                    count=len(df),
                    percentage=100.0
                ))

        except Exception as e:
            errors.append(ValidationError(
                rule=ValidationRule.DATA_FRESHNESS,
                level=ValidationLevel.ERROR,
                message=f"Freshness validation failed: {str(e)}",
                count=len(df),
                percentage=100.0
            ))

    def _calculate_validation_quality_score(
        self,
        total_records: int,
        valid_records: int,
        total_errors: int,
        total_warnings: int
    ) -> float:
        """Calculate overall validation quality score."""
        if total_records == 0:
            return 0.0

        # Base score from valid records percentage
        valid_percentage = valid_records / total_records
        base_score = valid_percentage * 100

        # Deductions for errors and warnings
        error_penalty = (total_errors / total_records) * 50  # 50% weight for errors
        warning_penalty = (total_warnings / total_records) * 10  # 10% weight for warnings

        final_score = max(0, base_score - error_penalty - warning_penalty)
        return min(100, final_score)

    def _log_validation_result(self, result: ValidationResult) -> None:
        """Log validation result with structured information."""
        # Create validation log entry
        validation_log = {
            'validation_time': result.validation_time.isoformat(),
            'data_source': result.data_source.value,
            'symbol': result.symbol,
            'timeframe': result.timeframe,
            'total_records': result.total_records,
            'valid_records': result.valid_records,
            'quality_score': result.quality_score,
            'is_valid': result.is_valid,
            'error_count': len(result.errors),
            'warning_count': len(result.warnings),
            'critical_issues': result.critical_issues,
            'fixable_issues': result.fixable_issues
        }

        # Log validation result
        if result.is_valid:
            self.logger.info(
                f"Validation passed for {result.data_source.value} {result.symbol}: "
                f"Score: {result.quality_score:.1f}, Records: {result.valid_records}/{result.total_records}"
            )
        else:
            self.logger.error(
                f"Validation failed for {result.data_source.value} {result.symbol}: "
                f"Score: {result.quality_score:.1f}, Errors: {len(result.errors)}, Warnings: {len(result.warnings)}"
            )

            # Log critical issues separately
            for issue in result.critical_issues:
                self.logger.error(f"CRITICAL: {issue}")

        # Log data quality
        log_data_quality(
            data_source=f"{result.data_source.value}_{result.symbol}_{result.timeframe or ''}",
            record_count=result.total_records,
            quality_score=result.quality_score,
            issues=[error.message for error in result.errors]
        )

    def _find_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the primary timestamp column in DataFrame."""
        timestamp_candidates = ['timestamp', 'open_time', 'close_time', 'time', 'event_time', 'T', 'E']
        for candidate in timestamp_candidates:
            if candidate in df.columns:
                return candidate
        return None

    def _get_unique_columns(self, data_source: DataSource) -> List[str]:
        """Get columns that should be unique for each data source."""
        unique_columns_map = {
            DataSource.KLINES: ['timestamp'],
            DataSource.AGG_TRADES: ['agg_trade_id'],
            DataSource.TRADES: ['id', 'time'],
            DataSource.DEPTH: ['timestamp'],
            DataSource.MARK_PRICE: ['timestamp'],
            DataSource.FUNDING_RATE: ['timestamp'],
            DataSource.OPEN_INTEREST: ['timestamp'],
            DataSource.BOOK_TICKER: [],  # Real-time, no uniqueness required
        }

        return unique_columns_map.get(data_source, ['timestamp'])

    def _get_required_fields_config(self) -> Dict[DataSource, List[str]]:
        """Get required fields configuration for each data source."""
        return {
            DataSource.KLINES: ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time'],
            DataSource.AGG_TRADES: ['agg_trade_id', 'price', 'qty', 'first_id', 'last_id', 'time', 'is_buyer_maker'],
            DataSource.TRADES: ['id', 'price', 'qty', 'time', 'is_buyer_maker'],
            DataSource.DEPTH: ['bids', 'asks', 'timestamp'],
            DataSource.MARK_PRICE: ['markPrice', 'indexPrice', 'timestamp', 'fundingRate'],
            DataSource.FUNDING_RATE: ['fundingRate', 'timestamp'],
            DataSource.OPEN_INTEREST: ['openInterest', 'timestamp'],
            DataSource.BOOK_TICKER: ['symbol', 'bidPrice', 'bidQty', 'askPrice', 'askQty']
        }

    def _get_value_ranges_config(self) -> Dict[DataSource, Dict[str, Tuple[float, float]]]:
        """Get value ranges configuration for each data source."""
        return {
            DataSource.KLINES: {
                'open': (0, float('inf')),
                'high': (0, float('inf')),
                'low': (0, float('inf')),
                'close': (0, float('inf')),
                'volume': (0, float('inf')),
            },
            DataSource.AGG_TRADES: {
                'price': (0, float('inf')),
                'qty': (0, float('inf')),
            },
            DataSource.TRADES: {
                'price': (0, float('inf')),
                'qty': (0, float('inf')),
            },
            DataSource.MARK_PRICE: {
                'markPrice': (0, float('inf')),
                'indexPrice': (0, float('inf')),
                'fundingRate': (-1.0, 1.0),
            },
            DataSource.FUNDING_RATE: {
                'fundingRate': (-1.0, 1.0),
            },
            DataSource.OPEN_INTEREST: {
                'openInterest': (0, float('inf')),
            }
        }

    def _get_business_rules_config(self) -> Dict[DataSource, Dict[str, Any]]:
        """Get business rules configuration for each data source."""
        return {
            DataSource.KLINES: {
                'min_price_movement': {'min_movement': 0.00001},
                'max_price_change': {'max_change_pct': 50.0},
            },
            DataSource.AGG_TRADES: {
                'max_trade_size': {'max_size': 10000000},  # Max trade size in base currency
            },
            DataSource.TRADES: {
                'max_trade_size': {'max_size': 1000000},
            }
        }