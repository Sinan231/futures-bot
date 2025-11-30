"""
Time-based and cyclical feature engineering for trading signals.

Creates temporal features (hour, day-of-week, month), market session
indicators, cyclical transformations using sin/cos functions,
and market hours analysis to capture time-based patterns in market data.
"""

import math
import warnings
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..utils.logging import get_logger, performance_monitor
from ..utils.helpers import parse_timeframe_to_seconds, normalize_timestamp
from ..utils.config import DataConfig


class TimeBasedFeatures:
    """Time-based feature engineering for market data."""

    def __init__(self, config: DataConfig):
        """Initialize time-based features.

        Args:
            config: Data configuration
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Feature configuration
        self.enabled_features = set([
            'hour_of_day', 'day_of_week', 'month_of_year', 'quarter_of_year',
            'year', 'week_of_year', 'day_of_month', 'days_in_month',
            'is_weekend', 'is_holiday', 'is_trading_session',
            'market_session', 'seconds_from_open', 'minutes_from_open',
            'seconds_to_close', 'days_to_earnings', 'is_month_end'
        ])

    @performance_monitor("time_features_engineering")
    def create_time_features(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """Create comprehensive time-based features.

        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column

        Returns:
            DataFrame with time features added
        """
        if len(df) == 0:
            return df.copy()

        # Validate timestamp column
        if timestamp_col not in df.columns:
            self.logger.warning(f"Timestamp column '{timestamp_col}' not found in DataFrame")
            return df.copy()

        # Ensure timestamp is datetime
        df_normalized = self._normalize_timestamps(df, timestamp_col)

        feature_df = df_normalized.copy()

        # Add features if enabled
        if 'hour_of_day' in self.enabled_features:
            feature_df = self._add_hour_of_day(feature_df, timestamp_col)
        if 'day_of_week' in self.enabled_features:
            feature_df = self._add_day_of_week(feature_df, timestamp_col)
        if 'month_of_year' in self.enabled_features:
            feature_df = self._add_month_of_year(feature_df, timestamp_col)
        if 'quarter_of_year' in self.enabled_features:
            feature_df = self._add_quarter_of_year(feature_df, timestamp_col)
        if 'year' in self.enabled_features:
            feature_df = self._add_year(feature_df, timestamp_col)
        if 'week_of_year' in self.enabled_features:
            feature_df = self._add_week_of_year(feature_df, timestamp_col)
        if 'day_of_month' in self.enabled_features:
            feature_df = self._add_day_of_month(feature_df, timestamp_col)
        if 'days_in_month' in self.enabled_features:
            feature_df = self._add_days_in_month(feature_df, timestamp_col)
        if 'is_weekend' in self.enabled_features:
            feature_df = self._add_is_weekend(feature_df, timestamp_col)
        if 'is_holiday' in self.enabled_features:
            feature_df = self._add_is_holiday(feature_df, timestamp_col)
        if 'is_trading_session' in self.enabled_features:
            feature_df = self._add_trading_session_indicator(feature_df, timestamp_col)
        if 'market_session' in self.enabled_features:
            feature_df = self._add_market_session(feature_df, timestamp_col)
        if 'seconds_from_open' in self.enabled_features:
            feature_df = self._add_seconds_from_market_open(feature_df, timestamp_col)
        if 'minutes_from_open' in self.enabled_features:
            feature_df = self._add_minutes_from_market_open(feature_df, timestamp_col)
        if 'seconds_to_close' in self.enabled_features:
            feature_df = self._add_seconds_to_market_close(feature_df, timestamp_col)
        if 'days_to_earnings' in self.enabled_features:
            feature_df = self._add_days_to_earnings(feature_df, timestamp_col)
        if 'is_month_end' in self.enabled_features:
            feature_df = self._add_is_month_end(feature_df, timestamp_col)

        self.logger.info(f"Created {len([col for col in feature_df.columns if col not in df.columns])} time-based features")
        return feature_df

    def _normalize_timestamps(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Normalize timestamp column to datetime with UTC timezone."""
        df_normalized = df.copy()

        if not pd.api.types.is_datetime64_any_dtype(df_normalized[timestamp_col]):
            try:
                df_normalized[timestamp_col] = pd.to_datetime(df_normalized[timestamp_col], utc=True, errors='coerce')
            except Exception as e:
                self.logger.warning(f"Failed to convert timestamp: {e}")
                df_normalized[timestamp_col] = pd.to_datetime(df_normalized[timestamp_col], unit='ms', utc=True, errors='coerce')

        return df_normalized

    def _add_hour_of_day(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add hour of day (0-23) feature."""
        try:
            df['hour_of_day'] = df[timestamp_col].dt.hour
        except Exception as e:
            self.logger.error(f"Failed to add hour_of_day: {e}")
            df['hour_of_day'] = 0

        return df

    def _add_day_of_week(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add day of week (0-6, Monday=0) feature."""
        try:
            df['day_of_week'] = df[timestamp_col].dt.dayofweek
        except Exception as e:
            self.logger.error(f"Failed to add day_of_week: {e}")
            df['day_of_week'] = 0

        return df

    def _add_month_of_year(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add month of year (1-12) feature."""
        try:
            df['month_of_year'] = df[timestamp_col].dt.month
        except Exception as e:
            self.logger.error(f"Failed to add month_of_year: {e}")
            df['month_of_year'] = 0

        return df

    def _add_quarter_of_year(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add quarter of year (1-4) feature."""
        try:
            df['quarter_of_year'] = ((df[timestamp_col].dt.month - 1) // 3) + 1
        except Exception as e:
            self.logger.error(f"Failed to add quarter_of_year: {e}")
            df['quarter_of_year'] = 1

        return df

    def _add_year(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add year feature."""
        try:
            df['year'] = df[timestamp_col].dt.year
        except Exception as e:
            self.logger.error(f"Failed to add year: {e}")
            df['year'] = 2000

        return df

    def _add_week_of_year(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add week of year (1-52) feature."""
        try:
            df['week_of_year'] = df[timestamp_col].dt.isocalendar().week
        except Exception as e:
            self.logger.error(f"Failed to add week_of_year: {e}")
            df['week_of_year'] = 1

        return df

    def _add_day_of_month(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add day of month (1-31) feature."""
        try:
            df['day_of_month'] = df[timestamp_col].dt.day
        except Exception as e:
            self.logger.error(f"Failed to add day_of_month: {e}")
            df['day_of_month'] = 1

        return df

    def _add_days_in_month(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add days in month (1-31) feature."""
        try:
            df['days_in_month'] = df[timestamp_col].dt.days_in_month
        except Exception as e:
            self.logger.error(f"Failed to add days_in_month: {e}")
            df['days_in_month'] = 30

        return df

    def _add_is_weekend(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add is_weekend feature (True for Saturday/Sunday)."""
        try:
            df['is_weekend'] = df[timestamp_col].dt.dayofweek >= 5  # 5=Saturday, 6=Sunday
        except Exception as e:
            self.logger.error(f"Failed to add is_weekend: {e}")
            df['is_weekend'] = False

        return df

    def _add_is_holiday(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add is_holiday feature based on predefined holidays."""
        # Define major market holidays (simplified - could be expanded)
        holidays = [
            (1, 1),   # New Year's Day
            (1, 7),   # Orthodox Christmas
            (12, 25),  # Christmas Day
            (7, 4),    # Independence Day (US)
            (9, 2),    # Labor Day (US)
            (11, 23),  # Thanksgiving (US)
            (12, 24),  # Christmas Eve
            (12, 31),  # New Year's Eve
        ]

        try:
            df['is_holiday'] = df[timestamp_col].apply(
                lambda x: self._is_market_holiday(x, holidays)
            )
        except Exception as e:
            self.logger.error(f"Failed to add is_holiday: {e}")
            df['is_holiday'] = False

        return df

    def _is_market_holiday(self, timestamp: datetime, holidays: List[Tuple[int, int]]) -> bool:
        """Check if timestamp falls on a market holiday."""
        if not timestamp:
            return False

        month_day = (timestamp.month, timestamp.day)
        return month_day in holidays

    def _add_trading_session_indicator(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add trading session indicator based on market hours."""
        try:
            df['is_trading_session'] = self._get_trading_session(df[timestamp_col])
        except Exception as e:
            self.logger.error(f"Failed to add trading_session: {e}")
            df['is_trading_session'] = True

        return df

    def _get_trading_session(self, timestamp: pd.Timestamp) -> bool:
        """Determine if timestamp is during trading hours."""
        # Simplified trading session logic (24/7 crypto markets are always active)
        # This could be expanded for session-specific patterns
        hour = timestamp.hour
        return True  # For crypto, market is always open

    def _add_market_session(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add market session identification."""
        try:
            df['market_session'] = df[timestamp_col].apply(self._get_market_session_name)
        except Exception as e:
            self.logger.error(f"Failed to add market_session: {e}")
            df['market_session'] = 'continuous'

        return df

    def _get_market_session_name(self, timestamp: pd.Timestamp) -> str:
        """Get market session name based on hour."""
        hour = timestamp.hour

        if hour >= 0 and hour < 8:
            return 'asian'
        elif hour >= 8 and hour < 12:
            return 'european'
        elif hour >= 12 and hour < 16:
            return 'american'
        elif hour >= 16 and hour < 20:
            return 'european'
        elif hour >= 20 and hour < 24:
            return 'asian'
        else:
            return 'unknown'

    def _add_seconds_from_market_open(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add seconds since market open (for session analysis)."""
        try:
            # Simplified: market opens at 00:00:00 UTC
            market_open = df[timestamp_col].dt.normalize()
            df['seconds_from_open'] = (market_open - market_open).total_seconds()
        except Exception as e:
            self.logger.error(f"Failed to add seconds_from_open: {e}")
            df['seconds_from_open'] = 0

        return df

    def _add_minutes_from_market_open(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add minutes since market open."""
        try:
            df['minutes_from_open'] = df['seconds_from_open'] / 60
        except Exception as e:
            self.logger.error(f"Failed to add minutes_from_open: {e}")
            df['minutes_from_open'] = 0

        return df

    def _add_seconds_to_market_close(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add seconds until market close (for session analysis)."""
        try:
            # Simplified: market never closes (24/7 crypto)
            df['seconds_to_close'] = 86400 - df['seconds_from_open']  # 24 hours in seconds
        except Exception as e:
            self.logger.error(f"Failed to add seconds_to_close: {e}")
            df['seconds_to_close'] = 43200

        return df

    def _add_minutes_to_market_close(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add minutes until market close."""
        try:
            df['minutes_to_close'] = df['seconds_to_close'] / 60
        except Exception as e:
            self.logger.error(f"Failed to add minutes_to_close: {e}")
            df['minutes_to_close'] = 720

        return df

    def _add_days_to_earnings(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add days until end of month/quarter."""
        try:
            df['days_to_earnings'] = (31 - df[timestamp_col].dt.day) % 31  # Days until month end
        except Exception as e:
            self.logger.error(f"Failed to add days_to_earnings: {e}")
            df['days_to_earnings'] = 15

        return df

    def _add_is_month_end(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add is_month_end feature."""
        try:
            df['is_month_end'] = (df[timestamp_col].dt.day >= 28)  # Approximate month end
        except Exception as e:
            self.logger.error(f"Failed to add is_month_end: {e}")
            df['is_month_end'] = False

        return df


class CyclicalFeatures:
    """Cyclical feature engineering using sin/cos transformations."""

    def __init__(self, config: DataConfig):
        """Initialize cyclical features.

        Args:
            config: Data configuration
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Feature configuration
        self.enabled_features = set([
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos',
            'day_of_year_sin', 'day_of_year_cos',
            'month_progress_sin', 'month_progress_cos'
        ])

    @performance_monitor("cyclical_features_engineering")
    def create_cyclical_features(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """Create cyclical features using sin/cos transformations.

        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column

        Returns:
            DataFrame with cyclical features added
        """
        if len(df) == 0:
            return df.copy()

        # Validate timestamp column
        if timestamp_col not in df.columns:
            self.logger.warning(f"Timestamp column '{timestamp_col}' not found in DataFrame")
            return df.copy()

        # Ensure timestamp is datetime
        df_normalized = self._normalize_timestamps(df, timestamp_col)

        feature_df = df_normalized.copy()

        # Add cyclical features if enabled
        if 'hour_sin' in self.enabled_features:
            feature_df = self._add_hour_sin_cos(feature_df, timestamp_col, 'sin')
        if 'hour_cos' in self.enabled_features:
            feature_df = self._add_hour_sin_cos(feature_df, timestamp_col, 'cos')
        if 'day_sin' in self.enabled_features:
            feature_df = self._add_day_sin_cos(feature_df, timestamp_col, 'sin')
        if 'day_cos' in self.enabled_features:
            feature_df = self._add_day_sin_cos(feature_df, timestamp_col, 'cos')
        if 'month_sin' in self.enabled_features:
            feature_df = self._add_month_sin_cos(feature_df, timestamp_col, 'sin')
        if 'month_cos' in self.enabled_features:
            feature_df = self._add_month_sin_cos(feature_df, timestamp_col, 'cos')
        if 'quarter_sin' in self.enabled_features:
            feature_df = self._add_quarter_sin_cos(feature_df, timestamp_col, 'sin')
        if 'quarter_cos' in self.enabled_features:
            feature_df = self._add_quarter_sin_cos(feature_df, timestamp_col, 'cos')
        if 'day_of_year_sin' in self.enabled_features:
            feature_df = self._add_day_of_year_sin_cos(feature_df, timestamp_col, 'sin')
        if 'day_of_year_cos' in self.enabled_features:
            feature_df = self._add_day_of_year_sin_cos(feature_df, timestamp_col, 'cos')
        if 'month_progress_sin' in self.enabled_features:
            feature_df = self._add_month_progress_sin_cos(feature_df, timestamp_col, 'sin')
        if 'month_progress_cos' in self.enabled_features:
            feature_df = self._add_month_progress_sin_cos(feature_df, timestamp_col, 'cos')

        self.logger.info(f"Created {len([col for col in feature_df.columns if col not in df.columns])} cyclical features")
        return feature_df

    def _normalize_timestamps(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Normalize timestamp column to datetime with UTC timezone."""
        df_normalized = df.copy()

        if not pd.api.types.is_datetime64_any_dtype(df_normalized[timestamp_col]):
            try:
                df_normalized[timestamp_col] = pd.to_datetime(df_normalized[timestamp_col], utc=True, errors='coerce')
            except Exception as e:
                self.logger.warning(f"Failed to convert timestamp: {e}")
                df_normalized[timestamp_col] = pd.to_datetime(df_normalized[timestamp_col], unit='ms', utc=True, errors='coerce')

        return df_normalized

    def _add_hour_sin_cos(self, df: pd.DataFrame, timestamp_col: str, func_type: str) -> pd.DataFrame:
        """Add hour cyclical features (sin/cos)."""
        try:
            hour = df[timestamp_col].dt.hour
            if func_type == 'sin':
                df[f'hour_{func_type}'] = np.sin(2 * np.pi * hour / 24)
            else:  # cos
                df[f'hour_{func_type}'] = np.cos(2 * np.pi * hour / 24)
        except Exception as e:
            self.logger.error(f"Failed to add hour_{func_type}: {e}")

        return df

    def _add_day_sin_cos(self, df: pd.DataFrame, timestamp_col: str, func_type: str) -> pd.DataFrame:
        """Add day cyclical features (sin/cos)."""
        try:
            day_of_year = df[timestamp_col].dt.dayofyear
            if func_type == 'sin':
                df[f'day_{func_type}'] = np.sin(2 * np.pi * day_of_year / 365)
            else:  # cos
                df[f'day_{func_type}'] = np.cos(2 * np.pi * day_of_year / 365)
        except Exception as e:
            self.logger.error(f"Failed to add day_{func_type}: {e}")

        return df

    def _add_month_sin_cos(self, df: pd.DataFrame, timestamp_col: str, func_type: str) -> pd.DataFrame:
        """Add month cyclical features (sin/cos)."""
        try:
            month = df[timestamp_col].dt.month
            if func_type == 'sin':
                df[f'month_{func_type}'] = np.sin(2 * np.pi * month / 12)
            else:  # cos
                df[f'month_{func_type}'] = np.cos(2 * np.pi * month / 12)
        except Exception as e:
            self.logger.error(f"Failed to add month_{func_type}: {e}")

        return df

    def _add_quarter_sin_cos(self, df: pd.DataFrame, timestamp_col: str, func_type: str) -> pd.DataFrame:
        """Add quarter cyclical features (sin/cos)."""
        try:
            quarter = df[timestamp_col].dt.quarter
            if func_type == 'sin':
                df[f'quarter_{func_type}'] = np.sin(2 * np.pi * quarter / 4)
            else:  # cos
                df[f'quarter_{func_type}'] = np.cos(2 * np.pi * quarter / 4)
        except Exception as e:
            self.logger.error(f"Failed to add quarter_{func_type}: {e}")

        return df

    def _add_day_of_year_sin_cos(self, df: pd.DataFrame, timestamp_col: str, func_type: str) -> pd.DataFrame:
        """Add day of year cyclical features (sin/cos)."""
        try:
            day_of_year = df[timestamp_col].dt.dayofyear
            if func_type == 'sin':
                df[f'day_of_year_{func_type}'] = np.sin(2 * np.pi * day_of_year / 365)
            else:  # cos
                df[f'day_of_year_{func_type}'] = np.cos(2 * np.pi * day_of_year / 365)
        except Exception as e:
            self.logger.error(f"Failed to add day_of_year_{func_type}: {e}")

        return df

    def _add_month_progress_sin_cos(self, df: pd.DataFrame, timestamp_col: str, func_type: str) -> pd.DataFrame:
        """Add month progress cyclical features (sin/cos)."""
        try:
            day = df[timestamp_col].dt.day
            days_in_month = df[timestamp_col].dt.days_in_month
            month_progress = (day - 1) / days_in_month
            if func_type == 'sin':
                df[f'month_progress_{func_type}'] = np.sin(2 * np.pi * month_progress)
            else:  # cos
                df[f'month_progress_{func_type}'] = np.cos(2 * np.pi * month_progress)
        except Exception as e:
            self.logger.error(f"Failed to add month_progress_{func_type}: {e}")

        return df


class SessionFeatures:
    """Market session feature engineering for different trading sessions."""

    def __init__(self, config: DataConfig):
        """Initialize session features.

        Args:
            config: Data configuration
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Feature configuration
        self.enabled_features = set([
            'is_asian_session', 'is_european_session', 'is_us_session',
            'is_london_session', 'is_tokyo_session',
            'session_overlap', 'session_transition',
            'time_to_session_change', 'session_duration_hours'
        ])

        # Session definitions
        self.sessions = {
            'asian': {'start': 0, 'end': 8, 'timezone': 'Asia/Tokyo'},
            'european': {'start': 7, 'end': 16, 'timezone': 'Europe/London'},
            'us': {'start': 13, 'end': 22, 'timezone': 'America/New_York'},
            'london': {'start': 8, 'end': 16, 'timezone': 'Europe/London'},  # London session overlaps with European
            'tokyo': {'start': 23, 'end': 7, 'timezone': 'Asia/Tokyo'},  # Tokyo session overlaps with Asian
        }

    @performance_monitor("session_features_engineering")
    def create_session_features(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """Create market session features.

        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column

        Returns:
            DataFrame with session features added
        """
        if len(df) == 0:
            return df.copy()

        # Validate timestamp column
        if timestamp_col not in df.columns:
            self.logger.warning(f"Timestamp column '{timestamp_col}' not found in DataFrame")
            return df.copy()

        # Ensure timestamp is datetime
        df_normalized = self._normalize_timestamps(df, timestamp_col)

        feature_df = df_normalized.copy()

        # Add session features if enabled
        if 'is_asian_session' in self.enabled_features:
            feature_df = self._add_session_indicator(feature_df, timestamp_col, 'asian')
        if 'is_european_session' in self.enabled_features:
            feature_df = self._add_session_indicator(feature_df, timestamp_col, 'european')
        if 'is_us_session' in self.enabled_features:
            feature_df = self._add_session_indicator(feature_df, timestamp_col, 'us')
        if 'is_london_session' in self.enabled_features:
            feature_df = self._add_session_indicator(feature_df, timestamp_col, 'london')
        if 'is_tokyo_session' in self.enabled_features:
            feature_df = self._add_session_indicator(feature_df, timestamp_col, 'tokyo')
        if 'session_overlap' in self.enabled_features:
            feature_df = self._add_session_overlap(feature_df, timestamp_col)
        if 'session_transition' in self.enabled_features:
            feature_df = self._add_session_transition(feature_df, timestamp_col)
        if 'time_to_session_change' in self.enabled_features:
            feature_df = self._add_time_to_session_change(feature_df, timestamp_col)
        if 'session_duration_hours' in self.enabled_features:
            feature_df = self._add_session_duration_hours(feature_df, timestamp_col)

        self.logger.info(f"Created {len([col for col in feature_df.columns if col not in df.columns])} session features")
        return feature_df

    def _normalize_timestamps(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Normalize timestamp column to datetime with UTC timezone."""
        df_normalized = df.copy()

        if not pd.api.types.is_datetime64_any_dtype(df_normalized[timestamp_col]):
            try:
                df_normalized[timestamp_col] = pd.to_datetime(df_normalized[timestamp_col], utc=True, errors='coerce')
            except Exception as e:
                self.logger.warning(f"Failed to convert timestamp: {e}")
                df_normalized[timestamp_col] = pd.to_datetime(df_normalized[timestamp_col], unit='ms', utc=True, errors='coerce')

        return df_normalized

    def _add_session_indicator(self, df: pd.DataFrame, timestamp_col: str, session_name: str) -> pd.DataFrame:
        """Add session indicator feature."""
        try:
            df[f'is_{session_name}_session'] = self._is_in_session(df[timestamp_col], session_name)
        except Exception as e:
            self.logger.error(f"Failed to add session indicator for {session_name}: {e}")
            df[f'is_{session_name}_session'] = False

        return df

    def _is_in_session(self, timestamp: pd.Timestamp, session_name: str) -> bool:
        """Check if timestamp is in specified session."""
        if session_name not in self.sessions:
            return False

        session = self.sessions[session_name]
        hour = timestamp.hour

        # Handle timezone-aware timestamps
        if timestamp.tzinfo is not None:
            hour = timestamp.hour

        # Check if hour is within session range
        if session['start'] <= session['end']:
            # Normal session (e.g., 0-8)
            if session['start'] <= session['end']:
                return session['start'] <= hour < session['end']
            # Cross-midnight session (e.g., 23-7)
            else:
                return (hour >= session['start']) or (hour < session['end'])
        else:
            return False

    def _add_session_overlap(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add session overlap indicator."""
        try:
            df['session_overlap_count'] = df.apply(
                lambda x: self._count_overlapping_sessions(x[timestamp_col]),
                axis=1
            )
        except Exception as e:
            self.logger.error(f"Failed to add session overlap: {e}")
            df['session_overlap_count'] = 0

        return df

    def _count_overlapping_sessions(self, timestamp: pd.Timestamp) -> int:
        """Count overlapping sessions at a given timestamp."""
        overlap_count = 0

        for session_name, session in self.sessions.items():
            if self._is_in_session(timestamp, session_name):
                overlap_count += 1

        return overlap_count

    def _add_session_transition(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add session transition indicator."""
        try:
            # Sort by timestamp and get previous session
            df_sorted = df.sort_values(timestamp_col)
            df[f'session_transition'] = df_sorted.apply(
                lambda row: self._get_session_transition(row, row.name, timestamp_col),
                axis=1
            )
        except Exception as e:
            self.logger.error(f"Failed to add session transition: {e}")
            df[f'session_transition'] = 0

        return df

    def _get_session_transition(self, current_row: pd.Series, current_index: int, timestamp_col: str) -> str:
        """Get session transition type."""
        if current_index == 0:
            return 'session_start'

        previous_row = df.iloc[current_index - 1]
        previous_session = self._get_active_session(previous_row[timestamp_col])
        current_session = self._get_active_session(current_row[timestamp_col])

        if previous_session is None and current_session is not None:
            return 'session_start'
        elif previous_session is None and current_session is not None:
            return 'no_change'
        elif previous_session == current_session:
            return 'session_continuation'
        else:
            return 'session_change'

    def _get_active_session(self, timestamp: pd.Timestamp) -> Optional[str]:
        """Get active session name for timestamp."""
        for session_name, session in self.sessions.items():
            if self._is_in_session(timestamp, session_name):
                return session_name
        return None

    def _add_time_to_session_change(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add time to next session change."""
        try:
            # Sort by timestamp and calculate time to next session boundary
            df_sorted = df.sort_values(timestamp_col)

            def time_to_next_change(row):
                hour = row[timestamp_col].hour
                session = self._get_active_session(row[timestamp_col])

                if session:
                    session_info = self.sessions[session]
                    # Time until session end
                    if session['start'] <= session['end']:
                        if session['start'] <= hour < session['end']:
                            next_change = (session['end'] - hour) * 3600
                        else:
                            next_change = 0  # Session continues through midnight
                    else:
                        # Cross-midnight session
                        next_change = ((24 - session['start']) + session['end']) * 3600
                else:
                    next_change = 3600000  # No session

                return next_change

            df['time_to_session_change'] = df_sorted.apply(time_to_next_change, axis=1)
        except Exception as e:
            self.logger.error(f"Failed to add time_to_session_change: {e}")
            df['time_to_session_change'] = 3600

        return df

    def _add_session_duration_hours(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add session duration since start."""
        try:
            # Sort by timestamp and calculate session duration
            df_sorted = df.sort_values(timestamp_col)

            def calculate_session_duration(row):
                timestamp = row[timestamp_col]
                # Find session start
                session_start = timestamp

                # Search backwards for session start
                for i in range(max(0, len(df_sorted) - 10, len(df_sorted))):  # Look back at most 10 rows
                    prev_timestamp = df_sorted.iloc[max(0, len(df_sorted) - 10)][timestamp_col]
                    if prev_timestamp < timestamp:
                        prev_session = self._get_active_session(prev_timestamp)
                        if prev_session:
                            session_start = prev_timestamp
                            break

                # Calculate duration in hours
                duration_seconds = (timestamp - session_start).total_seconds()
                return duration_seconds / 3600

            df['session_duration_hours'] = df_sorted.apply(calculate_session_duration, axis=1)
        except Exception as e:
            self.logger.error(f"Failed to add session_duration_hours: {e}")
            df['session_duration_hours'] = 0

        return df


class MarketHoursFeatures:
    """Market hours and volatility features based on time of day."""

    def __init__(self, config: DataConfig):
        """Initialize market hours features.

        Args:
            config: Data configuration
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Feature configuration
        self.enabled_features = set([
            'is_market_open', 'time_to_market_open', 'time_to_market_close',
            'is_pre_market', 'is_post_market', 'is_lunch_time',
            'volatility_by_hour', 'volume_profile_by_hour', 'price_range_by_hour'
        ])

        # Market hours (simplified for crypto markets)
        self.market_hours = {
            'weekday': {'open': 0, 'close': 24},
            'weekend': {'open': 0, 'close': 24}
        }

    @performance_monitor("market_hours_features_engineering")
    def create_market_hours_features(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """Create market hours based features.

        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column

        Returns:
            DataFrame with market hours features added
        """
        if len(df) == 0:
            return df.copy()

        # Validate timestamp column
        if timestamp_col not in df.columns:
            self.logger.warning(f"Timestamp column '{timestamp_col}' not found in DataFrame")
            return df.copy()

        # Ensure timestamp is datetime
        df_normalized = self._normalize_timestamps(df, timestamp_col)

        feature_df = df_normalized.copy()

        # Add market hours features if enabled
        if 'is_market_open' in self.enabled_features:
            feature_df = self._add_is_market_open(feature_df, timestamp_col)
        if 'time_to_market_open' in self.enabled_features:
            feature_df = self._add_time_to_market_open(feature_df, timestamp_col)
        if 'time_to_market_close' in self.enabled_features:
            feature_df = self._add_time_to_market_close(feature_df, timestamp_col)
        if 'is_pre_market' in self.enabled_features:
            feature_df = self._add_is_pre_market(feature_df, timestamp_col)
        if 'is_post_market' in self.enabled_features:
            feature_df = self._add_is_post_market(feature_df, timestamp_col)
        if 'is_lunch_time' in self.enabled_features:
            feature_df = self._add_is_lunch_time(feature_df, timestamp_col)
        if 'volatility_by_hour' in self.enabled_features:
            feature_df = self._add_volatility_by_hour(feature_df, timestamp_col)
        if 'volume_profile_by_hour' in self.enabled_features:
            feature_df = self._add_volume_profile_by_hour(feature_df, timestamp_col)
        if 'price_range_by_hour' in self.enabled_features:
            feature_df = self._add_price_range_by_hour(feature_df, timestamp_col)

        self.logger.info(f"Created {len([col for col in feature_df.columns if col not in df.columns])} market hours features")
        return feature_df

    def _normalize_timestamps(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Normalize timestamp column to datetime with UTC timezone."""
        df_normalized = df.copy()

        if not pd.api.types.is_datetime64_any_dtype(df_normalized[timestamp_col]):
            try:
                df_normalized[timestamp_col] = pd.to_datetime(df_normalized[timestamp_col], utc=True, errors='coerce')
            except Exception as e:
                self.logger.warning(f"Failed to convert timestamp: {e}")
                df_normalized[timestamp_col] = pd.to_datetime(df_normalized[timestamp_col], unit='ms', utc=True, errors='coerce')

        return df_normalized

    def _add_is_market_open(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add is_market_open feature."""
        try:
            df['is_market_open'] = True  # Crypto markets are always open
        except Exception as e:
            self.logger.error(f"Failed to add is_market_open: {e}")
            df['is_market_open'] = False

        return df

    def _add_time_to_market_open(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add time to market open (always 0 for crypto)."""
        try:
            df['time_to_market_open'] = 0
        except Exception as e:
            self.logger.error(f"Failed to add time_to_market_open: {e}")
            df['time_to_market_open'] = 0

        return df

    def _add_time_to_market_close(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add time to market close (seconds until next day close)."""
        try:
            df['time_to_market_close'] = 0  # Always open, so this is always 0
        except Exception as e:
            self.logger.error(f"Failed to add time_to_market_close: {e}")
            df['time_to_market_close'] = 0

        return df

    def _add_is_pre_market(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add pre-market indicator."""
        try:
            hour = df[timestamp_col].dt.hour
            df['is_pre_market'] = ((hour >= 0) & (hour < 9))  # 9 AM
        except Exception as e:
            self.logger.error(f"Failed to add is_pre_market: {e}")
            df['is_pre_market'] = False

        return df

    def _add_is_post_market(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add post-market indicator."""
        try:
            hour = df[timestamp_col].dt.hour
            df['is_post_market'] = ((hour >= 16) & (hour < 24))  # 4 PM - 12 AM
        except Exception as e:
            self.logger.error(f"Failed to add is_post_market: {e}")
            df['is_post_market'] = False

        return df

    def _add_is_lunch_time(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add lunch time indicator."""
        try:
            hour = df[timestamp_col].dt.hour
            df['is_lunch_time'] = ((hour >= 11) & (hour < 14))  # 11 AM - 2 PM
        except Exception as e:
            self.logger.error(f"Failed to add is_lunch_time: {e}")
            df['is_lunch_time'] = False

        return df

    def _add_volatility_by_hour(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add volatility features by hour."""
        try:
            # This would typically require price column
            if 'close' in df.columns:
                df['hour'] = df[timestamp_col].dt.hour
                df['volatility'] = df.groupby('hour')['close'].transform('std')
            else:
                df['volatility'] = 0
        except Exception as e:
            self.logger.error(f"Failed to add volatility_by_hour: {e}")
            df['volatility'] = 0

        return df

    def _add_volume_profile_by_hour(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add volume profile by hour."""
        try:
            # This would typically require volume column
            if 'volume' in df.columns:
                df['hour'] = df[timestamp_col].dt.hour
                df['volume_profile'] = df.groupby('hour')['volume'].transform('mean')
            else:
                df['volume_profile'] = 0
        except Exception as e:
            self.logger.error(f"Failed to add volume_profile_by_hour: {e}")
            df['volume_profile'] = 0

        return df

    def _add_price_range_by_hour(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add price range features by hour."""
        try:
            # This would typically require price columns
            if 'high' in df.columns and 'low' in df.columns:
                df['hour'] = df[timestamp_col].dt.hour
                df['price_range'] = df.groupby('hour')[['high', 'low']].apply(lambda x: x['high'] - x['low'])
            else:
                df['price_range'] = 0
        except Exception as e:
            self.logger.error(f"Failed to add price_range_by_hour: {e}")
            df['price_range'] = 0

        return df