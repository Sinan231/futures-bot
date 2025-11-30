"""
Data storage and validation system for trading signal system.

Provides Parquet-based storage, data quality monitoring,
metadata management, and efficient querying for large
historical datasets with proper validation and error handling.
"""

import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import concurrent.futures

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import compute as pc

from ..utils.logging import get_logger, log_data_quality, performance_monitor
from ..utils.helpers import calculate_data_quality_score, is_outlier, remove_outliers
from ..utils.config import DataConfig
from .historical_fetcher import DataSource


@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics."""
    source: str
    symbol: str
    start_time: datetime
    end_time: datetime
    total_records: int
    missing_records: int
    duplicate_records: int
    invalid_prices: int
    invalid_volumes: int
    outlier_count: int
    quality_score: float
    timestamp_gaps: int
    completeness_ratio: float
    consistency_score: float


@dataclass
class DataStorageInfo:
    """Information about stored dataset."""
    source: DataSource
    symbol: str
    timeframe: Optional[str]
    start_time: datetime
    end_time: datetime
    record_count: int
    file_path: str
    file_size_mb: float
    created_at: datetime
    last_modified: datetime
    checksum: str
    compression: str
    metadata: Dict[str, Any]


class DataStorageManager:
    """Comprehensive data storage manager with quality monitoring."""

    def __init__(
        self,
        config: DataConfig,
        compression: str = "snappy",
        max_workers: int = 4,
        enable_checksum: bool = True
    ):
        """Initialize storage manager.

        Args:
            config: Data configuration
            compression: Parquet compression algorithm
            max_workers: Maximum number of parallel operations
            enable_checksum: Whether to generate file checksums
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.compression = compression
        self.max_workers = max_workers
        self.enable_checksum = enable_checksum

        # Initialize storage paths
        self._ensure_directories()

        # Initialize metadata database
        self._init_metadata_db()

        # Cache for frequently accessed data
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._cache_max_size = 10
        self._cache_ttl_seconds = 300  # 5 minutes

        # Performance metrics
        self._operation_stats = {
            'saves': 0,
            'loads': 0,
            'save_time_total': 0.0,
            'load_time_total': 0.0,
            'bytes_saved': 0,
            'bytes_loaded': 0
        }

    def _ensure_directories(self) -> None:
        """Create all required storage directories."""
        directories = [
            self.config.data_storage_path,
            Path(self.config.data_storage_path) / "raw",
            Path(self.config.data_storage_path) / "processed",
            Path(self.config.data_storage_path) / "features",
            Path(self.config.data_storage_path) / "signals",
            self.config.artifacts_storage_path,
            Path(self.config.artifacts_storage_path) / "models",
            Path(self.config.artifacts_storage_path) / "scalers",
            Path(self.config.artifacts_storage_path) / "features",
            Path(self.config.artifacts_storage_path) / "reports",
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def _init_metadata_db(self) -> None:
        """Initialize SQLite database for metadata tracking."""
        self.db_path = Path(self.config.data_storage_path) / "metadata.db"

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_storage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    record_count INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size_mb REAL NOT NULL,
                    checksum TEXT,
                    compression TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    last_modified TEXT NOT NULL,
                    UNIQUE(source, symbol, timeframe, start_time, end_time)
                )
            """)

            # Create indexes for efficient querying
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source_symbol ON data_storage(source, symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_start_time ON data_storage(start_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_end_time ON data_storage(end_time)")

            # Create table for data quality metrics
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT,
                    assessment_time TEXT NOT NULL,
                    total_records INTEGER NOT NULL,
                    missing_records INTEGER,
                    duplicate_records INTEGER,
                    invalid_prices INTEGER,
                    invalid_volumes INTEGER,
                    outlier_count INTEGER,
                    quality_score REAL,
                    timestamp_gaps INTEGER,
                    completeness_ratio REAL,
                    consistency_score REAL,
                    FOREIGN KEY (source, symbol, timeframe)
                    REFERENCES data_storage(source, symbol, timeframe)
                )
            """)

            conn.commit()

    @performance_monitor("data_save")
    def save_dataframe(
        self,
        df: pd.DataFrame,
        source: DataSource,
        symbol: str,
        timeframe: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        overwrite: bool = False,
        partition_by: Optional[str] = None
    ) -> Path:
        """Save DataFrame to Parquet with metadata and quality checks.

        Args:
            df: DataFrame to save
            source: Data source type
            symbol: Trading symbol
            timeframe: Timeframe (for kline data)
            start_time: Dataset start time
            end_time: Dataset end time
            overwrite: Whether to overwrite existing files
            partition_by: Column to partition by (default: date)

        Returns:
            Path to saved file
        """
        if len(df) == 0:
            self.logger.warning(f"Empty DataFrame for {source.value} {symbol} {timeframe or ''}")
            return None

        # Determine time range from data if not provided
        if 'timestamp' in df.columns:
            df_start = pd.to_datetime(df['timestamp'].min(), utc=True)
            df_end = pd.to_datetime(df['timestamp'].max(), utc=True)
            start_time = start_time or df_start
            end_time = end_time or df_end
        elif 'open_time' in df.columns:
            df_start = pd.to_datetime(df['open_time'].min(), utc=True)
            df_end = pd.to_datetime(df['open_time'].max(), utc=True)
            start_time = start_time or df_start
            end_time = end_time or df_end
        else:
            start_time = start_time or datetime.now(timezone.utc)
            end_time = end_time or datetime.now(timezone.utc)

        # Generate file path
        file_path = self.get_data_path(
            source, symbol, timeframe, start_time, end_time
        )

        # Check if file exists
        if file_path.exists() and not overwrite:
            self.logger.warning(f"File already exists: {file_path}")
            return file_path

        # Data quality assessment
        quality_metrics = self._assess_data_quality(df, source, symbol, timeframe)

        # Clean data if quality score is low
        if quality_metrics.quality_score < 50:
            self.logger.warning(
                f"Low quality score ({quality_metrics.quality_score:.1f}) "
                f"for {source.value} {symbol} {timeframe or ''}, attempting cleanup"
            )
            df = self._clean_dataframe(df, source)

        # Add metadata columns
        df = self._add_metadata_columns(df, source, symbol, timeframe)

        # Convert timestamp columns to datetime
        df = self._normalize_timestamps(df)

        # Save with Parquet
        self._save_parquet(
            df, file_path, source, symbol, timeframe, partition_by
        )

        # Generate checksum if enabled
        checksum = None
        if self.enable_checksum:
            checksum = self._generate_checksum(file_path)

        # Get file info
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        # Update metadata database
        storage_info = DataStorageInfo(
            source=source,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            record_count=len(df),
            file_path=str(file_path),
            file_size_mb=file_size_mb,
            created_at=datetime.now(timezone.utc),
            last_modified=datetime.now(timezone.utc),
            checksum=checksum,
            compression=self.compression,
            metadata=self._extract_metadata(df, source)
        )

        self._save_storage_info(storage_info)

        # Log quality metrics
        log_data_quality(
            data_source=f"{source.value}_{symbol}_{timeframe or ''}",
            record_count=len(df),
            quality_score=quality_metrics.quality_score,
            issues=self._get_quality_issues(quality_metrics)
        )

        # Update stats
        self._operation_stats['saves'] += 1
        self._operation_stats['bytes_saved'] += len(df) * df.memory_usage(deep=True).sum() / 1024

        self.logger.info(
            f"Saved {len(df):,} records to {file_path.name} "
            f"({file_size_mb:.2f} MB, quality: {quality_metrics.quality_score:.1f})"
        )

        return file_path

    @performance_monitor("data_load")
    def load_dataframe(
        self,
        source: DataSource,
        symbol: str,
        timeframe: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        columns: Optional[List[str]] = None,
        use_cache: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> Optional[pd.DataFrame]:
        """Load DataFrame from storage with optional filtering.

        Args:
            source: Data source type
            symbol: Trading symbol
            timeframe: Timeframe (for kline data)
            start_time: Filter by start time
            end_time: Filter by end time
            columns: Specific columns to load
            use_cache: Whether to use in-memory cache
            filters: Additional filters to apply

        Returns:
            DataFrame or None if not found
        """
        # Check cache first
        cache_key = self._generate_cache_key(source, symbol, timeframe, start_time, end_time)
        if use_cache and cache_key in self._data_cache:
            cached_df = self._data_cache[cache_key]
            self.logger.debug(f"Using cached data for {cache_key}")
            return cached_df[columns] if columns else cached_df

        # Find matching files in metadata
        files_info = self._find_matching_files(source, symbol, timeframe, start_time, end_time)
        if not files_info:
            self.logger.info(f"No files found for {source.value} {symbol} {timeframe or ''}")
            return None

        # Load and concatenate DataFrames
        dataframes = []
        for file_info in files_info:
            try:
                df = self._load_parquet(file_info.file_path, columns, filters)
                if len(df) > 0:
                    dataframes.append(df)
                    self.logger.debug(f"Loaded {len(df)} records from {file_info.file_path.name}")
            except Exception as e:
                self.logger.error(f"Failed to load {file_info.file_path}: {e}")
                continue

        if not dataframes:
            return None

        # Combine DataFrames
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Sort by timestamp
        if 'timestamp' in combined_df.columns:
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        elif 'open_time' in combined_df.columns:
            combined_df = combined_df.sort_values('open_time').reset_index(drop=True)

        # Apply time range filtering
        if start_time or end_time:
            time_col = 'timestamp' if 'timestamp' in combined_df.columns else 'open_time'
            if start_time:
                combined_df = combined_df[combined_df[time_col] >= start_time]
            if end_time:
                combined_df = combined_df[combined_df[time_col] <= end_time]

        # Cache result
        if use_cache and len(self._data_cache) < self._cache_max_size:
            self._data_cache[cache_key] = combined_df.copy()

        # Update stats
        self._operation_stats['loads'] += 1
        self._operation_stats['bytes_loaded'] += (
            len(combined_df) * combined_df.memory_usage(deep=True).sum() / 1024
        )

        self.logger.info(
            f"Loaded {len(combined_df):,} records for {source.value} {symbol} {timeframe or ''}"
        )

        return combined_df[columns] if columns else combined_df

    def get_data_path(
        self,
        source: DataSource,
        symbol: str,
        timeframe: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Path:
        """Generate standardized file path for data storage.

        Args:
            source: Data source type
            symbol: Trading symbol
            timeframe: Timeframe (for kline data)
            start_time: Dataset start time
            end_time: Dataset end time

        Returns:
            Path for data file
        """
        # Create directory structure
        base_dir = Path(self.config.data_storage_path)
        source_dir = base_dir / "raw" / source.value.lower()
        symbol_dir = source_dir / symbol.lower()

        # Generate filename
        timestamp_str = ""
        if start_time and end_time:
            timestamp_str = f"_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}"

        if timeframe:
            filename = f"{symbol.lower()}_{timeframe}{timestamp_str}.parquet"
        else:
            filename = f"{symbol.lower()}_{source.value.lower()}{timestamp_str}.parquet"

        return symbol_dir / filename

    def list_available_data(
        self,
        source: Optional[DataSource] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[DataStorageInfo]:
        """List available data in storage.

        Args:
            source: Filter by data source
            symbol: Filter by symbol
            timeframe: Filter by timeframe
            limit: Maximum number of results

        Returns:
            List of data storage information
        """
        query = "SELECT * FROM data_storage WHERE 1=1"
        params = []

        if source:
            query += " AND source = ?"
            params.append(source.value)

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe)

        query += " ORDER BY last_modified DESC"
        if limit:
            query += f" LIMIT {limit}"

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append(DataStorageInfo(
                source=DataSource(row[1]),
                symbol=row[2],
                timeframe=row[3],
                start_time=datetime.fromisoformat(row[4]),
                end_time=datetime.fromisoformat(row[5]),
                record_count=row[6],
                file_path=row[7],
                file_size_mb=row[8],
                checksum=row[9],
                compression=row[10],
                metadata=json.loads(row[11]) if row[11] else {},
                created_at=datetime.fromisoformat(row[12]),
                last_modified=datetime.fromisoformat(row[13])
            ))

        return results

    def get_quality_metrics(
        self,
        source: Optional[DataSource] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        days: int = 7
    ) -> List[DataQualityMetrics]:
        """Get data quality metrics for recent period.

        Args:
            source: Filter by data source
            symbol: Filter by symbol
            timeframe: Filter by timeframe
            days: Number of days to look back

        Returns:
            List of quality metrics
        """
        since_time = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        query = """
            SELECT * FROM data_quality
            WHERE assessment_time >= ?
            AND (? IS NULL OR source = ?)
            AND (? IS NULL OR symbol = ?)
            AND (? IS NULL OR timeframe = ?)
            ORDER BY assessment_time DESC
        """
        params = [since_time,
                   source.value if source else None, source.value if source else None,
                   symbol, symbol,
                   timeframe, timeframe]

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append(DataQualityMetrics(
                source=DataSource(row[1]),
                symbol=row[2],
                timeframe=row[3],
                assessment_time=datetime.fromisoformat(row[4]),
                total_records=row[5],
                missing_records=row[6],
                duplicate_records=row[7],
                invalid_prices=row[8],
                invalid_volumes=row[9],
                outlier_count=row[10],
                quality_score=row[11],
                timestamp_gaps=row[12],
                completeness_ratio=row[13],
                consistency_score=row[14]
            ))

        return results

    def cleanup_old_data(
        self,
        days_to_keep: int = 365,
        source: Optional[DataSource] = None
    ) -> int:
        """Clean up old data files.

        Args:
            days_to_keep: Days of data to retain
            source: Optional source filter

        Returns:
            Number of files cleaned up
        """
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_to_keep)).isoformat()

        query = """
            SELECT file_path FROM data_storage
            WHERE last_modified < ?
            AND (? IS NULL OR source = ?)
        """
        params = [cutoff_date, source.value if source else None, source.value if source else None]

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(query, params)
            file_paths = cursor.fetchall()

        cleaned_count = 0
        for (file_path,) in file_paths:
            try:
                path = Path(file_path)
                if path.exists():
                    path.unlink()
                    cleaned_count += 1
                    self.logger.info(f"Deleted old file: {path}")
            except Exception as e:
                self.logger.error(f"Failed to delete {file_path}: {e}")

        # Update database
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "DELETE FROM data_storage WHERE last_modified < ? AND (? IS NULL OR source = ?)",
                [cutoff_date, source.value if source else None, source.value if source else None]
            )
            conn.commit()

        self.logger.info(f"Cleaned up {cleaned_count} old data files")
        return cleaned_count

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics and metrics."""
        with sqlite3.connect(str(self.db_path)) as conn:
            # Total datasets
            total_datasets = conn.execute("SELECT COUNT(*) FROM data_storage").fetchone()[0]

            # Total size by source
            size_by_source = conn.execute("""
                SELECT source, SUM(file_size_mb) as total_mb
                FROM data_storage
                GROUP BY source
            """).fetchall()

            # Total size by symbol
            size_by_symbol = conn.execute("""
                SELECT symbol, SUM(file_size_mb) as total_mb
                FROM data_storage
                GROUP BY symbol
            """).fetchall()

            # Date range
            date_range = conn.execute("""
                SELECT MIN(start_time) as min_date, MAX(end_time) as max_date
                FROM data_storage
            """).fetchone()

        # Operation stats
        avg_save_time = (
            self._operation_stats['save_time_total'] / max(1, self._operation_stats['saves'])
        )
        avg_load_time = (
            self._operation_stats['load_time_total'] / max(1, self._operation_stats['loads'])
        )

        return {
            'total_datasets': total_datasets,
            'total_saves': self._operation_stats['saves'],
            'total_loads': self._operation_stats['loads'],
            'avg_save_time_seconds': avg_save_time,
            'avg_load_time_seconds': avg_load_time,
            'total_mb_saved': self._operation_stats['bytes_saved'] / (1024 * 1024),
            'total_mb_loaded': self._operation_stats['bytes_loaded'] / (1024 * 1024),
            'size_by_source': dict(size_by_source),
            'size_by_symbol': dict(size_by_symbol),
            'date_range': date_range,
            'cache_size': len(self._data_cache),
            'cache_usage_mb': sum(
                df.memory_usage(deep=True).sum()
                for df in self._data_cache.values()
            ) / (1024 * 1024)
        }

    # Private helper methods
    def _assess_data_quality(
        self,
        df: pd.DataFrame,
        source: DataSource,
        symbol: str,
        timeframe: Optional[str]
    ) -> DataQualityMetrics:
        """Comprehensive data quality assessment."""
        if len(df) == 0:
            return DataQualityMetrics(
                source=source,
                symbol=symbol,
                timeframe=timeframe,
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                total_records=0,
                missing_records=0,
                duplicate_records=0,
                invalid_prices=0,
                invalid_volumes=0,
                outlier_count=0,
                quality_score=0.0,
                timestamp_gaps=0,
                completeness_ratio=0.0,
                consistency_score=0.0
            )

        # Missing data
        missing_records = df.isnull().sum().sum()
        total_records = len(df)
        completeness_ratio = (total_records - missing_records) / total_records if total_records > 0 else 0

        # Duplicate detection
        time_col = 'timestamp' if 'timestamp' in df.columns else 'open_time'
        duplicate_records = df[time_col].duplicated().sum()

        # Price validation
        price_columns = [col for col in df.columns if 'price' in col.lower() or col in ['open', 'high', 'low', 'close']]
        invalid_prices = 0
        for col in price_columns:
            if col in df.columns:
                invalid_prices += (df[col] <= 0).sum()

        # Volume validation
        volume_columns = [col for col in df.columns if 'volume' in col.lower() or col in ['qty']]
        invalid_volumes = 0
        for col in volume_columns:
            if col in df.columns:
                invalid_volumes += (df[col] < 0).sum()

        # Outlier detection
        numeric_columns = df.select_dtypes(include=['number']).columns
        outlier_count = 0
        for col in numeric_columns:
            if col in df.columns:
                outliers = self._detect_outliers(df[col])
                outlier_count += outliers.sum()

        # Timestamp gaps detection
        timestamp_gaps = 0
        if time_col in df.columns and len(df) > 1:
            sorted_timestamps = df[time_col].sort_values()
            time_diffs = sorted_timestamps.diff().dt.total_seconds()
            # Define expected interval based on source
            expected_interval = self._get_expected_interval(source, timeframe)
            if expected_interval:
                gaps = time_diffs > expected_interval * 1.5  # 50% tolerance
                timestamp_gaps = gaps.sum()

        # Consistency score (price high/low relationships, etc.)
        consistency_score = self._calculate_consistency_score(df, source)

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            completeness_ratio, duplicate_records, invalid_prices,
            invalid_volumes, outlier_count, timestamp_gaps, consistency_score
        )

        # Determine time range
        start_time = df[time_col].min() if time_col in df.columns else datetime.now(timezone.utc)
        end_time = df[time_col].max() if time_col in df.columns else datetime.now(timezone.utc)

        return DataQualityMetrics(
            source=source,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            total_records=total_records,
            missing_records=missing_records,
            duplicate_records=duplicate_records,
            invalid_prices=invalid_prices,
            invalid_volumes=invalid_volumes,
            outlier_count=outlier_count,
            quality_score=quality_score,
            timestamp_gaps=timestamp_gaps,
            completeness_ratio=completeness_ratio,
            consistency_score=consistency_score
        )

    def _detect_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method."""
        if len(series) == 0:
            return pd.Series([], dtype=bool)

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        return (series < lower_bound) | (series > upper_bound)

    def _get_expected_interval(self, source: DataSource, timeframe: Optional[str]) -> Optional[float]:
        """Get expected time interval in seconds for data source."""
        if source == DataSource.KLINES and timeframe:
            timeframe_map = {
                '1m': 60, '3m': 180, '5m': 300, '15m': 900,
                '30m': 1800, '1h': 3600, '2h': 7200, '4h': 14400,
                '6h': 21600, '8h': 28800, '12h': 43200,
                '1d': 86400, '3d': 259200, '1w': 604800
            }
            return timeframe_map.get(timeframe)
        elif source == DataSource.MARK_PRICE:
            return 3.0  # Mark price updates every ~3 seconds
        elif source == DataSource.FUNDING_RATE:
            return 28800  # Every 8 hours
        elif source == DataSource.OPEN_INTEREST:
            return 300  # Every 5 minutes
        else:
            return None

    def _calculate_consistency_score(self, df: pd.DataFrame, source: DataSource) -> float:
        """Calculate data consistency score."""
        if len(df) == 0:
            return 0.0

        consistency_issues = 0

        # OHLC consistency checks
        if source == DataSource.KLINES and all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # High should be >= all OHLC values
            high_issues = (df['high'] < df['open']).sum() + \
                          (df['high'] < df['low']).sum() + \
                          (df['high'] < df['close']).sum()

            # Low should be <= all OHLC values
            low_issues = (df['low'] > df['open']).sum() + \
                        (df['low'] > df['high']).sum() + \
                        (df['low'] > df['close']).sum()

            consistency_issues += high_issues + low_issues

        # Price monotonicity within reasonable bounds
        if 'price' in df.columns and len(df) > 1:
            price_changes = df['price'].abs().pct_change()
            extreme_changes = (price_changes > 0.1).sum()  # >10% change
            consistency_issues += extreme_changes

        total_checks = len(df)
        return max(0, 1 - (consistency_issues / total_checks))

    def _calculate_quality_score(
        self,
        completeness: float,
        duplicates: int,
        invalid_prices: int,
        invalid_volumes: int,
        outliers: int,
        timestamp_gaps: int,
        consistency: float
    ) -> float:
        """Calculate overall quality score (0-100)."""
        # Weight components
        completeness_score = completeness * 100
        duplicate_penalty = min(20, (duplicates / max(1, completeness_score)) * 100)
        invalid_values_penalty = ((invalid_prices + invalid_volumes) / max(1, completeness_score)) * 100
        outlier_penalty = (outliers / max(1, completeness_score)) * 10
        gap_penalty = (timestamp_gaps / max(1, completeness_score)) * 100
        consistency_score = consistency * 100

        # Calculate weighted average
        total_score = (
            completeness_score * 0.3 +
            (100 - duplicate_penalty) * 0.2 +
            (100 - invalid_values_penalty) * 0.2 +
            (100 - outlier_penalty) * 0.1 +
            (100 - gap_penalty) * 0.1 +
            consistency_score * 0.1
        )

        return max(0, min(100, total_score))

    def _clean_dataframe(self, df: pd.DataFrame, source: DataSource) -> pd.DataFrame:
        """Clean DataFrame by removing invalid data."""
        df_clean = df.copy()

        # Remove duplicate timestamps
        time_col = 'timestamp' if 'timestamp' in df.columns else 'open_time'
        if time_col in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=[time_col], keep='last')

        # Remove invalid prices
        price_columns = [col for col in df_clean.columns if 'price' in col.lower() or col in ['open', 'high', 'low', 'close']]
        for col in price_columns:
            if col in df_clean.columns:
                df_clean = df_clean[df_clean[col] > 0]

        # Remove invalid volumes
        volume_columns = [col for col in df_clean.columns if 'volume' in col.lower() or col in ['qty']]
        for col in volume_columns:
            if col in df_clean.columns:
                df_clean = df_clean[df_clean[col] >= 0]

        # Remove extreme outliers (more than 5 standard deviations)
        numeric_columns = df_clean.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            if col in df_clean.columns and len(df_clean[col]) > 0:
                mean_val = df_clean[col].mean()
                std_val = df_clean[col].std()
                if std_val > 0:
                    df_clean = df_clean[abs(df_clean[col] - mean_val) <= 5 * std_val]

        return df_clean

    def _add_metadata_columns(
        self,
        df: pd.DataFrame,
        source: DataSource,
        symbol: str,
        timeframe: Optional[str]
    ) -> pd.DataFrame:
        """Add metadata columns to DataFrame."""
        df_metadata = df.copy()

        # Add standard metadata
        df_metadata['symbol'] = symbol
        df_metadata['data_source'] = source.value
        if timeframe:
            df_metadata['timeframe'] = timeframe

        # Add load timestamp
        df_metadata['loaded_at'] = datetime.now(timezone.utc)

        return df_metadata

    def _normalize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize timestamp columns to datetime with UTC timezone."""
        df_normalized = df.copy()

        timestamp_columns = ['timestamp', 'open_time', 'close_time', 'time', 'last_update_time']
        for col in timestamp_columns:
            if col in df_normalized.columns:
                df_normalized[col] = pd.to_datetime(df_normalized[col], utc=True)

        return df_normalized

    def _save_parquet(
        self,
        df: pd.DataFrame,
        file_path: Path,
        source: DataSource,
        symbol: str,
        timeframe: Optional[str],
        partition_by: Optional[str]
    ) -> None:
        """Save DataFrame to Parquet with optional partitioning."""
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Partition by date if specified
        if partition_by and partition_by in df.columns:
            # Extract date from timestamp column
            df['partition_date'] = pd.to_datetime(df[partition_by]).dt.strftime('%Y-%m-%d')

            # Save partitioned
            pq.write_to_dataset(
                df,
                root_path=file_path.parent,
                partition_cols=['partition_date'],
                compression=self.compression,
                write_metadata_file=True
            )
        else:
            # Save single file
            df.to_parquet(
                file_path,
                engine='pyarrow',
                compression=self.compression,
                index=False,
                write_metadata_file=True
            )

    def _load_parquet(
        self,
        file_path: Path,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Load Parquet file with optional column and row filtering."""
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # If it's a partitioned dataset, read from directory
        if file_path.is_dir() or file_path.name.endswith('/'):
            dataset = pq.ParquetDataset(file_path.parent)
            table = dataset.read(
                columns=columns,
                filters=filters
            ).to_table()
        else:
            # Read single file
            table = pq.read_table(
                file_path,
                columns=columns,
                filters=filters
            )

        return table.to_pandas()

    def _generate_checksum(self, file_path: Path) -> str:
        """Generate SHA-256 checksum for file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _extract_metadata(self, df: pd.DataFrame, source: DataSource) -> Dict[str, Any]:
        """Extract metadata from DataFrame."""
        metadata = {
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'shape': df.shape,
            'data_source': source.value
        }

        # Add time range if available
        time_col = 'timestamp' if 'timestamp' in df.columns else 'open_time'
        if time_col in df.columns:
            metadata['time_range'] = {
                'start': df[time_col].min().isoformat(),
                'end': df[time_col].max().isoformat(),
                'duration_hours': (
                    df[time_col].max() - df[time_col].min()
                ).total_seconds() / 3600
            }

        # Add statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            metadata['statistics'] = df[numeric_cols].describe().to_dict()

        return metadata

    def _save_storage_info(self, storage_info: DataStorageInfo) -> None:
        """Save storage information to metadata database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO data_storage
                (source, symbol, timeframe, start_time, end_time, record_count,
                 file_path, file_size_mb, checksum, compression, metadata,
                 created_at, last_modified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                storage_info.source.value,
                storage_info.symbol,
                storage_info.timeframe,
                storage_info.start_time.isoformat(),
                storage_info.end_time.isoformat(),
                storage_info.record_count,
                storage_info.file_path,
                storage_info.file_size_mb,
                storage_info.checksum,
                storage_info.compression,
                json.dumps(storage_info.metadata),
                storage_info.created_at.isoformat(),
                storage_info.last_modified.isoformat()
            ))

    def _find_matching_files(
        self,
        source: DataSource,
        symbol: str,
        timeframe: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[DataStorageInfo]:
        """Find files matching criteria in metadata database."""
        query = """
            SELECT * FROM data_storage
            WHERE source = ? AND symbol = ?
        """
        params = [source.value, symbol]

        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe)
        else:
            query += " AND timeframe IS NULL"

        if start_time:
            query += " AND end_time >= ?"
            params.append(start_time.isoformat())

        if end_time:
            query += " AND start_time <= ?"
            params.append(end_time.isoformat())

        query += " ORDER BY start_time, end_time"

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append(DataStorageInfo(
                source=DataSource(row[1]),
                symbol=row[2],
                timeframe=row[3],
                start_time=datetime.fromisoformat(row[4]),
                end_time=datetime.fromisoformat(row[5]),
                record_count=row[6],
                file_path=row[7],
                file_size_mb=row[8],
                checksum=row[9],
                compression=row[10],
                metadata=json.loads(row[11]) if row[11] else {},
                created_at=datetime.fromisoformat(row[12]),
                last_modified=datetime.fromisoformat(row[13])
            ))

        return results

    def _generate_cache_key(
        self,
        source: DataSource,
        symbol: str,
        timeframe: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> str:
        """Generate cache key for data lookup."""
        key_parts = [source.value, symbol]
        if timeframe:
            key_parts.append(timeframe)
        if start_time:
            key_parts.append(start_time.strftime('%Y%m%d'))
        if end_time:
            key_parts.append(end_time.strftime('%Y%m%d'))
        return "_".join(key_parts)

    def _get_quality_issues(self, metrics: DataQualityMetrics) -> List[str]:
        """Get list of quality issues from metrics."""
        issues = []

        if metrics.missing_records > 0:
            issues.append(f"Missing data: {metrics.missing_records} records")

        if metrics.duplicate_records > 0:
            issues.append(f"Duplicate timestamps: {metrics.duplicate_records} records")

        if metrics.invalid_prices > 0:
            issues.append(f"Invalid prices: {metrics.invalid_prices} records")

        if metrics.invalid_volumes > 0:
            issues.append(f"Invalid volumes: {metrics.invalid_volumes} records")

        if metrics.outlier_count > 0:
            issues.append(f"Outliers: {metrics.outlier_count} records")

        if metrics.timestamp_gaps > 0:
            issues.append(f"Timestamp gaps: {metrics.timestamp_gaps}")

        if metrics.consistency_score < 80:
            issues.append(f"Low consistency score: {metrics.consistency_score:.1f}")

        return issues