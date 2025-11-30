"""
Historical data fetcher for Binance market data.

Downloads and processes historical data for multiple timeframes and
data types including klines, trades, order book depth, funding rates,
open interest, and other market microstructure data with robust
error handling and resume capability.
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import concurrent.futures
from dataclasses import dataclass, asdict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rich.progress import Progress, TaskID

from .binance_client import BinanceClient, BinanceAPIError, BinanceRateLimitError
from .storage import DataStorageManager
from ..utils.logging import get_logger, performance_monitor, log_data_quality
from ..utils.helpers import (
    parse_timeframe_to_seconds,
    chunk_dataframe,
    display_progress_with_rich
)
from ..config import DataConfig


class DataSource(Enum):
    """Enumeration of available data sources."""
    KLINES = "klines"
    AGG_TRADES = "agg_trades"
    DEPTH = "depth"
    TRADES = "trades"
    MARK_PRICE = "mark_price"
    FUNDING_RATE = "funding_rate"
    OPEN_INTEREST = "open_interest"
    OPEN_INTEREST_HIST = "open_interest_hist"
    TICKER_24HR = "ticker_24hr"
    BOOK_TICKER = "book_ticker"


@dataclass
class DataDownloadResult:
    """Result of data download operation."""
    source: DataSource
    symbol: str
    timeframe: Optional[str]
    start_time: datetime
    end_time: datetime
    records_count: int
    success: bool
    error_message: Optional[str] = None
    file_path: Optional[str] = None
    download_duration_seconds: Optional[float] = None


class HistoricalDataFetcher:
    """Comprehensive historical data downloader for Binance."""

    def __init__(
        self,
        config: DataConfig,
        storage_manager: Optional[DataStorageManager] = None,
        max_workers: int = 4
    ):
        """Initialize historical data fetcher.

        Args:
            config: Data configuration
            storage_manager: Data storage manager
            max_workers: Maximum number of concurrent downloads
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.max_workers = max_workers

        # Initialize components
        self.client = BinanceClient(config, config.binance_testnet)
        self.storage = storage_manager or DataStorageManager(config)

        # Progress tracking
        self.download_results: List[DataDownloadResult] = []
        self.total_records = 0
        self.failed_downloads = 0

    @performance_monitor("download_all_data")
    def download_all_data(
        self,
        symbols: List[str],
        timeframes: Optional[List[str]] = None,
        sources: Optional[List[DataSource]] = None,
        months: Optional[int] = None,
        parallel: bool = True
    ) -> List[DataDownloadResult]:
        """Download all historical data for specified symbols and configurations.

        Args:
            symbols: List of trading symbols to download
            timeframes: List of timeframes (for klines data)
            sources: List of data sources to download
            months: Number of months of history to download
            parallel: Whether to download in parallel

        Returns:
            List of download results
        """
        self.logger.info(f"Starting historical data download for {len(symbols)} symbols")

        # Use defaults from config if not specified
        if timeframes is None:
            timeframes = self.config.timeframes
        if sources is None:
            sources = [
                DataSource.KLINES,
                DataSource.AGG_TRADES,
                DataSource.DEPTH,
                DataSource.MARK_PRICE,
                DataSource.FUNDING_RATE,
                DataSource.OPEN_INTEREST_HIST,
            ]
        if months is None:
            months = self.config.history_months

        # Calculate date range
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=months * 30)

        self.logger.info(f"Downloading data from {start_time} to {end_time}")

        # Generate download tasks
        download_tasks = self._generate_download_tasks(
            symbols, timeframes, sources, start_time, end_time
        )

        # Execute downloads
        if parallel:
            results = self._download_parallel(download_tasks)
        else:
            results = self._download_sequential(download_tasks)

        # Log summary
        self._log_download_summary(results)

        return results

    def _generate_download_tasks(
        self,
        symbols: List[str],
        timeframes: List[str],
        sources: List[DataSource],
        start_time: datetime,
        end_time: datetime
    ) -> List[Tuple[DataSource, str, Optional[str], datetime, datetime]]:
        """Generate list of download tasks."""
        tasks = []

        for symbol in symbols:
            for source in sources:
                if source == DataSource.KLINES:
                    # Generate klines tasks for each timeframe
                    for timeframe in timeframes:
                        tasks.append((source, symbol, timeframe, start_time, end_time))
                elif source in [
                    DataSource.AGG_TRADES,
                    DataSource.DEPTH,
                    DataSource.TRADES,
                    DataSource.BOOK_TICKER,
                    DataSource.MARK_PRICE,
                    DataSource.FUNDING_RATE,
                    DataSource.OPEN_INTEREST,
                    DataSource.OPEN_INTEREST_HIST,
                    DataSource.TICKER_24HR
                ]:
                    # These don't require timeframe
                    tasks.append((source, symbol, None, start_time, end_time))

        return tasks

    def _download_parallel(
        self,
        download_tasks: List[Tuple[DataSource, str, Optional[str], datetime, datetime]]
    ) -> List[DataDownloadResult]:
        """Execute downloads in parallel using thread pool."""
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._download_single, *task): task
                for task in download_tasks
            }

            # Process completed tasks
            with Progress() as progress:
                task_progress = progress.add_task("Downloading data...", total=len(download_tasks))

                for future in concurrent.futures.as_completed(future_to_task):
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout
                        results.append(result)
                        progress.update(task_progress, advance=1)
                    except Exception as e:
                        task = future_to_task[future]
                        error_result = DataDownloadResult(
                            source=task[0],
                            symbol=task[1],
                            timeframe=task[2],
                            start_time=task[3],
                            end_time=task[4],
                            records_count=0,
                            success=False,
                            error_message=str(e)
                        )
                        results.append(error_result)
                        progress.update(task_progress, advance=1)

        return results

    def _download_sequential(
        self,
        download_tasks: List[Tuple[DataSource, str, Optional[str], datetime, datetime]]
    ) -> List[DataDownloadResult]:
        """Execute downloads sequentially."""
        results = []

        with Progress() as progress:
            task_progress = progress.add_task("Downloading data...", total=len(download_tasks))

            for task in download_tasks:
                try:
                    result = self._download_single(*task)
                    results.append(result)
                    progress.update(task_progress, advance=1)
                except Exception as e:
                    error_result = DataDownloadResult(
                        source=task[0],
                        symbol=task[1],
                        timeframe=task[2],
                        start_time=task[3],
                        end_time=task[4],
                        records_count=0,
                        success=False,
                        error_message=str(e)
                    )
                    results.append(error_result)

        return results

    def _download_single(
        self,
        source: DataSource,
        symbol: str,
        timeframe: Optional[str],
        start_time: datetime,
        end_time: datetime
    ) -> DataDownloadResult:
        """Download single data source for symbol and timeframe."""
        start_download = datetime.now()
        self.logger.debug(f"Downloading {source.value} for {symbol} {timeframe or ''}")

        try:
            if source == DataSource.KLINES:
                return self._download_klines(symbol, timeframe, start_time, end_time, start_download)
            elif source == DataSource.AGG_TRADES:
                return self._download_agg_trades(symbol, start_time, end_time, start_download)
            elif source == DataSource.DEPTH:
                return self._download_depth(symbol, start_time, end_time, start_download)
            elif source == DataSource.TRADES:
                return self._download_trades(symbol, start_time, end_time, start_download)
            elif source == DataSource.MARK_PRICE:
                return self._download_mark_price(symbol, start_time, end_time, start_download)
            elif source == DataSource.FUNDING_RATE:
                return self._download_funding_rate(symbol, start_time, end_time, start_download)
            elif source == DataSource.OPEN_INTEREST:
                return self._download_open_interest(symbol, start_time, end_time, start_download)
            elif source == DataSource.OPEN_INTEREST_HIST:
                return self._download_open_interest_hist(symbol, start_time, end_time, start_download)
            elif source == DataSource.TICKER_24HR:
                return self._download_ticker_24hr(symbol, start_time, end_time, start_download)
            elif source == DataSource.BOOK_TICKER:
                return self._download_book_ticker(symbol, start_time, end_time, start_download)
            else:
                raise ValueError(f"Unsupported data source: {source}")

        except Exception as e:
            duration = (datetime.now() - start_download).total_seconds()
            return DataDownloadResult(
                source=source,
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                records_count=0,
                success=False,
                error_message=str(e),
                download_duration_seconds=duration
            )

    def _download_klines(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        start_download: datetime
    ) -> DataDownloadResult:
        """Download kline/candlestick data."""
        all_klines = []
        current_start = start_time
        batch_size_days = 30  # Download in 30-day batches

        while current_start < end_time:
            batch_end = min(current_start + timedelta(days=batch_size_days), end_time)

            # Download batch
            klines_data = self.client.get_klines(
                symbol=symbol,
                interval=timeframe,
                start_time=current_start,
                end_time=batch_end,
                limit=1500
            )

            if not klines_data:
                current_start = batch_end
                continue

            all_klines.extend(klines_data)
            current_start = batch_end

            # Small delay to respect rate limits
            time.sleep(0.1)

        # Convert to DataFrame
        if all_klines:
            df = self._process_klines_data(all_klines, timeframe)

            # Save to storage
            file_path = self._save_dataframe(
                df, source=DataSource.KLINES, symbol=symbol,
                timeframe=timeframe, start_time=start_time, end_time=end_time
            )

            duration = (datetime.now() - start_download).total_seconds()

            return DataDownloadResult(
                source=DataSource.KLINES,
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                records_count=len(df),
                success=True,
                file_path=str(file_path),
                download_duration_seconds=duration
            )
        else:
            return DataDownloadResult(
                source=DataSource.KLINES,
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                records_count=0,
                success=True,
                download_duration_seconds=(datetime.now() - start_download).total_seconds()
            )

    def _download_agg_trades(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        start_download: datetime
    ) -> DataDownloadResult:
        """Download aggregate trades data."""
        all_trades = []
        current_start = start_time
        batch_size_days = 7  # Download in 7-day batches (trade data is larger)

        while current_start < end_time:
            batch_end = min(current_start + timedelta(days=batch_size_days), end_time)

            # Download batch
            trades_data = self.client.get_agg_trades(
                symbol=symbol,
                start_time=current_start,
                end_time=batch_end,
                limit=1000
            )

            if not trades_data:
                current_start = batch_end
                continue

            all_trades.extend(trades_data)
            current_start = batch_end

            # Small delay to respect rate limits
            time.sleep(0.1)

        # Convert to DataFrame
        if all_trades:
            df = self._process_agg_trades_data(all_trades)

            # Save to storage
            file_path = self._save_dataframe(
                df, source=DataSource.AGG_TRADES, symbol=symbol,
                start_time=start_time, end_time=end_time
            )

            duration = (datetime.now() - start_download).total_seconds()

            return DataDownloadResult(
                source=DataSource.AGG_TRADES,
                symbol=symbol,
                timeframe=None,
                start_time=start_time,
                end_time=end_time,
                records_count=len(df),
                success=True,
                file_path=str(file_path),
                download_duration_seconds=duration
            )
        else:
            return DataDownloadResult(
                source=DataSource.AGG_TRADES,
                symbol=symbol,
                timeframe=None,
                start_time=start_time,
                end_time=end_time,
                records_count=0,
                success=True,
                download_duration_seconds=(datetime.now() - start_download).total_seconds()
            )

    def _download_depth(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        start_download: datetime
    ) -> DataDownloadResult:
        """Download order book depth snapshots."""
        # For historical depth data, we'll take snapshots at regular intervals
        depth_snapshots = []
        current_time = start_time
        snapshot_interval = timedelta(hours=1)  # Take snapshot every hour

        while current_time <= end_time:
            try:
                depth_data = self.client.get_depth(symbol=symbol, limit=1000)

                # Add timestamp to snapshot
                depth_data['timestamp'] = current_time
                depth_snapshots.append(depth_data)

                current_time += snapshot_interval

                # Small delay
                time.sleep(0.2)

            except Exception as e:
                self.logger.warning(f"Failed to get depth snapshot at {current_time}: {e}")
                current_time += snapshot_interval

        # Convert to DataFrame
        if depth_snapshots:
            df = self._process_depth_data(depth_snapshots)

            # Save to storage
            file_path = self._save_dataframe(
                df, source=DataSource.DEPTH, symbol=symbol,
                start_time=start_time, end_time=end_time
            )

            duration = (datetime.now() - start_download).total_seconds()

            return DataDownloadResult(
                source=DataSource.DEPTH,
                symbol=symbol,
                timeframe=None,
                start_time=start_time,
                end_time=end_time,
                records_count=len(df),
                success=True,
                file_path=str(file_path),
                download_duration_seconds=duration
            )
        else:
            return DataDownloadResult(
                source=DataSource.DEPTH,
                symbol=symbol,
                timeframe=None,
                start_time=start_time,
                end_time=end_time,
                records_count=0,
                success=True,
                download_duration_seconds=(datetime.now() - start_download).total_seconds()
            )

    def _download_mark_price(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        start_download: datetime
    ) -> DataDownloadResult:
        """Download mark price and funding rate data."""
        mark_price_data = self.client.get_funding_rate(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            limit=1000
        )

        if mark_price_data:
            df = self._process_mark_price_data(mark_price_data)

            # Save to storage
            file_path = self._save_dataframe(
                df, source=DataSource.MARK_PRICE, symbol=symbol,
                start_time=start_time, end_time=end_time
            )

            duration = (datetime.now() - start_download).total_seconds()

            return DataDownloadResult(
                source=DataSource.MARK_PRICE,
                symbol=symbol,
                timeframe=None,
                start_time=start_time,
                end_time=end_time,
                records_count=len(df),
                success=True,
                file_path=str(file_path),
                download_duration_seconds=duration
            )
        else:
            return DataDownloadResult(
                source=DataSource.MARK_PRICE,
                symbol=symbol,
                timeframe=None,
                start_time=start_time,
                end_time=end_time,
                records_count=0,
                success=True,
                download_duration_seconds=(datetime.now() - start_download).total_seconds()
            )

    def _download_funding_rate(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        start_download: datetime
    ) -> DataDownloadResult:
        """Download funding rate history."""
        # This is included in mark price data, so we use the same endpoint
        return self._download_mark_price(symbol, start_time, end_time, start_download)

    def _download_open_interest_hist(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        start_download: datetime
    ) -> DataDownloadResult:
        """Download open interest history."""
        oi_data = self.client.get_open_interest_hist(
            symbol=symbol,
            period="5m",  # 5-minute intervals
            start_time=start_time,
            end_time=end_time,
            limit=500
        )

        if oi_data:
            df = self._process_open_interest_hist_data(oi_data)

            # Save to storage
            file_path = self._save_dataframe(
                df, source=DataSource.OPEN_INTEREST_HIST, symbol=symbol,
                start_time=start_time, end_time=end_time
            )

            duration = (datetime.now() - start_download).total_seconds()

            return DataDownloadResult(
                source=DataSource.OPEN_INTEREST_HIST,
                symbol=symbol,
                timeframe=None,
                start_time=start_time,
                end_time=end_time,
                records_count=len(df),
                success=True,
                file_path=str(file_path),
                download_duration_seconds=duration
            )
        else:
            return DataDownloadResult(
                source=DataSource.OPEN_INTEREST_HIST,
                symbol=symbol,
                timeframe=None,
                start_time=start_time,
                end_time=end_time,
                records_count=0,
                success=True,
                download_duration_seconds=(datetime.now() - start_download).total_seconds()
            )

    def _download_trades(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        start_download: datetime
    ) -> DataDownloadResult:
        """Download recent trades data."""
        # For historical trades, this would be very large
        # We'll get a sample for now
        trades_data = self.client.get_trades(symbol=symbol, limit=1000)

        if trades_data:
            df = self._process_trades_data(trades_data)

            # Save to storage
            file_path = self._save_dataframe(
                df, source=DataSource.TRADES, symbol=symbol,
                start_time=start_time, end_time=end_time
            )

            duration = (datetime.now() - start_download).total_seconds()

            return DataDownloadResult(
                source=DataSource.TRADES,
                symbol=symbol,
                timeframe=None,
                start_time=start_time,
                end_time=end_time,
                records_count=len(df),
                success=True,
                file_path=str(file_path),
                download_duration_seconds=duration
            )
        else:
            return DataDownloadResult(
                source=DataSource.TRADES,
                symbol=symbol,
                timeframe=None,
                start_time=start_time,
                end_time=end_time,
                records_count=0,
                success=True,
                download_duration_seconds=(datetime.now() - start_download).total_seconds()
            )

    def _download_open_interest(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        start_download: datetime
    ) -> DataDownloadResult:
        """Download current open interest (single snapshot)."""
        oi_data = self.client.get_open_interest(symbol=symbol)

        if oi_data:
            # Create a single record DataFrame
            df = pd.DataFrame([{
                'symbol': oi_data['symbol'],
                'open_interest': float(oi_data['openInterest']),
                'time': datetime.now(timezone.utc)
            }])

            # Save to storage
            file_path = self._save_dataframe(
                df, source=DataSource.OPEN_INTEREST, symbol=symbol,
                start_time=start_time, end_time=end_time
            )

            duration = (datetime.now() - start_download).total_seconds()

            return DataDownloadResult(
                source=DataSource.OPEN_INTEREST,
                symbol=symbol,
                timeframe=None,
                start_time=start_time,
                end_time=end_time,
                records_count=1,
                success=True,
                file_path=str(file_path),
                download_duration_seconds=duration
            )
        else:
            return DataDownloadResult(
                source=DataSource.OPEN_INTEREST,
                symbol=symbol,
                timeframe=None,
                start_time=start_time,
                end_time=end_time,
                records_count=0,
                success=True,
                download_duration_seconds=(datetime.now() - start_download).total_seconds()
            )

    def _download_ticker_24hr(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        start_download: datetime
    ) -> DataDownloadResult:
        """Download 24hr ticker data."""
        ticker_data = self.client.get_ticker_24hr(symbol=symbol)

        if ticker_data:
            df = pd.DataFrame([{
                **ticker_data,
                'download_time': datetime.now(timezone.utc)
            }])

            # Save to storage
            file_path = self._save_dataframe(
                df, source=DataSource.TICKER_24HR, symbol=symbol,
                start_time=start_time, end_time=end_time
            )

            duration = (datetime.now() - start_download).total_seconds()

            return DataDownloadResult(
                source=DataSource.TICKER_24HR,
                symbol=symbol,
                timeframe=None,
                start_time=start_time,
                end_time=end_time,
                records_count=1,
                success=True,
                file_path=str(file_path),
                download_duration_seconds=duration
            )
        else:
            return DataDownloadResult(
                source=DataSource.TICKER_24HR,
                symbol=symbol,
                timeframe=None,
                start_time=start_time,
                end_time=end_time,
                records_count=0,
                success=True,
                download_duration_seconds=(datetime.now() - start_download).total_seconds()
            )

    def _download_book_ticker(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        start_download: datetime
    ) -> DataDownloadResult:
        """Download order book ticker data."""
        book_ticker_data = self.client.get_book_ticker(symbol=symbol)

        if book_ticker_data:
            df = pd.DataFrame([{
                **book_ticker_data,
                'download_time': datetime.now(timezone.utc)
            }])

            # Save to storage
            file_path = self._save_dataframe(
                df, source=DataSource.BOOK_TICKER, symbol=symbol,
                start_time=start_time, end_time=end_time
            )

            duration = (datetime.now() - start_download).total_seconds()

            return DataDownloadResult(
                source=DataSource.BOOK_TICKER,
                symbol=symbol,
                timeframe=None,
                start_time=start_time,
                end_time=end_time,
                records_count=1,
                success=True,
                file_path=str(file_path),
                download_duration_seconds=duration
            )
        else:
            return DataDownloadResult(
                source=DataSource.BOOK_TICKER,
                symbol=symbol,
                timeframe=None,
                start_time=start_time,
                end_time=end_time,
                records_count=0,
                success=True,
                download_duration_seconds=(datetime.now() - start_download).total_seconds()
            )

    # Data processing methods
    def _process_klines_data(self, klines_data: List[List], timeframe: str) -> pd.DataFrame:
        """Process klines data into DataFrame."""
        columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades_count', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ]

        df = pd.DataFrame(klines_data, columns=columns)

        # Convert data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                         'taker_buy_volume', 'taker_buy_quote_volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['trades_count'] = pd.to_numeric(df['trades_count'], errors='coerce')

        # Convert timestamps
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)

        # Add derived columns
        df['symbol'] = self.config.default_pair
        df['timeframe'] = timeframe
        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3  # Typical price

        # Remove unnecessary column
        df = df.drop(columns=['ignore'])

        return df

    def _process_agg_trades_data(self, trades_data: List[Dict]) -> pd.DataFrame:
        """Process aggregate trades data into DataFrame."""
        df = pd.DataFrame(trades_data)

        # Convert data types
        numeric_columns = ['price', 'qty', 'quote_qty', 'first_id', 'last_id']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

        return df

    def _process_depth_data(self, depth_snapshots: List[Dict]) -> pd.DataFrame:
        """Process depth snapshots data into DataFrame."""
        # Flatten the depth data
        processed_data = []

        for snapshot in depth_snapshots:
            timestamp = snapshot.pop('timestamp', datetime.now(timezone.utc))

            # Process bids
            for i, (price, quantity) in enumerate(snapshot.get('bids', [])):
                processed_data.append({
                    'timestamp': timestamp,
                    'side': 'bid',
                    'level': i + 1,
                    'price': float(price),
                    'quantity': float(quantity)
                })

            # Process asks
            for i, (price, quantity) in enumerate(snapshot.get('asks', [])):
                processed_data.append({
                    'timestamp': timestamp,
                    'side': 'ask',
                    'level': i + 1,
                    'price': float(price),
                    'quantity': float(quantity)
                })

        df = pd.DataFrame(processed_data)
        return df

    def _process_mark_price_data(self, mark_price_data: List[Dict]) -> pd.DataFrame:
        """Process mark price and funding rate data into DataFrame."""
        df = pd.DataFrame(mark_price_data)

        # Convert data types
        numeric_columns = ['markPrice', 'indexPrice', 'estimatedSettlePrice', 'fundingRate', 'nextFundingTime']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert timestamps
        df['timestamp'] = pd.to_datetime(df['time'], unit='ms', utc=True)
        if 'nextFundingTime' in df.columns:
            df['next_funding_time'] = pd.to_datetime(df['nextFundingTime'], unit='ms', utc=True)

        # Rename columns for consistency
        df = df.rename(columns={
            'markPrice': 'mark_price',
            'indexPrice': 'index_price',
            'estimatedSettlePrice': 'estimated_settle_price',
            'fundingRate': 'funding_rate'
        })

        return df

    def _process_open_interest_hist_data(self, oi_data: List[Dict]) -> pd.DataFrame:
        """Process open interest history data into DataFrame."""
        df = pd.DataFrame(oi_data)

        # Convert data types
        numeric_columns = ['sumOpenInterest', 'sumOpenInterestValue', 'timestamp']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

        # Rename columns for consistency
        df = df.rename(columns={
            'sumOpenInterest': 'open_interest',
            'sumOpenInterestValue': 'open_interest_value'
        })

        return df

    def _process_trades_data(self, trades_data: List[Dict]) -> pd.DataFrame:
        """Process trades data into DataFrame."""
        df = pd.DataFrame(trades_data)

        # Convert data types
        numeric_columns = ['price', 'qty', 'quoteQty']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['time'], unit='ms', utc=True)

        # Add is_buyer_maker column
        df['is_buyer_maker'] = df['isBuyerMaker'].astype(bool)

        return df

    def _save_dataframe(
        self,
        df: pd.DataFrame,
        source: DataSource,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        timeframe: Optional[str] = None
    ) -> Path:
        """Save DataFrame to Parquet file."""
        # Generate file path
        file_path = self.storage.get_data_path(
            source=source,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )

        # Save as Parquet with compression
        df.to_parquet(
            file_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

        # Log data quality
        log_data_quality(
            data_source=f"{source.value}_{symbol}",
            record_count=len(df),
            quality_score=self._calculate_data_quality_score(df),
            issues=self._validate_data_quality(df)
        )

        self.logger.debug(f"Saved {len(df)} records to {file_path}")

        return file_path

    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score (0-100)."""
        if len(df) == 0:
            return 0.0

        # Check for missing values
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        missing_score = (1 - missing_ratio) * 100

        # Check for duplicate timestamps
        if 'timestamp' in df.columns:
            duplicate_ratio = df['timestamp'].duplicated().sum() / len(df)
            duplicate_score = (1 - duplicate_ratio) * 100
        else:
            duplicate_score = 100

        # Check for outliers (basic check)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            outlier_count = 0
            total_values = 0

            for col in numeric_columns:
                values = df[col].dropna()
                if len(values) > 0:
                    q1 = values.quantile(0.25)
                    q3 = values.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    outliers = ((values < lower_bound) | (values > upper_bound)).sum()
                    outlier_count += outliers
                    total_values += len(values)

            outlier_ratio = outlier_count / total_values if total_values > 0 else 0
            outlier_score = (1 - outlier_ratio) * 100
        else:
            outlier_score = 100

        # Weighted average of scores
        quality_score = (missing_score * 0.4 + duplicate_score * 0.3 + outlier_score * 0.3)

        return min(100, max(0, quality_score))

    def _validate_data_quality(self, df: pd.DataFrame) -> List[str]:
        """Validate data quality and return list of issues."""
        issues = []

        # Check for empty data
        if len(df) == 0:
            issues.append("Empty dataset")
            return issues

        # Check for missing values
        missing_columns = df.columns[df.isnull().any()].tolist()
        if missing_columns:
            issues.append(f"Missing values in columns: {missing_columns}")

        # Check for duplicate timestamps
        if 'timestamp' in df.columns:
            duplicate_count = df['timestamp'].duplicated().sum()
            if duplicate_count > 0:
                issues.append(f"Duplicate timestamps: {duplicate_count}")

        # Check for invalid prices
        price_columns = [col for col in df.columns if 'price' in col.lower() or col in ['open', 'high', 'low', 'close']]
        for col in price_columns:
            if col in df.columns:
                invalid_prices = (df[col] <= 0).sum()
                if invalid_prices > 0:
                    issues.append(f"Invalid prices in {col}: {invalid_prices}")

        # Check for invalid volumes
        volume_columns = [col for col in df.columns if 'volume' in col.lower() or col == 'qty']
        for col in volume_columns:
            if col in df.columns:
                invalid_volumes = (df[col] < 0).sum()
                if invalid_volumes > 0:
                    issues.append(f"Invalid volumes in {col}: {invalid_volumes}")

        return issues

    def _log_download_summary(self, results: List[DataDownloadResult]) -> None:
        """Log summary of download results."""
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r.success)
        failed_tasks = total_tasks - successful_tasks
        total_records = sum(r.records_count for r in results if r.success)

        self.logger.info(
            f"Download summary: {successful_tasks}/{total_tasks} successful, "
            f"{failed_tasks} failed, {total_records:,} total records"
        )

        # Log failed downloads
        if failed_tasks > 0:
            self.logger.error("Failed downloads:")
            for result in results:
                if not result.success:
                    self.logger.error(
                        f"  {result.source.value} {result.symbol} {result.timeframe or ''}: "
                        f"{result.error_message}"
                    )

    def get_download_statistics(self) -> Dict[str, Any]:
        """Get statistics about recent downloads."""
        if not self.download_results:
            return {}

        total_downloads = len(self.download_results)
        successful_downloads = sum(1 for r in self.download_results if r.success)
        total_records = sum(r.records_count for r in self.download_results if r.success)

        avg_duration = np.mean([r.download_duration_seconds for r in self.download_results
                                if r.download_duration_seconds is not None])

        return {
            'total_downloads': total_downloads,
            'successful_downloads': successful_downloads,
            'success_rate': successful_downloads / total_downloads if total_downloads > 0 else 0,
            'total_records': total_records,
            'average_duration_seconds': avg_duration,
            'total_records_per_second': total_records / sum(r.download_duration_seconds for r in self.download_results
                                                       if r.download_duration_seconds is not None and r.download_duration_seconds > 0)
                                                       if any(r.download_duration_seconds for r in self.download_results) else 1
        }