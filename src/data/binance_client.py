"""
Binance API client for trading signal system.

Provides robust REST API client with rate limiting, error handling,
and comprehensive support for all required Binance endpoints including
spot, futures, and data APIs.
"""

import asyncio
import hashlib
import hmac
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlencode

import aiohttp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..utils.logging import get_logger, performance_monitor
from ..utils.helpers import retry_with_exponential_backoff, rate_limiter
from ..config import DataConfig


class BinanceAPIError(Exception):
    """Base exception for Binance API errors."""

    def __init__(self, message: str, code: Optional[int] = None):
        super().__init__(message)
        self.code = code
        self.message = message


class BinanceRateLimitError(BinanceAPIError):
    """Exception raised when API rate limit is exceeded."""

    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class BinanceClient:
    """Comprehensive Binance REST API client."""

    def __init__(
        self,
        config: DataConfig,
        testnet: bool = False,
        session_timeout: int = 30,
        max_retries: int = 3,
    ):
        """Initialize Binance client.

        Args:
            config: Data configuration with API credentials
            testnet: Whether to use testnet endpoints
            session_timeout: Session timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.config = config
        self.testnet = testnet
        self.logger = get_logger(__name__)

        # API base URLs
        if testnet:
            self.spot_base_url = "https://testnet.binance.vision"
            self.futures_base_url = "https://testnet.binancefuture.com"
            self.fapi_base_url = "https://testnet.binancefuture.com"
        else:
            self.spot_base_url = "https://api.binance.com"
            self.futures_base_url = "https://fapi.binance.com"
            self.fapi_base_url = "https://fapi.binance.com"

        # Rate limiting
        self._rate_limiter = rate_limiter(
            calls=config.binance_rate_limit,
            period_seconds=60.0
        )

        # Session management
        self.session_timeout = session_timeout
        self.max_retries = max_retries

        # Initialize sessions
        self._init_sync_session()
        self._async_session = None

        # Request weight tracking
        self._request_weights = {
            'general': 1,
            'klines': 1,
            'agg_trades': 1,
            'depth': 5,
            'trades': 1,
            'ticker': 1,
            'book_ticker': 1,
            'mark_price': 1,
            'open_interest': 1,
            'funding_rate': 1,
        }

    def _init_sync_session(self) -> None:
        """Initialize synchronous HTTP session."""
        self.session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "DELETE", "PUT"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set headers
        self.session.headers.update({
            'User-Agent': 'FuturesBot/1.0.0',
            'Content-Type': 'application/json'
        })

    async def _get_async_session(self) -> aiohttp.ClientSession:
        """Get or create async HTTP session."""
        if self._async_session is None or self._async_session.closed:
            timeout = aiohttp.ClientTimeout(total=self.session_timeout)
            self._async_session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'User-Agent': 'FuturesBot/1.0.0',
                    'Content-Type': 'application/json'
                }
            )
        return self._async_session

    def _get_signature(self, query_string: str) -> str:
        """Generate HMAC signature for authenticated requests."""
        return hmac.new(
            self.config.binance_api_secret.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()

    def _prepare_request_params(
        self,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Prepare request parameters with signature if needed."""
        if params is None:
            params = {}

        if signed:
            # Add timestamp
            if timestamp is None:
                timestamp = datetime.now(timezone.utc)

            params['timestamp'] = int(timestamp.timestamp() * 1000)

            # Create query string and signature
            query_string = urlencode(sorted(params.items()))
            params['signature'] = self._get_signature(query_string)

        return params

    @performance_monitor("binance_request")
    def _make_request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make synchronous HTTP request to Binance API."""
        # Prepare parameters
        request_params = self._prepare_request_params(params, signed)

        # Select base URL
        if base_url is None:
            base_url = self.spot_base_url

        # Rate limiting
        with self._rate_limiter:
            try:
                # Make request
                response = self.session.request(
                    method=method,
                    url=base_url + url,
                    params=request_params if method == 'GET' else None,
                    json=data if method in ['POST', 'PUT'] else None,
                    timeout=self.session_timeout
                )

                # Handle response
                if response.status_code == 429:
                    retry_after = response.headers.get('Retry-After')
                    raise BinanceRateLimitError(
                        f"Rate limit exceeded: {response.text}",
                        retry_after=int(retry_after) if retry_after else None
                    )
                elif not response.ok:
                    try:
                        error_data = response.json()
                        raise BinanceAPIError(
                            f"API Error: {error_data.get('msg', response.text)}",
                            error_data.get('code')
                        )
                    except ValueError:
                        raise BinanceAPIError(f"HTTP Error: {response.status_code} {response.text}")

                return response.json()

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed: {e}")
                raise BinanceAPIError(f"Request failed: {e}")

    async def _make_async_request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make asynchronous HTTP request to Binance API."""
        # Prepare parameters
        request_params = self._prepare_request_params(params, signed)

        # Select base URL
        if base_url is None:
            base_url = self.spot_base_url

        # Rate limiting
        time_until_available = self._rate_limiter.time_until_available()
        if time_until_available > 0:
            await asyncio.sleep(time_until_available)

        try:
            session = await self._get_async_session()

            async with session.request(
                method=method,
                url=base_url + url,
                params=request_params if method == 'GET' else None,
                json=data if method in ['POST', 'PUT'] else None
            ) as response:
                # Handle response
                if response.status == 429:
                    retry_after = response.headers.get('Retry-After')
                    raise BinanceRateLimitError(
                        f"Rate limit exceeded: {await response.text()}",
                        retry_after=int(retry_after) if retry_after else None
                    )
                elif not response.ok:
                    try:
                        error_data = await response.json()
                        raise BinanceAPIError(
                            f"API Error: {error_data.get('msg', await response.text())}",
                            error_data.get('code')
                        )
                    except ValueError:
                        raise BinanceAPIError(f"HTTP Error: {response.status} {await response.text()}")

                return await response.json()

        except aiohttp.ClientError as e:
            self.logger.error(f"Async request failed: {e}")
            raise BinanceAPIError(f"Async request failed: {e}")

    # Exchange Information
    @retry_with_exponential_backoff(max_attempts=3, base_delay=1.0)
    def get_server_time(self) -> Dict[str, Any]:
        """Get Binance server time."""
        return self._make_request('GET', '/api/v1/time')

    @retry_with_exponential_backoff(max_attempts=3, base_delay=1.0)
    def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange trading rules and symbol information."""
        return self._make_request('GET', '/api/v1/exchangeInfo')

    # Market Data Endpoints
    @retry_with_exponential_backoff(max_attempts=3, base_delay=1.0)
    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[List[Any]]:
        """Get kline/candlestick data."""
        params = {
            'symbol': symbol,
            'interval': interval,
        }

        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        if limit:
            params['limit'] = min(limit, 1500)  # Binance limit

        return self._make_request('GET', '/api/v3/klines', params=params)

    @retry_with_exponential_backoff(max_attempts=3, base_delay=1.0)
    def get_agg_trades(
        self,
        symbol: str,
        from_id: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get compressed/aggregate trades data."""
        params = {'symbol': symbol}

        if from_id:
            params['fromId'] = from_id
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        if limit:
            params['limit'] = min(limit, 1000)  # Binance limit

        return self._make_request('GET', '/api/v3/aggTrades', params=params)

    @retry_with_exponential_backoff(max_attempts=3, base_delay=1.0)
    def get_depth(
        self,
        symbol: str,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get order book depth."""
        params = {'symbol': symbol}

        if limit:
            params['limit'] = limit

        return self._make_request('GET', '/api/v3/depth', params=params)

    @retry_with_exponential_backoff(max_attempts=3, base_delay=1.0)
    def get_trades(
        self,
        symbol: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get recent trades list."""
        params = {'symbol': symbol}

        if limit:
            params['limit'] = min(limit, 1000)  # Binance limit

        return self._make_request('GET', '/api/v1/trades', params=params)

    @retry_with_exponential_backoff(max_attempts=3, base_delay=1.0)
    def get_ticker_24hr(self, symbol: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get 24hr ticker price change statistics."""
        params = {}
        if symbol:
            params['symbol'] = symbol

        return self._make_request('GET', '/api/v3/ticker/24hr', params=params)

    @retry_with_exponential_backoff(max_attempts=3, base_delay=1.0)
    def get_ticker_price(self, symbol: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get symbol price ticker."""
        params = {}
        if symbol:
            params['symbol'] = symbol

        return self._make_request('GET', '/api/v3/ticker/price', params=params)

    @retry_with_exponential_backoff(max_attempts=3, base_delay=1.0)
    def get_book_ticker(self, symbol: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get symbol order book ticker."""
        params = {}
        if symbol:
            params['symbol'] = symbol

        return self._make_request('GET', '/api/v3/ticker/bookTicker', params=params)

    # Futures Market Data Endpoints
    @retry_with_exponential_backoff(max_attempts=3, base_delay=1.0)
    def get_futures_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[List[Any]]:
        """Get futures kline/candlestick data."""
        params = {
            'symbol': symbol,
            'interval': interval,
        }

        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        if limit:
            params['limit'] = min(limit, 1500)  # Binance limit

        return self._make_request('GET', '/fapi/v1/klines', params=params, base_url=self.futures_base_url)

    @retry_with_exponential_backoff(max_attempts=3, base_delay=1.0)
    def get_futures_depth(
        self,
        symbol: str,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get futures order book depth."""
        params = {'symbol': symbol}

        if limit:
            params['limit'] = limit

        return self._make_request('GET', '/fapi/v1/depth', params=params, base_url=self.futures_base_url)

    @retry_with_exponential_backoff(max_attempts=3, base_delay=1.0)
    def get_futures_trades(
        self,
        symbol: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get recent futures trades list."""
        params = {'symbol': symbol}

        if limit:
            params['limit'] = min(limit, 1000)  # Binance limit

        return self._make_request('GET', '/fapi/v1/trades', params=params, base_url=self.futures_base_url)

    @retry_with_exponential_backoff(max_attempts=3, base_delay=1.0)
    def get_futures_agg_trades(
        self,
        symbol: str,
        from_id: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get compressed/aggregate futures trades data."""
        params = {'symbol': symbol}

        if from_id:
            params['fromId'] = from_id
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        if limit:
            params['limit'] = min(limit, 1000)  # Binance limit

        return self._make_request('GET', '/fapi/v1/aggTrades', params=params, base_url=self.futures_base_url)

    @retry_with_exponential_backoff(max_attempts=3, base_delay=1.0)
    def get_mark_price(self, symbol: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get mark price and funding rate."""
        params = {}
        if symbol:
            params['symbol'] = symbol

        return self._make_request('GET', '/fapi/v1/premiumIndex', params=params, base_url=self.futures_base_url)

    @retry_with_exponential_backoff(max_attempts=3, base_delay=1.0)
    def get_funding_rate(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get funding rate history."""
        params = {}
        if symbol:
            params['symbol'] = symbol
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        if limit:
            params['limit'] = min(limit, 1000)  # Binance limit

        return self._make_request('GET', '/fapi/v1/fundingRate', params=params, base_url=self.futures_base_url)

    @retry_with_exponential_backoff(max_attempts=3, base_delay=1.0)
    def get_open_interest(
        self,
        symbol: Optional[str] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get present open interest of a specific symbol."""
        params = {}
        if symbol:
            params['symbol'] = symbol

        return self._make_request('GET', '/fapi/v1/openInterest', params=params, base_url=self.futures_base_url)

    @retry_with_exponential_backoff(max_attempts=3, base_delay=1.0)
    def get_open_interest_hist(
        self,
        symbol: str,
        period: str = "5m",
        limit: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get open interest history."""
        params = {
            'symbol': symbol,
            'period': period,
        }

        if limit:
            params['limit'] = min(limit, 500)  # Binance limit
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)

        return self._make_request('GET', '/fapi/v1/openInterestHist', params=params, base_url=self.futures_base_url)

    @retry_with_exponential_backoff(max_attempts=3, base_delay=1.0)
    def get_futures_24hr_ticker(self, symbol: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get 24hr ticker price change statistics for futures."""
        params = {}
        if symbol:
            params['symbol'] = symbol

        return self._make_request('GET', '/fapi/v1/ticker/24hr', params=params, base_url=self.futures_base_url)

    # User Account Data (requires authentication)
    @retry_with_exponential_backoff(max_attempts=3, base_delay=1.0)
    def get_account_info(self) -> Dict[str, Any]:
        """Get current account information."""
        return self._make_request('GET', '/api/v3/account', signed=True)

    @retry_with_exponential_backoff(max_attempts=3, base_delay=1.0)
    def get_futures_account_info(self) -> Dict[str, Any]:
        """Get current futures account information."""
        return self._make_request('GET', '/fapi/v2/account', signed=True, base_url=self.futures_base_url)

    @retry_with_exponential_backoff(max_attempts=3, base_delay=1.0)
    def get_futures_position_info(self, symbol: Optional[str] = None) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Get current futures position information."""
        params = {}
        if symbol:
            params['symbol'] = symbol

        return self._make_request('GET', '/fapi/v2/positionRisk', params=params, signed=True, base_url=self.futures_base_url)

    # Async versions of key methods
    async def get_klines_async(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[List[Any]]:
        """Async version of get_klines."""
        params = {
            'symbol': symbol,
            'interval': interval,
        }

        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        if limit:
            params['limit'] = min(limit, 1500)  # Binance limit

        return await self._make_async_request('GET', '/api/v3/klines', params=params)

    async def get_futures_depth_async(
        self,
        symbol: str,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Async version of get_futures_depth."""
        params = {'symbol': symbol}

        if limit:
            params['limit'] = limit

        return await self._make_async_request('GET', '/fapi/v1/depth', params=params, base_url=self.futures_base_url)

    # Utility methods
    def close(self) -> None:
        """Close HTTP session."""
        if self.session:
            self.session.close()

    async def aclose(self) -> None:
        """Close async HTTP session."""
        if self._async_session and not self._async_session.closed:
            await self._async_session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.aclose()