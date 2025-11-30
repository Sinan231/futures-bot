"""
Real-time WebSocket data streams for Binance trading data.

Handles multiple WebSocket connections with reconnection logic,
data buffering, timestamp synchronization, and comprehensive
error handling for all required market data streams.
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from dataclasses import dataclass, asdict, field
import asyncio
import websockets
from collections import defaultdict, deque
import threading

from .binance_client import BinanceClient
from ..utils.logging import get_logger, log_trading_event, performance_monitor
from ..utils.helpers import CircularBuffer, clamp, normalize_timestamp
from ..utils.config import DataConfig


class StreamType(Enum):
    """Enumeration of supported WebSocket stream types."""
    KLINES = "klines"
    AGG_TRADES = "aggTrades"
    TRADES = "trades"
    DEPTH = "depth"
    BOOK_TICKER = "bookTicker"
    MARK_PRICE = "markPrice"
    FUNDING_RATE = "fundingRate"
    OPEN_INTEREST = "openInterest"
    LIQUIDATION_ORDERS = "liquidationOrders"  # If available


class WebSocketState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class StreamSubscription:
    """Represents a WebSocket stream subscription."""
    stream_type: StreamType
    symbol: str
    interval: Optional[str] = None  # For klines
    levels: Optional[int] = None    # For order book depth
    callback: Optional[Callable[[Dict[str, Any]], None]] = None
    active: bool = True
    reconnect_attempts: int = 0
    last_data_time: Optional[datetime] = None
    data_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None


@dataclass
class StreamMessage:
    """Represents a WebSocket message with metadata."""
    stream_type: StreamType
    symbol: str
    data: Dict[str, Any]
    timestamp: datetime
    interval: Optional[str] = None
    sequence_number: Optional[int] = None


@dataclass
class StreamStats:
    """Statistics for a WebSocket stream."""
    stream_type: StreamType
    symbol: str
    messages_received: int = 0
    messages_per_second: float = 0.0
    last_message_time: Optional[datetime] = None
    reconnect_count: int = 0
    total_downtime_seconds: float = 0.0
    average_message_size_bytes: float = 0.0
    error_count: int = 0


class WebSocketConnection:
    """Individual WebSocket connection for a specific stream."""

    def __init__(
        self,
        subscription: StreamSubscription,
        config: DataConfig,
        max_buffer_size: int = 1000
    ):
        """Initialize WebSocket connection.

        Args:
            subscription: Stream subscription details
            config: Data configuration
            max_buffer_size: Maximum number of messages to buffer
        """
        self.subscription = subscription
        self.config = config
        self.logger = get_logger(f"{__name__}.{subscription.symbol}.{subscription.stream_type.value}")

        self.max_buffer_size = max_buffer_size
        self.buffer = CircularBuffer(max_buffer_size)
        self.state = WebSocketState.DISCONNECTED

        # Connection management
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.task: Optional[asyncio.Task] = None
        self.ping_task: Optional[asyncio.Task] = None

        # Statistics tracking
        self.stats = StreamStats(
            stream_type=subscription.stream_type,
            symbol=subscription.symbol
        )
        self.last_stats_update = time.time()
        self.connection_start_time: Optional[float] = None

        # Message queue for processing
        self.message_queue = asyncio.Queue(maxsize=max_buffer_size)
        self.processing_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Establish WebSocket connection."""
        try:
            self.state = WebSocketState.CONNECTING
            stream_url = self._build_stream_url()
            self.logger.info(f"Connecting to {stream_url}")

            # Connect with timeout
            self.websocket = await asyncio.wait_for(
                websockets.connect(
                    stream_url,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=10,
                    max_queue=self.max_buffer_size
                ),
                timeout=30.0
            )

            self.state = WebSocketState.CONNECTED
            self.connection_start_time = time.time()
            self.logger.info("WebSocket connection established")

            # Start message processing
            self.processing_task = asyncio.create_task(self._process_messages())
            self.ping_task = asyncio.create_task(self._ping_loop())

            return True

        except Exception as e:
            self.state = WebSocketState.ERROR
            self.subscription.error_count += 1
            self.subscription.last_error = str(e)
            self.logger.error(f"Failed to connect: {e}")
            return False

    async def disconnect(self) -> None:
        """Gracefully disconnect WebSocket."""
        self.state = WebSocketState.STOPPED
        self.subscription.active = False

        # Cancel tasks
        if self.task:
            self.task.cancel()
        if self.ping_task:
            self.ping_task.cancel()
        if self.processing_task:
            self.processing_task.cancel()

        # Close websocket
        if self.websocket:
            try:
                await self.websocket.close()
                self.logger.info("WebSocket disconnected gracefully")
            except Exception as e:
                self.logger.warning(f"Error during disconnect: {e}")

        # Update stats
        if self.connection_start_time:
            self.stats.total_downtime_seconds += time.time() - self.connection_start_time

    def _build_stream_url(self) -> str:
        """Build WebSocket stream URL."""
        base_url = "wss://stream.binance.com:9443/ws"

        if self.config.binance_testnet:
            base_url = "wss://testnet.binance.vision/ws"

        # Build stream path based on type
        stream_paths = []

        if self.subscription.stream_type == StreamType.KLINES:
            stream_paths.append(f"{self.subscription.symbol.lower()}@kline_{self.subscription.interval}")
        elif self.subscription.stream_type == StreamType.AGG_TRADES:
            stream_paths.append(f"{self.subscription.symbol.lower()}@aggTrade")
        elif self.subscription.stream_type == StreamType.TRADES:
            stream_paths.append(f"{self.subscription.symbol.lower()}@trade")
        elif self.subscription.stream_type == StreamType.DEPTH:
            stream_paths.append(f"{self.subscription.symbol.lower()}@depth{self.subscription.levels or 20}")
        elif self.subscription.stream_type == StreamType.BOOK_TICKER:
            stream_paths.append(f"{self.subscription.symbol.lower()}@bookTicker")
        elif self.subscription.stream_type == StreamType.MARK_PRICE:
            stream_paths.append(f"{self.subscription.symbol.lower()}@markPrice")
        elif self.subscription.stream_type == StreamType.FUNDING_RATE:
            stream_paths.append(f"{self.subscription.symbol.lower()}@fundingRate")
        elif self.subscription.stream_type == StreamType.OPEN_INTEREST:
            stream_paths.append(f"{self.subscription.symbol.lower()}@openInterest")
        elif self.subscription.stream_type == StreamType.LIQUIDATION_ORDERS:
            stream_paths.append(f"{self.subscription.symbol.lower()}@liquidationOrders")

        stream_path = "/".join(stream_paths)
        return f"{base_url}/{stream_path}"

    async def _listen_messages(self) -> None:
        """Listen for incoming WebSocket messages."""
        try:
            async for message in self.websocket:
                if self.state != WebSocketState.CONNECTED:
                    break

                self.stats.messages_received += 1
                self.stats.last_message_time = datetime.now(timezone.utc)

                # Add to buffer
                self.buffer.append(message)

                # Add to processing queue
                try:
                    await self.message_queue.put(message)
                except asyncio.QueueFull:
                    self.logger.warning("Message queue is full, dropping message")

                # Update subscription stats
                self.subscription.data_count += 1
                self.subscription.last_data_time = datetime.now(timezone.utc)

                # Periodically update per-second stats
                current_time = time.time()
                if current_time - self.last_stats_update >= 1.0:
                    time_diff = current_time - self.last_stats_update
                    self.stats.messages_per_second = self.stats.messages_received / max(1, time_diff)
                    self.last_stats_update = current_time

        except websockets.exceptions.ConnectionClosed:
            self.logger.info("WebSocket connection closed")
        except Exception as e:
            self.logger.error(f"Error in message listener: {e}")
            self.stats.error_count += 1
            self.subscription.error_count += 1
            self.subscription.last_error = str(e)

    async def _process_messages(self) -> None:
        """Process messages from the queue."""
        while self.state == WebSocketState.CONNECTED:
            try:
                # Get message with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )

                # Parse and process message
                try:
                    parsed_message = json.loads(message)
                    stream_message = self._parse_message(parsed_message)

                    if stream_message and self.subscription.callback:
                        await self._handle_callback(stream_message)

                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON message: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")

                # Update message size stats
                self.stats.average_message_size_bytes = (
                    self.stats.average_message_size_bytes * 0.9 + len(message) * 0.1
                )

            except asyncio.TimeoutError:
                # Normal timeout, continue
                continue
            except asyncio.QueueEmpty:
                # Queue empty, continue
                continue
            except Exception as e:
                self.logger.error(f"Error in message processor: {e}")

    async def _ping_loop(self) -> None:
        """Maintain WebSocket connection with pings."""
        while self.state == WebSocketState.CONNECTED:
            try:
                await asyncio.sleep(30)  # Ping every 30 seconds
                if self.websocket and self.state == WebSocketState.CONNECTED:
                    await self.websocket.ping()
            except Exception as e:
                self.logger.warning(f"Error in ping loop: {e}")
                break

    def _parse_message(self, raw_message: Dict[str, Any]) -> Optional[StreamMessage]:
        """Parse raw WebSocket message into StreamMessage."""
        try:
            # Handle different message formats
            if 'stream' in raw_message:
                # Binance combined stream format
                stream_parts = raw_message['stream'].split('@')
                symbol = stream_parts[0].upper()
                stream_type_str = stream_parts[1]
                data = raw_message['data']
            else:
                # Direct message format
                symbol = self.subscription.symbol
                stream_type_str = self.subscription.stream_type.value
                data = raw_message

            # Determine stream type
            stream_type = self._determine_stream_type(stream_type_str)
            if not stream_type:
                self.logger.warning(f"Unknown stream type: {stream_type_str}")
                return None

            # Extract timestamp from data
            timestamp = self._extract_timestamp(data, stream_type)
            if not timestamp:
                timestamp = datetime.now(timezone.utc)

            # Create stream message
            stream_message = StreamMessage(
                stream_type=stream_type,
                symbol=symbol,
                data=data,
                timestamp=timestamp,
                interval=self.subscription.interval,
                sequence_number=data.get('U') or data.get('u')  # For trades
            )

            return stream_message

        except Exception as e:
            self.logger.error(f"Error parsing message: {e}")
            return None

    def _determine_stream_type(self, stream_type_str: str) -> Optional[StreamType]:
        """Determine StreamType from string."""
        type_mapping = {
            'kline': StreamType.KLINES,
            'aggTrade': StreamType.AGG_TRADES,
            'trade': StreamType.TRADES,
            'depth': StreamType.DEPTH,
            'bookTicker': StreamType.BOOK_TICKER,
            'markPrice': StreamType.MARK_PRICE,
            'fundingRate': StreamType.FUNDING_RATE,
            'openInterest': StreamType.OPEN_INTEREST,
            'liquidationOrders': StreamType.LIQUIDATION_ORDERS
        }

        return type_mapping.get(stream_type_str.split('_')[0])

    def _extract_timestamp(self, data: Dict[str, Any], stream_type: StreamType) -> datetime:
        """Extract timestamp from message data."""
        timestamp_fields = {
            StreamType.KLINES: ['E', 'T'],  # Event time, Kline close time
            StreamType.AGG_TRADES: ['T'],  # Trade time
            StreamType.TRADES: ['T'],  # Trade time
            StreamType.DEPTH: ['E'],  # Event time
            StreamType.BOOK_TICKER: [],  # No timestamp in book ticker
            StreamType.MARK_PRICE: ['E', 'T'],  # Event time, Mark price time
            StreamType.FUNDING_RATE: ['T'],  # Funding rate time
            StreamType.OPEN_INTEREST: ['E'],  # Event time
            StreamType.LIQUIDATION_ORDERS: ['E'],  # Event time
        }

        # Try to find timestamp in data
        for field in timestamp_fields.get(stream_type, []):
            if field in data and data[field]:
                try:
                    return datetime.fromtimestamp(data[field] / 1000, timezone.utc)
                except (ValueError, TypeError):
                    continue

        # Fallback to current time
        return datetime.now(timezone.utc)

    async def _handle_callback(self, stream_message: StreamMessage) -> None:
        """Handle message callback with error handling."""
        try:
            if asyncio.iscoroutinefunction(self.subscription.callback):
                await self.subscription.callback(stream_message)
            else:
                self.subscription.callback(stream_message)
        except Exception as e:
            self.logger.error(f"Error in callback: {e}")

    def get_stats(self) -> StreamStats:
        """Get current stream statistics."""
        # Update per-second rate
        current_time = time.time()
        if self.last_stats_update > 0:
            time_diff = current_time - self.last_stats_update
            if time_diff > 0:
                self.stats.messages_per_second = (
                    self.stats.messages_received / time_diff
                )

        return self.stats

    def get_buffered_messages(self, max_count: int = None) -> List[StreamMessage]:
        """Get buffered messages from the circular buffer."""
        raw_messages = self.buffer.get_latest(max_count or self.max_buffer_size)
        parsed_messages = []

        for raw_msg in raw_messages:
            try:
                parsed_message = self._parse_message(json.loads(raw_msg))
                if parsed_message:
                    parsed_messages.append(parsed_message)
            except Exception:
                continue  # Skip invalid messages

        return parsed_messages


class RealTimeStreamManager:
    """Manages multiple WebSocket connections for real-time market data."""

    def __init__(
        self,
        config: DataConfig,
        max_connections: int = 20,
        reconnection_interval: float = 5.0,
        max_reconnect_attempts: int = 10
    ):
        """Initialize stream manager.

        Args:
            config: Data configuration
            max_connections: Maximum number of concurrent connections
            reconnection_interval: Seconds between reconnection attempts
            max_reconnect_attempts: Maximum reconnection attempts
        """
        self.config = config
        self.logger = get_logger(__name__)

        self.max_connections = max_connections
        self.reconnection_interval = reconnection_interval
        self.max_reconnect_attempts = max_reconnect_attempts

        # Connection management
        self.connections: Dict[str, WebSocketConnection] = {}
        self.subscriptions: List[StreamSubscription] = []

        # State management
        self.running = False
        self.supervisor_task: Optional[asyncio.Task] = None

        # Statistics
        self.total_messages = 0
        self.connection_errors = 0
        self.reconnection_attempts = 0
        self.start_time: Optional[float] = None

        # Event callbacks
        self.message_handlers: Dict[StreamType, List[Callable]] = defaultdict(list)
        self.error_handlers: List[Callable] = []

    async def start(self) -> None:
        """Start the stream manager."""
        if self.running:
            self.logger.warning("Stream manager is already running")
            return

        self.running = True
        self.start_time = time.time()
        self.logger.info("Starting real-time stream manager")

        # Start supervision task
        self.supervisor_task = asyncio.create_task(self._supervise_connections())

        # Establish all connections
        await self._connect_all()

    async def stop(self) -> None:
        """Stop the stream manager and all connections."""
        self.running = False
        self.logger.info("Stopping real-time stream manager")

        # Cancel supervision task
        if self.supervisor_task:
            self.supervisor_task.cancel()

        # Disconnect all connections
        disconnect_tasks = []
        for connection in self.connections.values():
            disconnect_tasks.append(connection.disconnect())

        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)

        # Clear state
        self.connections.clear()
        self.subscriptions.clear()

        self.logger.info("Stream manager stopped")

    def subscribe(
        self,
        stream_type: StreamType,
        symbol: str,
        callback: Optional[Callable[[StreamMessage], None]] = None,
        interval: Optional[str] = None,
        levels: Optional[int] = None
    ) -> str:
        """Subscribe to a new stream.

        Args:
            stream_type: Type of stream to subscribe to
            symbol: Trading symbol
            callback: Optional callback for stream messages
            interval: Time interval (for klines)
            levels: Order book depth levels

        Returns:
            Subscription ID
        """
        # Create subscription
        subscription = StreamSubscription(
            stream_type=stream_type,
            symbol=symbol.upper(),
            interval=interval,
            levels=levels,
            callback=callback
        )

        # Add to subscriptions
        self.subscriptions.append(subscription)

        # Register message handler
        if callback:
            self.message_handlers[stream_type].append(callback)

        # Create connection if manager is running
        if self.running:
            asyncio.create_task(self._create_connection(subscription))

        subscription_id = f"{stream_type.value}_{symbol}_{interval or 'default'}"
        self.logger.info(f"Subscribed to {subscription_id}")

        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from a stream.

        Args:
            subscription_id: ID returned by subscribe()

        Returns:
            True if unsubscribed successfully
        """
        # Find and remove subscription
        for i, subscription in enumerate(self.subscriptions):
            current_id = f"{subscription.stream_type.value}_{subscription.symbol}_{subscription.interval or 'default'}"
            if current_id == subscription_id:
                subscription.active = False
                del self.subscriptions[i]

                # Disconnect corresponding connection
                connection_key = self._get_connection_key(subscription)
                if connection_key in self.connections:
                    asyncio.create_task(self.connections[connection_key].disconnect())
                    del self.connections[connection_key]

                self.logger.info(f"Unsubscribed from {subscription_id}")
                return True

        return False

    def add_message_handler(self, stream_type: StreamType, handler: Callable) -> None:
        """Add a message handler for a specific stream type."""
        self.message_handlers[stream_type].append(handler)

    def add_error_handler(self, handler: Callable) -> None:
        """Add an error handler."""
        self.error_handlers.append(handler)

    def get_latest_messages(
        self,
        stream_type: Optional[StreamType] = None,
        symbol: Optional[str] = None,
        max_count: int = 100
    ) -> List[StreamMessage]:
        """Get latest messages from buffer.

        Args:
            stream_type: Filter by stream type
            symbol: Filter by symbol
            max_count: Maximum number of messages to return

        Returns:
            List of stream messages
        """
        all_messages = []

        for connection in self.connections.values():
            if stream_type and connection.subscription.stream_type != stream_type:
                continue
            if symbol and connection.subscription.symbol != symbol:
                continue

            messages = connection.get_buffered_messages(max_count)
            all_messages.extend(messages)

        # Sort by timestamp
        all_messages.sort(key=lambda x: x.timestamp, reverse=True)

        return all_messages[:max_count]

    def get_connection_stats(self) -> Dict[str, StreamStats]:
        """Get statistics for all connections."""
        return {
            key: conn.get_stats()
            for key, conn in self.connections.items()
        }

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics."""
        uptime = time.time() - self.start_time if self.start_time else 0
        active_connections = len([c for c in self.connections.values()
                               if c.state == WebSocketState.CONNECTED])

        return {
            'uptime_seconds': uptime,
            'total_messages': self.total_messages,
            'active_connections': active_connections,
            'total_subscriptions': len(self.subscriptions),
            'connection_errors': self.connection_errors,
            'reconnection_attempts': self.reconnection_attempts,
            'average_messages_per_second': self.total_messages / max(1, uptime)
        }

    async def _connect_all(self) -> None:
        """Connect all subscriptions."""
        for subscription in self.subscriptions:
            if subscription.active:
                await self._create_connection(subscription)

    async def _create_connection(self, subscription: StreamSubscription) -> None:
        """Create and manage a WebSocket connection."""
        if len(self.connections) >= self.max_connections:
            self.logger.warning(f"Maximum connections ({self.max_connections}) reached")
            return

        connection_key = self._get_connection_key(subscription)
        if connection_key in self.connections:
            self.logger.warning(f"Connection already exists for {connection_key}")
            return

        # Create connection
        connection = WebSocketConnection(subscription, self.config)
        self.connections[connection_key] = connection

        # Start connection with retry logic
        await self._manage_connection_lifecycle(connection, subscription)

    async def _manage_connection_lifecycle(
        self,
        connection: WebSocketConnection,
        subscription: StreamSubscription
    ) -> None:
        """Manage connection lifecycle with reconnection."""
        while subscription.active and subscription.reconnect_attempts < self.max_reconnect_attempts:
            try:
                # Attempt to connect
                connected = await connection.connect()

                if connected:
                    # Listen for messages
                    await connection._listen_messages()
                else:
                    # Connection failed, update stats
                    self.connection_errors += 1

            except Exception as e:
                self.logger.error(f"Connection error for {subscription.symbol}: {e}")
                self.connection_errors += 1

            # Check if we should reconnect
            if subscription.active and subscription.reconnect_attempts < self.max_reconnect_attempts:
                subscription.reconnect_attempts += 1
                self.reconnection_attempts += 1

                self.logger.info(
                    f"Reconnecting in {self.reconnection_interval}s "
                    f"(attempt {subscription.reconnect_attempts}/{self.max_reconnect_attempts})"
                )

                await asyncio.sleep(self.reconnection_interval)
            else:
                break

        # Remove connection if failed or inactive
        connection_key = self._get_connection_key(subscription)
        if connection_key in self.connections:
            del self.connections[connection_key]

        self.logger.warning(
            f"Connection for {subscription.symbol} stopped after "
            f"{subscription.reconnect_attempts} attempts"
        )

    async def _supervise_connections(self) -> None:
        """Supervise all connections and handle issues."""
        while self.running:
            try:
                # Check for stale connections
                current_time = datetime.now(timezone.utc)
                for connection_key, connection in list(self.connections.items()):
                    if connection.state == WebSocketState.CONNECTED:
                        # Check if we received data recently (within last 5 minutes)
                        if connection.subscription.last_data_time:
                            time_since_last_data = (
                                current_time - connection.subscription.last_data_time
                            ).total_seconds()
                            if time_since_last_data > 300:  # 5 minutes
                                self.logger.warning(
                                    f"No data from {connection_key} for {time_since_last_data}s, reconnecting"
                                )
                                asyncio.create_task(connection.disconnect())
                                connection.subscription.reconnect_attempts = 0

                # Periodic health check
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Error in supervision task: {e}")
                await asyncio.sleep(10)  # Wait before retrying

    def _get_connection_key(self, subscription: StreamSubscription) -> str:
        """Generate unique connection key for subscription."""
        interval_part = f"_{subscription.interval}" if subscription.interval else ""
        return f"{subscription.stream_type.value}_{subscription.symbol}{interval_part}"