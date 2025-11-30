# Production-Ready Trading Signal System

A comprehensive production-ready trading signal system that fetches multi-source market data from Binance, creates feature-rich datasets, trains machine learning models, and emits real-time trading signals with proper risk management.

## Features

- **Multi-source Data Fetching**: Historical and real-time data from Binance (spot & perpetual)
- **Feature Engineering**: 50+ technical indicators across multiple timeframes
- **ML Pipeline**: Multiple algorithms (LightGBM, XGBoost, LSTM, etc.) with hyperparameter tuning
- **Backtesting**: Event-driven backtesting with realistic market conditions
- **Real-time Signals**: Confidence-based trading signals with HMAC authentication
- **Risk Management**: Volatility-adjusted position sizing and risk controls
- **Monitoring**: Model drift detection and performance monitoring
- **Production Ready**: Dockerized with comprehensive testing and ops support

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Binance API credentials (for data access)

### Installation

1. Clone and build:
```bash
git clone <repository>
cd futures-bot
docker-compose build
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API credentials
```

3. Download historical data:
```bash
python scripts/download_data.py --pair BTCUSDT --timeframes 1m 5m 1h 4h --months 8
```

4. Train models:
```bash
python scripts/train.py --pair BTCUSDT --config config/model_config.yaml
```

5. Start real-time signals:
```bash
python scripts/serve.py --model artifacts/models/v1.0.0 --config config/trading_config.yaml
```

## System Architecture

- **Data Layer**: Historical fetching, real-time WebSocket streams, data validation
- **Feature Pipeline**: Technical indicators, order book features, time-based features
- **ML Pipeline**: Label generation, model training, validation, and selection
- **Backtesting**: Event-driven simulation with realistic market conditions
- **Signal Service**: Real-time inference with risk management and authentication
- **Monitoring**: Performance tracking, drift detection, and alerting

## Configuration

Key configuration files:
- `config/model_config.yaml`: Model training parameters
- `config/trading_config.yaml`: Trading and risk management settings
- `.env`: API keys and secret credentials

## Default Settings

- **History**: 7 months of historical data
- **Timeframes**: 1m, 5m, 1h, 4h, daily
- **Prediction Horizon**: 4 hours
- **Signal Confidence Threshold**: 0.80
- **Max Leverage**: 20x (recommended 10x)
- **Risk per Trade**: 0.5% of equity

## Documentation

- [API Documentation](docs/api.md)
- [Configuration Reference](docs/config.md)
- [Deployment Guide](docs/deployment.md)
- [Operations Manual](docs/operations.md)

## Testing

```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Run with coverage
pytest --cov=src tests/
```

## Security

- API keys stored in environment variables only
- HMAC-signed signals for authenticity
- No order execution (signal-only system)
- Comprehensive audit logging
- Rate limiting and input validation

## Performance

- **Signal Latency**: <5 seconds
- **Memory Usage**: <8GB normal operations
- **Processing Throughput**: >1000 ticks/second
- **System Uptime**: >99.5%

## License

[License information]

## Support

For issues and questions:
- Create GitHub issue
- Check troubleshooting guide
- Review operations manual