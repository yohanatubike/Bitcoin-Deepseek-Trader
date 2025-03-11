# Bitcoin Trading Bot

This trading bot fetches real-time BTC/USDT market data from Binance testnet, enriches it with technical, sentiment, order book, and macro indicators, and sends it to DeepSeek API for trade signal predictions. The bot executes trades based on the signals received.

## Features

- Connects to Binance Testnet API for real-time market data
- Calculates bid-ask spread and order imbalance from order book
- Adds technical indicators, sentiment data, and macroeconomic factors
- Sends enriched data to DeepSeek API for trade predictions
- Executes trades based on predictions with confidence above threshold
- Implements risk management with stop-loss and take-profit orders
- Adjusts position sizing based on market volatility
- Limits to a maximum of 3 concurrent trades for better risk management

## Prerequisites

- Python 3.8+
- Binance Testnet API key and secret
- DeepSeek API key

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/bitcoin-trading-bot.git
cd bitcoin-trading-bot
```

2. Install the dependencies:
```
pip install -r requirements.txt
```

3. Create a configuration file:
```
cp config.ini.example config.ini
```

4. Edit `config.ini` with your API keys and settings.

5. Set up your Binance Testnet API keys:
   - Follow the instructions in [BINANCE_SETUP.md](BINANCE_SETUP.md) to create Binance Testnet API keys
   - Make sure your API keys have the correct permissions

## Usage

Run the trading bot:

```
python main.py
```

## Troubleshooting

### Authentication Errors

If you see a `401 Unauthorized` error when connecting to Binance:
- Verify that your API key and secret are correct
- Generate new Testnet API keys (they expire periodically)
- Ensure you're using Testnet keys with the Testnet setting (`use_testnet = True`)

### DeepSeek API Connection Issues

If you encounter errors connecting to DeepSeek API:
- Verify your DeepSeek API key
- Check the API URL in your config file (should be `https://api.deepseek.com`)
- Make sure the model name is correct (default is `deepseek-chat`)

### Other Issues

- Check the log files in the `logs` directory for detailed error messages
- Ensure your internet connection is stable
- Verify that all required modules are installed (`pip install -r requirements.txt`)

## Configuration

The bot is configured using `config.ini`:

- `binance` section:
  - `api_key`: Your Binance API key
  - `api_secret`: Your Binance API secret
  - `use_testnet`: Whether to use testnet (True/False)

- `deepseek` section:
  - `api_key`: Your DeepSeek API key
  - `api_url`: DeepSeek API URL
  - `model_name`: DeepSeek model name to use (default: deepseek-chat)

- `trading` section:
  - `symbol`: Trading pair symbol (e.g., BTCUSDT)
  - `confidence_threshold`: Minimum confidence required to execute trades (0-1)
  - `run_interval_seconds`: Time between trading cycles (seconds)

## Position Management

The bot implements several risk management features:

- **Maximum Concurrent Trades**: The bot will never have more than 3 active trades at any time
- **Confidence Threshold**: Trades are only executed when the AI's confidence exceeds the threshold
- **Stop-Loss and Take-Profit**: Each trade automatically includes risk management orders
- **Position Sizing**: Trade size is adjusted based on market volatility
- **Graceful Shutdown**: Option to close all positions when stopping the bot

## Data Payload Format

The bot sends the following JSON payload to DeepSeek API:

```json
{
  "symbol": "BTCUSDT",
  "timestamp": 1712345678,
  "timeframes": {
    "5m": {
      "price": {
        "open": 64321.50,
        "high": 64500.00,
        "low": 64100.00,
        "close": 64250.75,
        "volume": 1500.5
      },
      "indicators": {
        "WVO": 0.0025,
        "ARSI": 18.7,
        "VWIO": 0.75
      }
    },
    "1h": {
      "price": {
        "open": 64200.00,
        "high": 64750.00,
        "low": 64050.00,
        "close": 64400.25,
        "volume": 7200.0
      },
      "indicators": {
        "Hourly_High_Low_Percentile": 0.15,
        "Hourly_Volume_Momentum": 1.2,
        "MACD_Histogram": -0.005,
        "Parabolic_SAR": 64230.00,
        "EMA_50_200_Crossover": "Bearish"
      }
    }
  },
  "order_book": {
    "bid_ask_spread": 1.25,
    "order_imbalance": 0.67
  },
  "sentiment": {
    "funding_rate": -0.0052,
    "fear_greed_index": 40,
    "open_interest": 2.3e9,
    "whale_activity": {
      "inflow": 400.5,
      "outflow": 1200.8
    }
  },
  "macro_factors": {
    "exchange_reserves": 2.1e6,
    "btc_hash_rate": 315e12,
    "fomc_event_impact": "Neutral"
  }
}
```

## Expected DeepSeek API Response

The DeepSeek API is expected to return a response in the following format:

```json
{
  "symbol": "BTCUSDT",
  "timestamp": 1712345678,
  "prediction": {
    "action": "BUY",
    "confidence": 0.92,
    "stop_loss": 64050.00,
    "take_profit": 65000.00
  }
}
```

## Trade Execution Rules

- BUY: If action = BUY with confidence > threshold, place a long order
- SELL: If action = SELL with confidence > threshold, enter a short position
- HOLD: If confidence < threshold or action = HOLD, do nothing
- Position size is adjusted based on volatility (1 / (1 + WVO))

## Disclaimer

This is a proof-of-concept and should not be used for real trading without proper risk management and testing. The bot uses mock data for some indicators in this demonstration version.

## License

MIT 