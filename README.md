# Bitcoin Trading Bot

This trading bot fetches real-time BTC/USDT market data from Binance testnet, enriches it with technical, sentiment, order book, and macro indicators, and sends it to DeepSeek API for trade signal predictions. The bot executes trades based on the signals received.

## Features

- Connects to Binance Testnet API for real-time market data
- Calculates bid-ask spread, depth imbalance, and large order detection from order book
- Adds technical indicators, sentiment data, correlation analysis, and macroeconomic factors
- Robust data validation and error handling for various data formats
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

```plaintext
git clone https://github.com/yourusername/bitcoin-trading-bot.git
cd bitcoin-trading-bot
```

2. Install the dependencies:

```plaintext
pip install -r requirements.txt
```

3. Create a configuration file:

```plaintext
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

### Common Issues and Solutions

#### Data Format Issues

- **KeyError in `_add_default_sentiment`**: If you see a KeyError related to klines data, ensure your data feed is providing klines in the expected format. The system now handles multiple formats (list and dictionary) but may require adjustment for custom data providers.

- **Order Book Data Issues**: The system validates order book data before processing. If you see warnings about invalid order book formats, check your data source and ensure bids and asks are properly formatted.

#### API Connection Issues

- **Binance API Connection Failures**: If you encounter connection errors to Binance API, check:
  - Your API keys are correctly set in the configuration
  - Network connectivity is stable
  - Binance API is not under maintenance

- **Santiment API Not Available**: The system will use default sentiment values when Santiment API is unavailable. To use real sentiment data, ensure your Santiment API key is configured correctly.

#### Performance Issues

- **High CPU Usage**: If the bot is consuming excessive CPU resources, consider:
  - Reducing the number of indicators calculated
  - Increasing the interval between trading cycles
  - Running on a more powerful machine

- **Memory Leaks**: If you notice increasing memory usage over time, restart the bot periodically and report the issue.

### Debugging Steps

1. **Enable Debug Logging**: Set `LOG_LEVEL` to `DEBUG` in your configuration to get more detailed logs.

2. **Check Log Files**: Examine `trading_bot.log` for detailed error messages and warnings.

3. **Test API Connections**: Use the test endpoints provided by Binance to verify your API connection before trading.

4. **Validate Data**: Use the `--validate` flag when starting the bot to perform validation checks on incoming data without executing trades.

5. **Contact Support**: For persistent issues, contact support with your log files and a description of the problem.

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
- **Position Sizing**: Trade size is adjusted based on market volatility and DeepSeek API recommendations
- **Dynamic Risk Management**: Ability to close positions partially, modify stop-loss and take-profit levels based on market conditions
- **Order Book Analysis**: Position entries consider depth imbalance and potential price impact
- **Correlation Awareness**: Position sizing accounts for correlations with other assets to manage systematic risk
- **Graceful Shutdown**: Option to close all positions when stopping the bot

The trade executor now supports:

- Closing positions (full or partial)
- Modifying stop-loss and take-profit levels for active positions
- Emergency market closing of positions when needed
- Position sizing based on market volatility, correlation analysis, and AI recommendations

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
        "MACD": {
          "macd_line": 150.25,
          "signal_line": 155.30,
          "histogram": -5.05
        },
        "Parabolic_SAR": "Bearish",
        "EMA_50_200_Crossover": "Bearish",
        "Ichimoku": "Neutral",
        "Fibonacci_Levels": {
          "0.236": 63950.25,
          "0.382": 63700.50,
          "0.5": 63500.75,
          "0.618": 63300.25,
          "0.786": 63050.00
        },
        "Volume_Profile_POC": 64275.50,
        "Pivot_Points": {
          "R3": 65200.00,
          "R2": 64900.00,
          "R1": 64650.00,
          "PP": 64400.00,
          "S1": 64150.00,
          "S2": 63900.00,
          "S3": 63650.00
        },
        "Williams_R": -75.5,
        "VWAP": 64325.75,
        "CVD": 850.5,
        "ADX": 22.3
      }
    }
  },
  "order_book": {
    "bid_ask_spread": 1.25,
    "depth_imbalance": 0.15,
    "bid_ask_ratio": 1.05,
    "large_orders": {
      "bids": [
        {"price": 64100.00, "quantity": 15.0},
        {"price": 64000.00, "quantity": 25.0}
      ],
      "asks": [
        {"price": 64500.00, "quantity": 20.0}
      ],
      "count": 3
    },
    "price_impact": {
      "buy": {
        "1": 0.02,
        "5": 0.12,
        "10": 0.25
      },
      "sell": {
        "1": 0.03,
        "5": 0.15,
        "10": 0.30
      }
    }
  },
  "sentiment": {
    "funding_rate": -0.0052,
    "fear_greed_index": 40,
    "social_sentiment": {
      "score": 0.65,
      "trend": "Bullish",
      "volume": 3500,
      "change_24h": 0.15
    },
    "news_sentiment": {
      "score": 0.45,
      "trend": "Neutral",
      "top_topics": ["Regulation", "Institutional Adoption"]
    },
    "open_interest": 2.3e9,
    "whale_activity": {
      "inflow": 400.5,
      "outflow": 1200.8,
      "netflow": -800.3,
      "large_transactions": 25
    }
  },
  "correlations": {
    "BTC_ETH": {
      "value": 0.82,
      "description": "Strong positive correlation"
    },
    "BTC_SP500": {
      "value": 0.35,
      "description": "Moderate positive correlation"
    },
    "BTC_Gold": {
      "value": -0.15,
      "description": "Weak negative correlation"
    },
    "BTC_DXY": {
      "value": -0.45,
      "description": "Moderate negative correlation"
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
    "take_profit": 65000.00,
    "position_sizing": 0.25,
    "reasoning": [
      "Strong buy pressure indicated by depth imbalance (0.15)",
      "Price is near Fibonacci support level (0.382)",
      "MACD histogram showing potential reversal",
      "Social sentiment trending bullish with increasing volume"
    ],
    "risk_assessment": {
      "volatility": "Medium",
      "market_liquidity": "High",
      "slippage_risk": "Low",
      "correlation_risk": "Medium"
    },
    "timeframe": {
      "entry_window": "Next 2 hours",
      "expected_hold_time": "24-48 hours"
    }
  }
}
```

The enhanced response includes the traditional action, confidence, stop-loss and take-profit levels, but also provides:

1. Position sizing recommendation based on risk analysis
2. Reasoning with key indicators that influenced the decision
3. Risk assessment across multiple dimensions
4. Timeframe guidance for entry and expected hold duration

This richer response format allows the trading bot to make more nuanced decisions and provides transparency in the AI's reasoning process.

## Trade Execution Rules

- BUY: If action = BUY with confidence > threshold, place a long order
- SELL: If action = SELL with confidence > threshold, enter a short position
- HOLD: If confidence < threshold or action = HOLD, do nothing
- Position size is adjusted based on volatility (1 / (1 + WVO))

## Implemented Indicators

The trading bot uses a wide range of technical indicators to inform its trading decisions. Recent updates have added the following indicators to enhance the bot's analysis capabilities:

### 1-Hour Timeframe Indicators

- Hourly High Low Percentile - Position of current price within the recent high-low range
- Hourly Volume Momentum - Momentum of trading volume
- MACD & MACD Histogram - Trend momentum indicator and signal line crossover detection
- Parabolic SAR - Trend direction and potential reversal points
- EMA 50/200 Crossover - Long-term trend identification through moving average crossovers
- Ichimoku Cloud - Comprehensive trend analysis system
- Fibonacci Levels - Key retracement and extension levels for support/resistance
- Volume Profile POC (Point of Control) - Price level with highest traded volume
- Pivot Points - Key support and resistance levels
- Williams %R - Momentum oscillator to identify overbought/oversold conditions
- VWAP (Volume Weighted Average Price) - Average price weighted by volume
- CVD (Cumulative Volume Delta) - Accumulation/distribution of volume
- ADX (Average Directional Index) - Trend strength measurement

### Order Book Analysis

- Depth Imbalance - Imbalance between buy and sell orders at various price levels
- Large Orders - Detection of significant market orders that may indicate whale activity
- Price Impact Analysis - Estimation of price impact from large market orders

### Market Sentiment Indicators

- Social Media Sentiment - Sentiment analysis from social platforms
- News Sentiment - Sentiment derived from financial news sources
- Whale Activity - Tracking of large holders' actions
- Fear & Greed Index - Market sentiment gauge

### Correlation Analysis

- BTC-ETH Correlation - Relationship between Bitcoin and Ethereum
- BTC-SP500 Correlation - Bitcoin's correlation with the S&P 500 index
- BTC-Gold Correlation - Bitcoin's correlation with gold
- BTC-DXY Correlation - Bitcoin's correlation with the US Dollar Index

These indicators provide a comprehensive view of market conditions across multiple timeframes and data sources, enabling the bot to make more informed trading decisions with higher confidence.

## Disclaimer

This is a proof-of-concept and should not be used for real trading without proper risk management and testing. The bot uses mock data for some indicators in this demonstration version.

## License

MIT
