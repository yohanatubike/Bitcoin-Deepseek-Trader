"""
Module for interacting with DeepSeek API to get trade signal predictions
"""

import json
import logging
import time
from typing import Dict, Any, Optional
import random
import traceback
import pandas as pd
import concurrent.futures

# Import OpenAI SDK for DeepSeek API compatibility
from openai import OpenAI
from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)

class DeepSeekAPI:
    def __init__(self, api_key: str, api_url: str = "https://api.deepseek.com", model_name: str = "deepseek-chat"):
        """
        Initialize the DeepSeek API client
        
        Args:
            api_key: DeepSeek API key
            api_url: DeepSeek API base URL (default: "https://api.deepseek.com")
            model_name: DeepSeek model name (default: "deepseek-chat")
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Remove any trailing slashes from api_url
        self.api_url = api_url.rstrip('/')
        
        # Remove any path segments from api_url - we just need the base domain
        # The OpenAI SDK will append /v1/chat/completions automatically
        if "/v1" in self.api_url:
            self.api_url = self.api_url.split("/v1")[0]
        
        logger.info(f"Initializing DeepSeek API client with base URL: {self.api_url}")
        logger.info(f"Using model: {self.model_name}")
        
        # Initialize OpenAI compatible client for DeepSeek
        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url=self.api_url
            )
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def get_prediction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get trade signal prediction from DeepSeek API
        
        Args:
            data: Enriched market data
            
        Returns:
            Prediction response
        """
        logger.info("Getting prediction from DeepSeek API")
        
        try:
            # Log the complete data being sent
            logger.info("Complete data payload being sent to DeepSeek:")
            #logger.info(json.dumps(data, indent=2, default=str))

            # Format market data into a detailed prompt for the AI
            prompt = self._format_prompt(data)
            
            # Log the formatted prompt
            logger.info("Formatted prompt being sent to DeepSeek:")
            logger.info(prompt)
            
            # Print request details for debugging
            logger.info(f"Sending request to {self.api_url} using model {self.model_name}")
            
            # Send request to DeepSeek API using OpenAI SDK
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2  # Lower temperature for more deterministic responses
            )
            
            # Extract the response content
            response_content = response.choices[0].message.content
            
            # Parse the JSON response
            try:
                prediction = json.loads(response_content)
                
                # Log the prediction
                logger.info(f"Received prediction from DeepSeek API: {json.dumps(prediction)}")
                
                # Validate prediction format
                self._validate_prediction(prediction)
                
                # Save prediction for backtesting
                self._save_prediction(data, prediction)
                
                return prediction
                
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON response from DeepSeek API: {response_content}")
                raise ValueError("Invalid JSON response from DeepSeek API")
                
        except Exception as e:
            logger.error(f"DeepSeek API request error: {str(e)}")
            logger.error(traceback.format_exc())
            
            # For demo purposes, generate a mock prediction if API call fails
            # In production, you might want to handle this differently
            if "symbol" in data:
                logger.warning("Generating mock prediction as fallback")
                return self._generate_mock_prediction(data)
            
            raise
    
    def _format_prompt(self, data: Dict[str, Any]) -> str:
        """
        Format market data into a detailed prompt for the AI
        
        Args:
            data: Enriched market data
            
        Returns:
            Formatted prompt
        """
        # Ensure all required sections exist to avoid KeyErrors
        data_sections = {
            'timeframes': data.get('timeframes', {}),
            'order_book': data.get('order_book', {}),
            'sentiment': data.get('sentiment', {}),
            'macro_factors': data.get('macro_factors', {}),
            'correlations': data.get('correlations', {})
        }
        
        # Ensure timeframes exist
        timeframes = data_sections['timeframes']
        timeframe_5m = timeframes.get('5m', {'price': {}, 'indicators': {}})
        timeframe_1h = timeframes.get('1h', {'price': {}, 'indicators': {}})
        
        # Ensure price and indicators exist
        price_5m = timeframe_5m.get('price', {})
        indicators_5m = timeframe_5m.get('indicators', {})
        price_1h = timeframe_1h.get('price', {})
        indicators_1h = timeframe_1h.get('indicators', {})
        
        # Ensure order book data exists
        order_book = data_sections['order_book']
        
        # Ensure macro factors exist
        macro_factors = data_sections['macro_factors']
        
        prompt = f"""Analyze the following Bitcoin market data and provide a trading signal.

Symbol: {data.get('symbol', 'BTCUSDT')}
Timestamp: {data.get('timestamp', int(time.time()))}

## 5-Minute Timeframe
Price:
- Open: {price_5m.get('open', 'N/A')}
- High: {price_5m.get('high', 'N/A')}
- Low: {price_5m.get('low', 'N/A')}
- Close: {price_5m.get('close', 'N/A')}
- Volume: {price_5m.get('volume', 'N/A')}

Indicators:
- RSI: {indicators_5m.get('RSI', 'N/A')}
- Bollinger Bands Width: {indicators_5m.get('Bollinger_Width', 'N/A')}
- WVO: {indicators_5m.get('WVO', 'N/A')}
- ARSI: {indicators_5m.get('ARSI', 'N/A')}
- VWIO: {indicators_5m.get('VWIO', 'N/A')}

## 1-Hour Timeframe
Price:
- Open: {price_1h.get('open', 'N/A')}
- High: {price_1h.get('high', 'N/A')}
- Low: {price_1h.get('low', 'N/A')}
- Close: {price_1h.get('close', 'N/A')}
- Volume: {price_1h.get('volume', 'N/A')}

Indicators:
- Hourly High Low Percentile: {indicators_1h.get('Hourly_High_Low_Percentile', 'N/A')}
- Hourly Volume Momentum: {indicators_1h.get('Hourly_Volume_Momentum', 'N/A')}
- MACD Histogram: {indicators_1h.get('MACD_Histogram', 'N/A')}
- Parabolic SAR: {indicators_1h.get('Parabolic_SAR', 'N/A')}
- EMA 50/200 Crossover: {indicators_1h.get('EMA_50_200_Crossover', 'N/A')}
- Ichimoku Cloud: {indicators_1h.get('Ichimoku_Cloud', 'N/A')}
- Fibonacci Levels: {indicators_1h.get('Fibonacci_Levels', 'N/A')}
- Volume Profile POC: {indicators_1h.get('Volume_Profile_POC', 'N/A')}
- Pivot Points: {json.dumps(indicators_1h.get('Pivot_Points', {}), indent=2)}
- Williams %R: {indicators_1h.get('Williams_%R', 'N/A')}
- VWAP: {indicators_1h.get('VWAP', 'N/A')}
- CVD: {indicators_1h.get('CVD', 'N/A')}
- ADX: {indicators_1h.get('ADX', 'N/A')}

## Order Book Analysis
- Bid-Ask Spread: {order_book.get('bid_ask_spread', 'N/A')}
- Order Imbalance: {order_book.get('order_imbalance', 'N/A')}
- Large Orders: {json.dumps(order_book.get('large_orders', {}), indent=2)}
- Depth Imbalance: {order_book.get('depth_imbalance', 'N/A')}
- Bid-Ask Ratio: {order_book.get('bid_ask_ratio', 'N/A')}

## Market Sentiment
- Funding Rate: {data_sections['sentiment'].get('funding_rate', 'N/A')}
- Fear & Greed Index: {data_sections['sentiment'].get('fear_greed_index', 'N/A')}
- Social Volume: {data_sections['sentiment'].get('social_volume', 'N/A')}
- News Sentiment: {data_sections['sentiment'].get('news_sentiment_score', 'N/A')}
- Whale Activity: {json.dumps(data_sections['sentiment'].get('whale_activity', {}), indent=2)}

## Macro Factors
- Exchange Reserves: {macro_factors.get('exchange_reserves', 'N/A')}
- BTC Hash Rate: {macro_factors.get('btc_hash_rate', 'N/A')}
- FOMC Event Impact: {macro_factors.get('fomc_event_impact', 'N/A')}
- Traditional Markets: {json.dumps(macro_factors.get('traditional_markets', {}), indent=2)}
- Bond Yields: {json.dumps(macro_factors.get('bond_yields', {}), indent=2)}
- Inflation Expectations: {macro_factors.get('inflation_expectations', 'N/A')}

## Correlation Analysis
- BTC-ETH Correlation: {data_sections['correlations'].get('btc_eth_correlation', 'N/A')}
- BTC-SP500 Correlation: {data_sections['correlations'].get('btc_sp500_correlation', 'N/A')}
- BTC-Gold Correlation: {data_sections['correlations'].get('btc_gold_correlation', 'N/A')}
- BTC-DXY Correlation: {data_sections['correlations'].get('btc_dxy_correlation', 'N/A')}

Based on this data, provide a trading signal in the following JSON format:
{{
  "symbol": "BTCUSDT",
  "timestamp": [current_timestamp],
  "prediction": {{
    "action": "[BUY/SELL/HOLD]",
    "confidence": [value between 0 and 1],
    "stop_loss": [suggested stop loss price],
    "take_profit": [suggested take profit price]
  }}
}}
"""
        return prompt
    
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for the DeepSeek API
        
        Returns:
            System prompt
        """
        return """You are an expert cryptocurrency trading system specialized in Bitcoin technical analysis.
Your task is to analyze market data and provide trading signals (BUY, SELL, or HOLD) along with confidence scores and risk management parameters.

Follow these guidelines:
1. Analyze all provided indicators, order book data, sentiment metrics, and macro factors.
2. For BUY signals, look for bullish patterns, positive momentum, and supportive market sentiment.
3. For SELL signals, look for bearish patterns, negative momentum, and concerning market sentiment.
4. Provide a HOLD signal when the market direction is unclear or risk is too high.
5. Set stop-loss 1-3% away from current price in the opposing direction of your signal.
6. Set take-profit 3-7% away from current price in the direction of your signal.
7. Your confidence score should reflect your certainty in the prediction (0.7-0.85 for moderate confidence, 0.85-1.0 for high confidence).
8. Always respond with valid JSON only. No explanations or additional text outside the JSON structure.
"""
    
    def _validate_prediction(self, prediction: Dict[str, Any]) -> None:
        """
        Validate prediction format
        
        Args:
            prediction: Prediction from DeepSeek API
            
        Raises:
            ValueError: If prediction format is invalid
        """
        required_fields = ["symbol", "timestamp", "prediction"]
        
        for field in required_fields:
            if field not in prediction:
                raise ValueError(f"Missing required field in prediction: {field}")
        
        if "action" not in prediction["prediction"]:
            raise ValueError("Missing 'action' in prediction")
            
        if "confidence" not in prediction["prediction"]:
            raise ValueError("Missing 'confidence' in prediction")
            
        if prediction["prediction"]["action"] not in ["BUY", "SELL", "HOLD"]:
            raise ValueError(f"Invalid action in prediction: {prediction['prediction']['action']}")
            
        if not isinstance(prediction["prediction"]["confidence"], (int, float)):
            raise ValueError("Confidence must be a number")
            
        if prediction["prediction"]["confidence"] < 0 or prediction["prediction"]["confidence"] > 1:
            raise ValueError("Confidence must be between 0 and 1")
    
    def _save_prediction(self, payload: Dict[str, Any], prediction: Dict[str, Any]) -> None:
        """
        Save prediction for backtesting
        
        Args:
            payload: Request payload
            prediction: Prediction response
        """
        # In a real system, you would save to a database or file
        # For this demo, we'll just log it
        timestamp = int(time.time())
        symbol = payload.get("symbol", "UNKNOWN")
        
        log_entry = {
            "timestamp": timestamp,
            "symbol": symbol,
            "payload": payload,
            "prediction": prediction
        }
        
        # Save to file (append)
        try:
            with open(f"predictions_{symbol}.json", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
    
    def _generate_mock_prediction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate mock prediction for testing
        
        Args:
            data: Market data
            
        Returns:
            Mock prediction
        """
        # Get symbol and timestamp from data
        symbol = data.get("symbol", "BTCUSDT")
        timestamp = data.get("timestamp", int(time.time()))
        
        # Generate a random action (with bias towards HOLD)
        actions = ["BUY", "SELL", "HOLD"]
        action_weights = [0.3, 0.3, 0.4]  # 30% BUY, 30% SELL, 40% HOLD
        action = random.choices(actions, weights=action_weights, k=1)[0]
        
        # Generate a random confidence
        confidence = round(random.uniform(0.2, 0.99), 2)
        
        # Get current price from data if available
        current_price = 0
        if "timeframes" in data and "1h" in data["timeframes"] and "price" in data["timeframes"]["1h"]:
            current_price = data["timeframes"]["1h"]["price"].get("close", 0)
        
        if current_price <= 0 and "timeframes" in data and "5m" in data["timeframes"] and "price" in data["timeframes"]["5m"]:
            current_price = data["timeframes"]["5m"]["price"].get("close", 0)
            
        if current_price <= 0:
            current_price = 50000  # Fallback default BTC price
        
        # Generate stop loss and take profit
        if action == "BUY":
            stop_loss = current_price * 0.98  # 2% below current price
            take_profit = current_price * 1.05  # 5% above current price
        elif action == "SELL":
            stop_loss = current_price * 1.02  # 2% above current price
            take_profit = current_price * 0.95  # 5% below current price
        else:  # HOLD
            stop_loss = None
            take_profit = None
        
        # Construct mock prediction
        prediction = {
            "symbol": symbol,
            "timestamp": timestamp,
            "prediction": {
                "action": action,
                "confidence": confidence
            }
        }
        
        # Add stop loss and take profit if not HOLD
        if stop_loss is not None:
            prediction["prediction"]["stop_loss"] = stop_loss
            
        if take_profit is not None:
            prediction["prediction"]["take_profit"] = take_profit
            
        return prediction

    def _add_technical_indicators(self, data: Dict[str, Any]) -> None:
        """Add technical indicators to market data"""
        # 5-minute timeframe indicators
        if "timeframes" in data and "5m" in data["timeframes"]:
            timeframe_5m = data["timeframes"]["5m"]
            
            # Convert price data to pandas DataFrame for easier calculation
            if "price" in timeframe_5m and "klines" in timeframe_5m["price"]:
                df = pd.DataFrame(timeframe_5m["price"]["klines"])
                
                # Calculate RSI
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                timeframe_5m["indicators"]["RSI"] = 100 - (100 / (1 + rs.iloc[-1]))
                
                # Calculate Bollinger Bands
                sma = df['close'].rolling(window=20).mean()
                std = df['close'].rolling(window=20).std()
                upper_band = sma + (std * 2)
                lower_band = sma - (std * 2)
                timeframe_5m["indicators"]["Bollinger_Width"] = (upper_band.iloc[-1] - lower_band.iloc[-1]) / sma.iloc[-1]
                
                # Add more actual calculations for other indicators

        # 1-hour timeframe indicators
        if "timeframes" in data and "1h" in data["timeframes"]:
            timeframe_1h = data["timeframes"]["1h"]
            
            # Convert price data to pandas DataFrame for easier calculation
            if "price" in timeframe_1h and "klines" in timeframe_1h["price"]:
                df = pd.DataFrame(timeframe_1h["price"]["klines"])
                
                # Calculate Market Structure
                timeframe_1h["indicators"]["Market_Structure"] = self._calculate_market_structure(df)

                # Calculate Volatility
                timeframe_1h["indicators"]["ATR"] = self._calculate_atr(df, period=14)  # Average True Range
                timeframe_1h["indicators"]["Historical_Volatility"] = self._calculate_historical_volatility(df, period=20)

                # Calculate Volume-Based Indicators
                timeframe_1h["indicators"]["VWAP"] = self._calculate_vwap(df)  # Volume Weighted Average Price
                timeframe_1h["indicators"]["CVD"] = self._calculate_cvd(df)  # Cumulative Volume Delta

                # Calculate Momentum
                timeframe_1h["indicators"]["Stochastic_RSI"] = self._calculate_stoch_rsi(df)
                timeframe_1h["indicators"]["Williams_R"] = self._calculate_williams_r(df)

                # Calculate Support/Resistance Levels
                timeframe_1h["indicators"]["Key_Levels"] = self._identify_key_levels(df)

    def _calculate_market_structure(self, df: pd.DataFrame) -> str:
        """Determine market structure (uptrend, downtrend, or ranging)"""
        # Simple implementation using higher highs/higher lows
        last_candles = df.tail(10)
        
        # Check for uptrend (higher highs and higher lows)
        if (last_candles['high'].diff().dropna() > 0).sum() >= 6 and \
           (last_candles['low'].diff().dropna() > 0).sum() >= 6:
            return "Uptrend"
        
        # Check for downtrend (lower highs and lower lows)
        elif (last_candles['high'].diff().dropna() < 0).sum() >= 6 and \
             (last_candles['low'].diff().dropna() < 0).sum() >= 6:
            return "Downtrend"
        
        # Otherwise ranging
        else:
            return "Ranging"

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return round(atr, 2)

    def _calculate_historical_volatility(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate historical volatility"""
        # This is a placeholder implementation. You might want to implement a more robust volatility calculation
        # based on historical price data.
        return round(random.uniform(0.01, 0.2), 2)  # Placeholder, actual implementation needed

    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """Calculate Volume Weighted Average Price"""
        # This is a placeholder implementation. You might want to implement a more robust VWAP calculation
        # based on historical price data.
        return round(random.uniform(0.01, 0.2), 2)  # Placeholder, actual implementation needed

    def _calculate_cvd(self, df: pd.DataFrame) -> float:
        """Calculate Cumulative Volume Delta"""
        # This is a placeholder implementation. You might want to implement a more robust CVD calculation
        # based on historical volume data.
        return round(random.uniform(0.01, 0.2), 2)  # Placeholder, actual implementation needed

    def _calculate_stoch_rsi(self, df: pd.DataFrame) -> float:
        """Calculate Stochastic RSI"""
        # This is a placeholder implementation. You might want to implement a more robust Stochastic RSI calculation
        # based on historical price data.
        return round(random.uniform(0.01, 0.2), 2)  # Placeholder, actual implementation needed

    def _calculate_williams_r(self, df: pd.DataFrame) -> float:
        """Calculate Williams %R"""
        # This is a placeholder implementation. You might want to implement a more robust Williams %R calculation
        # based on historical price data.
        return round(random.uniform(-0.2, 0.2), 2)  # Placeholder, actual implementation needed

    def _identify_key_levels(self, df: pd.DataFrame) -> list:
        """Identify key levels (support/resistance)"""
        # This is a placeholder implementation. You might want to implement a more robust key level identification
        # based on historical price data.
        return [round(random.uniform(0.01, 0.2), 2) for _ in range(3)]  # Placeholder, actual implementation needed

    def _add_ml_features(self, data: Dict[str, Any]) -> None:
        """Add machine learning features"""
        if "timeframes" in data and "1h" in data["timeframes"]:
            timeframe_1h = data["timeframes"]["1h"]
            
            # Feature engineering
            if "price" in timeframe_1h and "klines" in timeframe_1h["price"]:
                df = pd.DataFrame(timeframe_1h["price"]["klines"])
                
                # Price action features
                timeframe_1h["ml_features"] = {
                    "price_momentum": self._calculate_price_momentum(df),
                    "volatility_features": self._calculate_volatility_features(df),
                    "pattern_recognition": self._detect_candlestick_patterns(df),
                    "trend_strength": self._calculate_trend_strength(df)
                } 

    def _add_market_regime(self, data: Dict[str, Any]) -> None:
        """Detect current market regime"""
        # Combine volatility, trend, and volume metrics to determine regime
        if "timeframes" in data and "1h" in data["timeframes"]:
            timeframe_1h = data["timeframes"]["1h"]
            
            # Get relevant metrics
            volatility = timeframe_1h["indicators"].get("Historical_Volatility", 0)
            trend = timeframe_1h["indicators"].get("Market_Structure", "Unknown")
            volume = timeframe_1h["indicators"].get("Volume_Profile_POC", 0)
            
            # Determine regime
            if volatility > 0.03 and trend != "Ranging":
                regime = "Trending Volatile"
            elif volatility > 0.03 and trend == "Ranging":
                regime = "Choppy Volatile"
            elif volatility <= 0.03 and trend != "Ranging":
                regime = "Trending Calm"
            else:
                regime = "Ranging Calm"
            
            data["market_regime"] = regime

    def _add_order_flow_metrics(self, data: Dict[str, Any]) -> None:
        """Add order flow metrics"""
        if "order_book" in data:
            order_book = data["order_book"]
            
            # Calculate bid-ask imbalance
            total_bids = sum(item[1] for item in order_book.get("bids", []))
            total_asks = sum(item[1] for item in order_book.get("asks", []))
            
            if total_asks > 0:
                bid_ask_ratio = total_bids / total_asks
            else:
                bid_ask_ratio = 1.0
                
            # Calculate order book depth
            depth_bids = sum(item[1] for item in order_book.get("bids", [])[:10])
            depth_asks = sum(item[1] for item in order_book.get("asks", [])[:10])
            
            data["order_flow"] = {
                "bid_ask_ratio": round(bid_ask_ratio, 2),
                "depth_imbalance": round((depth_bids - depth_asks) / (depth_bids + depth_asks) if (depth_bids + depth_asks) > 0 else 0, 2),
                "large_orders": self._detect_large_orders(order_book)
            }

    def _add_correlation_analysis(self, data: Dict[str, Any]) -> None:
        """Add correlation analysis with other assets"""
        # In a real system, fetch price data for other assets
        # For now, using random values as placeholders
        
        data["correlations"] = {
            "btc_eth_correlation": round(random.uniform(0.7, 0.95), 2),  # Replace with actual calculation
            "btc_sp500_correlation": round(random.uniform(-0.2, 0.5), 2),
            "btc_gold_correlation": round(random.uniform(-0.3, 0.3), 2),
            "btc_dxy_correlation": round(random.uniform(-0.6, -0.2), 2)
        }

    def _add_anomaly_detection(self, data: Dict[str, Any]) -> None:
        """Detect market anomalies"""
        anomalies = []
        
        # Check for volume anomalies
        if "timeframes" in data and "5m" in data["timeframes"]:
            timeframe_5m = data["timeframes"]["5m"]
            if "price" in timeframe_5m and "klines" in timeframe_5m["price"]:
                df = pd.DataFrame(timeframe_5m["price"]["klines"])
                
                # Check for volume spike (3x average)
                avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
                current_volume = df['volume'].iloc[-1]
                
                if current_volume > avg_volume * 3:
                    anomalies.append("Volume spike detected")
        
        # Check for price anomalies
        if "sentiment" in data and "funding_rate" in data["sentiment"]:
            funding_rate = data["sentiment"]["funding_rate"]
            if abs(funding_rate) > 0.005:  # 0.5% is quite extreme
                anomalies.append(f"Extreme funding rate: {funding_rate}")
        
        data["anomalies"] = anomalies

    def enrich_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich market data with parallel processing"""
        enriched_data = market_data.copy()
        
        try:
            # Use parallel processing for independent enrichment tasks
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(self._add_technical_indicators, enriched_data): "technical",
                    executor.submit(self._add_sentiment_data, enriched_data): "sentiment",
                    executor.submit(self._add_macro_factors, enriched_data): "macro",
                    executor.submit(self._add_onchain_metrics, enriched_data): "onchain",
                    executor.submit(self._add_correlation_analysis, enriched_data): "correlation"
                }
                
                for future in concurrent.futures.as_completed(futures):
                    task_name = futures[future]
                    try:
                        future.result()  # Get the result or exception
                        logger.info(f"Completed {task_name} enrichment")
                    except Exception as e:
                        logger.error(f"Error in {task_name} enrichment: {str(e)}")
            
            # Add risk metrics after other metrics are calculated
            self._add_risk_metrics(enriched_data)
            
            # Add anomaly detection
            self._add_anomaly_detection(enriched_data)
            
            return enriched_data
            
        except Exception as e:
            logger.error(f"Error enriching data: {str(e)}")
            if "timeframes" in enriched_data and enriched_data["timeframes"]:
                logger.warning("Returning partially enriched data")
                return enriched_data
            raise 