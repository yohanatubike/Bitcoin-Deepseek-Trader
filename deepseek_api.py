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
import requests
import numpy as np

# Import OpenAI SDK for DeepSeek API compatibility
from openai import OpenAI
from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)

class DeepSeekAPI:
    def __init__(self, api_key: str, api_url: str = "https://api.deepseek.com", model_name: str = "deepseek-chat", test_mode: bool = False):
        """
        Initialize the DeepSeek API client
        
        Args:
            api_key: DeepSeek API key
            api_url: DeepSeek API base URL (default: "https://api.deepseek.com")
            model_name: DeepSeek model name (default: "deepseek-chat")
            test_mode: Whether to use mock predictions instead of calling the API
        """
        self.api_key = api_key
        self.model_name = model_name
        self.test_mode = test_mode
        self._market_data = {}  # Store market data for reference
        
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
            
            if test_mode:
                logger.warning("TEST MODE ENABLED: Will use mock predictions instead of real API calls")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _boost_confidence(self, prediction: Dict[str, Any], min_confidence: float = 0.85) -> Dict[str, Any]:
        """
        Boost the confidence score to meet minimum threshold
        
        Args:
            prediction: The prediction dictionary
            min_confidence: Minimum confidence score
            
        Returns:
            Updated prediction dictionary
        """
        if "prediction" in prediction and isinstance(prediction["prediction"], dict):
            pred_data = prediction["prediction"]
            if "confidence" in pred_data and isinstance(pred_data["confidence"], (int, float)):
                # Apply confidence boosting
                current_confidence = pred_data["confidence"]
                
                # Only boost if current confidence is somewhat high but below threshold
                if 0.7 <= current_confidence < min_confidence:
                    # Apply logarithmic scaling to maintain some relationship to original confidence
                    # while ensuring it meets the minimum threshold
                    normalized_confidence = (current_confidence - 0.7) / 0.3  # Scale to 0-1 range
                    boosted_confidence = min_confidence + ((1 - min_confidence) * normalized_confidence)
                    pred_data["confidence"] = round(boosted_confidence, 2)
                    logger.info(f"Boosted confidence from {current_confidence} to {pred_data['confidence']}")
                
        return prediction

    def get_prediction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get market prediction from DeepSeek API
        
        Args:
            data: Market data
            
        Returns:
            Prediction dictionary
        """
        # Store market data for validation purposes
        self._market_data = data
        
        # Skip API call if in test mode
        if self.test_mode:
            logger.info("Test mode enabled, generating mock prediction")
            return self._boost_confidence(self._generate_mock_prediction(data))
            
        # Prepare payload
        try:
            # Get symbol from data
            symbol = data.get("symbol", "BTCUSDT")
            
            # Format prompt
            prompt = self._format_trading_prompt(data)
            
            # Prepare API request
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ]
            
            # Make API request
            logger.info(f"Sending request to DeepSeek API")
            
            try:
                # Call the OpenAI-compatible API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    max_tokens=400
                )
                
                # Check if the response contains a valid message
                if not response or not hasattr(response, 'choices') or not response.choices:
                    logger.error(f"Invalid response format from DeepSeek API: {response}")
                    # Fallback to mock prediction
                    logger.warning("Generating mock prediction as fallback")
                    return self._boost_confidence(self._generate_mock_prediction(data))
                
                # Process the response
                return self._process_response(response, data)
                    
            except Exception as api_error:
                logger.error(f"API call error: {str(api_error)}")
                logger.warning("Generating mock prediction as fallback")
                return self._generate_mock_prediction(data)
                    
        except Exception as e:
            logger.error(f"Error getting prediction from DeepSeek API: {str(e)}")
            logger.error(traceback.format_exc())
            # Fallback to mock prediction
            logger.warning("Generating mock prediction as fallback")
            return self._generate_mock_prediction(data)
    
    def _process_response(self, response: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the response from DeepSeek API"""
        try:
            # Extract the response content from pydantic model
            content = response.choices[0].message.content
            
            # Parse the JSON response
            prediction = json.loads(content)
            
            # Add timestamp and symbol
            prediction['timestamp'] = int(time.time() * 1000)
            prediction['symbol'] = data.get('symbol', 'BTCUSDT')
            
            # Wrap in prediction object if not already
            if 'prediction' not in prediction:
                prediction = {'prediction': prediction}
            
            logger.info(f"✅ Received prediction from DeepSeek API: {json.dumps(prediction)}")
            return prediction
            
        except Exception as e:
            logger.error(f"DeepSeek API request error: {str(e)}")
            logger.error(traceback.format_exc())
            logger.warning("⚠️ Generating mock prediction as fallback")
            return self._generate_mock_prediction(data)
    
    def _format_trading_prompt(self, market_data: Dict[str, Any]) -> str:
        """Format the prompt for trading signals"""
        try:
            # Get indicators
            indicators = market_data.get('indicators', {})
            indicators_5m = indicators.get('5m', {})
            indicators_1h = indicators.get('1h', {})
            
            # Get order book data
            order_book = market_data.get('order_book', {})
            order_book_metrics = order_book.get('metrics', {})
            
            # Format prompt
            prompt = f"""Please analyze the following market data for {market_data.get('symbol', 'BTCUSDT')} and provide trading signals:

            5-Minute Timeframe Indicators:
            Momentum:
            - RSI: {indicators_5m.get('RSI', 50.0):.2f}
            - Adaptive RSI (ARSI): {indicators_5m.get('ARSI', 50.0):.2f}
            - Stochastic RSI: {indicators_5m.get('Stoch_RSI', {}).get('K', 0.0):.2f}/{indicators_5m.get('Stoch_RSI', {}).get('D', 0.0):.2f}
            
            Volatility:
            - Bollinger Bands Width: {indicators_5m.get('Bollinger_Bands_Width', 0.02):.4f}
            - ATR: {indicators_5m.get('ATR', 0.0):.2f}
            - Historical Volatility: {indicators_5m.get('Historical_Volatility', 0.0):.4f}
            
            Volume:
            - Volume Weighted Oscillator (WVO): {indicators_5m.get('WVO', 0.0):.2f}
            - Volume Weighted Intensity (VWIO): {indicators_5m.get('VWIO', 0.0):.2f}
            - Volume Profile POC: {indicators_5m.get('Volume_Profile_POC', 0.0):.2f}
            
            Trend:
            - ADX: {indicators_5m.get('ADX', 25.0):.2f}
            - Trend Strength: {indicators_5m.get('Trend_Strength', {}).get('value', 0.0):.2f}
            - Market Structure: {indicators_5m.get('Market_Structure', 'NEUTRAL')}

            1-Hour Timeframe Indicators:
            Momentum:
            - RSI: {indicators_1h.get('RSI', 48.0):.2f}
            - Williams %R: {indicators_1h.get('Williams_%R', -50.0):.2f}
            - Stochastic RSI: {indicators_1h.get('Stoch_RSI', {}).get('K', 0.0):.2f}/{indicators_1h.get('Stoch_RSI', {}).get('D', 0.0):.2f}
            
            Volatility:
            - Bollinger Bands Width: {indicators_1h.get('Bollinger_Bands_Width', 0.025):.4f}
            - ATR: {indicators_1h.get('ATR', 0.0):.2f}
            - Historical Volatility: {indicators_1h.get('Historical_Volatility', 0.0):.4f}
            
            Trend:
            - MACD:
              * MACD Line: {indicators_1h.get('MACD', {}).get('MACD', 0.0):.2f}
              * Signal Line: {indicators_1h.get('MACD', {}).get('Signal', 0.0):.2f}
              * Histogram: {indicators_1h.get('MACD', {}).get('Histogram', 0.0):.2f}
            - Parabolic SAR: {indicators_1h.get('Parabolic_SAR', 'NEUTRAL')}
            - EMA 50/200 Crossover: {indicators_1h.get('EMA_50_200_Crossover', 'NEUTRAL')}
            - ADX: {indicators_1h.get('ADX', 25.0):.2f}
            - Trend Strength: {indicators_1h.get('Trend_Strength', {}).get('value', 0.0):.2f}
            - Market Structure: {indicators_1h.get('Market_Structure', 'NEUTRAL')}
            
            Advanced Indicators:
            - Ichimoku Cloud Status: {indicators_1h.get('Ichimoku', 'NEUTRAL')}
            - Hourly High/Low Percentile: {indicators_1h.get('Hourly_High_Low_Percentile', 0.0):.2f}
            - Hourly Volume Momentum: {indicators_1h.get('Hourly_Volume_Momentum', 0.0):.2f}
            
            Support/Resistance:
            - Key Support: {indicators_1h.get('Support_Resistance', {}).get('support', 0.0):.2f}
            - Key Resistance: {indicators_1h.get('Support_Resistance', {}).get('resistance', 0.0):.2f}
            - Fibonacci Levels:
              * 0.236: {indicators_1h.get('Fibonacci_Levels', {}).get('0.236', 0.0):.2f}
              * 0.382: {indicators_1h.get('Fibonacci_Levels', {}).get('0.382', 0.0):.2f}
              * 0.618: {indicators_1h.get('Fibonacci_Levels', {}).get('0.618', 0.0):.2f}
            
            Order Book Analysis:
            Volume Analysis:
            - Bid/Ask Ratio: {order_book_metrics.get('bid_ask_ratio', 1.0):.4f}
            - Order Imbalance: {order_book_metrics.get('order_imbalance', 0.0):.4f}
            - Order Flow Score: {order_book_metrics.get('order_flow_score', 0.0):.2f}
            
            Price Analysis:
            - Spread: {order_book_metrics.get('spread', 0.0):.2f}
            - Spread %: {order_book_metrics.get('spread_percentage', 0.0):.4f}%
            - VWAP Analysis:
              * Bid VWAP: {order_book_metrics.get('bid_vwap', 0.0):.2f}
              * Ask VWAP: {order_book_metrics.get('ask_vwap', 0.0):.2f}
              * VWAP Midpoint: {order_book_metrics.get('vwap_midpoint', 0.0):.2f}
            
            Liquidity Analysis:
            - Liquidity Score: {order_book_metrics.get('liquidity_score', 0.0):.2f}
            - Depth Imbalance: {order_book_metrics.get('depth_imbalance', 0.0):.4f}
            - Large Orders Impact: {order_book_metrics.get('large_orders_impact', 0.0):.2f}

            Futures Data:
            - Funding Rate: {market_data.get('futures_data', {}).get('funding_rate', 0.0):.6f}
            - Next Funding Time: {market_data.get('futures_data', {}).get('next_funding_time', 'Unknown')}
            - Open Interest: {market_data.get('futures_data', {}).get('open_interest', 0.0):.2f}
            - Open Interest Value: ${market_data.get('futures_data', {}).get('open_interest_value', 0.0):.2f}
            - Long/Short Ratio: {market_data.get('futures_data', {}).get('long_short_ratio', 1.0):.2f}

            Based on this comprehensive market data, please provide:
            1. Trading action (BUY/SELL/HOLD)
            2. Confidence level (0.0-1.0)
            3. Suggested stop loss price
            4. Suggested take profit price
            5. Detailed reasoning for the recommendation, considering:
               - Trend analysis across timeframes
               - Support/Resistance levels
               - Volume and liquidity analysis
               - Market structure
               - Risk factors
            6. Risk/reward ratio for the trade
            7. Position sizing recommendation (0.0-1.0)

            Please format your response as a JSON object with the following structure:
            {{
                "action": "BUY/SELL/HOLD",
                "confidence": 0.XX,
                "stop_loss": XXX.XX,
                "take_profit": XXX.XX,
                "reasoning": "Your detailed analysis here",
                "risk_reward_ratio": X.XX,
                "position_size": 0.XX,
                "risk_factors": ["factor1", "factor2", ...],
                "key_levels": {{
                    "support": XXX.XX,
                    "resistance": XXX.XX
                }}
            }}
            """
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error formatting trading prompt: {str(e)}")
            logger.error(traceback.format_exc())
            return ""
    
    def _format_position_management_prompt(self, data: Dict[str, Any]) -> str:
        """Format the prompt for position management"""
        try:
            # Get position data
            position = data.get('position', {})
            
            # Get indicators
            indicators = data.get('indicators', {})
            indicators_5m = indicators.get('5m', {})
            indicators_1h = indicators.get('1h', {})
            
            # Get order book data
            order_book = data.get('order_book', {})
            order_book_metrics = order_book.get('metrics', {})
            
            # Format prompt
            prompt = f"""Please analyze the following market data and provide position management advice for {data.get('symbol', 'BTCUSDT')}:

            Current Position:
            - Side: {position.get('side', 'UNKNOWN')}
            - Entry Price: {position.get('entry_price', 0.0):.2f}
            - Current Price: {position.get('mark_price', 0.0):.2f}
            - Unrealized PnL: {position.get('unrealized_pnl', 0.0):.2f} ({position.get('roe', 0.0):.2f}%)
            - Position Size: {position.get('size', 0.0):.8f}
            - Leverage: {position.get('leverage', 1)}x
            - Margin Type: {position.get('margin_type', 'ISOLATED')}

            5-Minute Timeframe Indicators:
            - RSI: {indicators_5m.get('RSI', 50.0):.2f}
            - Bollinger Bands Width: {indicators_5m.get('Bollinger_Bands_Width', 0.02):.4f}
            - Volume Weighted Oscillator (WVO): {indicators_5m.get('WVO', 0.0):.2f}
            - Adaptive RSI (ARSI): {indicators_5m.get('ARSI', 50.0):.2f}
            - Volume Weighted Intensity (VWIO): {indicators_5m.get('VWIO', 0.0):.2f}
            - ADX: {indicators_5m.get('ADX', 25.0):.2f}

            1-Hour Timeframe Indicators:
            - RSI: {indicators_1h.get('RSI', 48.0):.2f}
            - Bollinger Bands Width: {indicators_1h.get('Bollinger_Bands_Width', 0.025):.4f}
            - MACD:
            * MACD Line: {indicators_1h.get('MACD', {}).get('MACD', 0.0):.2f}
            * Signal Line: {indicators_1h.get('MACD', {}).get('Signal', 0.0):.2f}
            * Histogram: {indicators_1h.get('MACD', {}).get('Histogram', 0.0):.2f}
            - Parabolic SAR: {indicators_1h.get('Parabolic_SAR', 'NEUTRAL')}
            - EMA 50/200 Crossover: {indicators_1h.get('EMA_50_200_Crossover', 'NEUTRAL')}
            - Williams %R: {indicators_1h.get('Williams_%R', -50.0):.2f}
            - ADX: {indicators_1h.get('ADX', 25.0):.2f}

            Order Book Analysis:
            - Bid/Ask Ratio: {order_book_metrics.get('bid_ask_ratio', 1.0):.4f}
            - Order Imbalance: {order_book_metrics.get('order_imbalance', 0.0):.4f}
            - Spread: {order_book_metrics.get('spread', 0.0):.2f}
            - Spread %: {order_book_metrics.get('spread_percentage', 0.0):.4f}%
            - VWAP:
            * Bid VWAP: {order_book_metrics.get('bid_vwap', 0.0):.2f}
            * Ask VWAP: {order_book_metrics.get('ask_vwap', 0.0):.2f}
            * VWAP Midpoint: {order_book_metrics.get('vwap_midpoint', 0.0):.2f}

            Futures Data:
            - Funding Rate: {data.get('futures_data', {}).get('funding_rate', 0.0):.6f}
            - Next Funding Time: {data.get('futures_data', {}).get('next_funding_time', 'Unknown')}
            - Open Interest: {data.get('futures_data', {}).get('open_interest', 0.0):.2f}
            - Open Interest Value: ${data.get('futures_data', {}).get('open_interest_value', 0.0):,.2f}

            Based on this data, please provide position management advice:
            1. Action (HOLD/CLOSE/PARTIAL_CLOSE)
            2. If PARTIAL_CLOSE, what percentage to close
            3. Update stop loss price (if needed)
            4. Update take profit price (if needed)
            5. Brief reasoning for the recommendation

            Please format your response as a JSON object with the following structure:
            {
                "action": "HOLD/CLOSE/PARTIAL_CLOSE",
                "close_percentage": XX.XX,  # Only if action is PARTIAL_CLOSE
                "stop_loss": XXX.XX,
                "take_profit": XXX.XX,
                "reasoning": "Your analysis here"
            }"""
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error formatting position management prompt: {str(e)}")
            logger.error(traceback.format_exc())
            return ""
    
    def _get_system_prompt(self) -> str:
        """
        Get system prompt for DeepSeek API
        
        Returns:
            System prompt
        """
        return """You are a professional cryptocurrency futures trader with expertise in technical analysis, on-chain metrics, and market sentiment.

        Your goal is to analyze the provided market data and generate precise trading signals or position management advice.

        For trade signals:
        1. Carefully analyze all timeframes, indicators, market sentiment, and on-chain data
        2. Provide either BUY, SELL, or HOLD recommendations with confidence score
        3. For BUY/SELL signals, include logical stop loss and take profit levels
        4. Be cautious - only recommend trades with strong confirmation across multiple indicators
        5. Explain your reasoning clearly and concisely in JSON format

        For position management:
        1. Carefully analyze existing positions in relation to current market conditions
        2. For each position, recommend whether to HOLD, CLOSE, PARTIAL_CLOSE, or MODIFY_SL_TP
        3. Include specific percentage for PARTIAL_CLOSE and specific levels for MODIFY_SL_TP
        4. Assess both technical and sentiment factors to justify your recommendation
        5. Return recommendations in the proper JSON format with confidence scores

        Only return valid JSON. Never include natural language explanations outside the JSON.
        """
    
    def _validate_prediction(self, prediction: Dict[str, Any]) -> None:
        """
        Validate prediction format and values
        
        Args:
            prediction: Prediction dictionary
            
        Raises:
            ValueError: If prediction is invalid
        """
        # Check if prediction exists
        if not prediction:
            raise ValueError("Empty prediction received")
            
        # Check if the prediction structure is as expected (nested with a 'prediction' key)
        if "prediction" not in prediction:
            raise ValueError("Missing 'prediction' key in response")
            
        # Get the nested prediction object
        pred_data = prediction["prediction"]
        
        # Check if action exists
        if "action" not in pred_data:
            raise ValueError("Prediction missing 'action' field")
            
        # Validate action
        action = pred_data.get("action", "").upper()
        if action not in ["BUY", "SELL", "HOLD"]:
            raise ValueError(f"Invalid action: {action}")
            
        # Validate confidence
        confidence = pred_data.get("confidence", 0)
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            logger.warning(f"Invalid confidence: {confidence}. Defaulting to 0.5")
            pred_data["confidence"] = 0.5
            
        # Get symbol for precision formatting
        symbol = prediction.get("symbol", "BTCUSDT")  # Get symbol from the response
        
        # Validate stop_loss and take_profit if action is not HOLD
        if action != "HOLD":
            # Validate stop_loss for BUY actions
            stop_loss = pred_data.get("stop_loss")
            if action == "BUY" and (stop_loss is None or stop_loss == "N/A" or not isinstance(stop_loss, (int, float)) or stop_loss <= 0):
                # Get current price and set default stop loss 2% below
                try:
                    # Get price data from stored market data
                    price_data = self._market_data.get("timeframes", {}).get("5m", {}).get("price", {})
                    current_price = price_data.get("close", 0)
                    
                    if current_price > 0:
                        default_stop = current_price * 0.98  # 2% below current price
                        
                        # Apply appropriate precision based on symbol
                        if symbol == "BTCUSDT":
                            default_stop = round(default_stop, 1)  # 1 decimal place for BTC
                            
                        logger.warning(f"Invalid stop_loss for BUY. Setting default 2% below current price: {default_stop}")
                        pred_data["stop_loss"] = default_stop
                    else:
                        logger.error("Could not set default stop_loss: current price unavailable")
                except Exception as e:
                    logger.error(f"Error setting default stop_loss: {str(e)}")
            elif stop_loss is not None and isinstance(stop_loss, (int, float)) and stop_loss > 0:
                # Format existing stop loss to correct precision
                if symbol == "BTCUSDT":
                    pred_data["stop_loss"] = round(stop_loss, 1)
            
            # Validate stop_loss for SELL actions
            if action == "SELL" and (stop_loss is None or stop_loss == "N/A" or not isinstance(stop_loss, (int, float)) or stop_loss <= 0):
                # Get current price and set default stop loss 2% above
                try:
                    price_data = self._market_data.get("timeframes", {}).get("5m", {}).get("price", {})
                    current_price = price_data.get("close", 0)
                    
                    if current_price > 0:
                        default_stop = current_price * 1.02  # 2% above current price
                        
                        # Apply appropriate precision based on symbol
                        if symbol == "BTCUSDT":
                            default_stop = round(default_stop, 1)  # 1 decimal place for BTC
                            
                        logger.warning(f"Invalid stop_loss for SELL. Setting default 2% above current price: {default_stop}")
                        pred_data["stop_loss"] = default_stop
                    else:
                        logger.error("Could not set default stop_loss: current price unavailable")
                except Exception as e:
                    logger.error(f"Error setting default stop_loss: {str(e)}")
            elif stop_loss is not None and isinstance(stop_loss, (int, float)) and stop_loss > 0:
                # Format existing stop loss to correct precision
                if symbol == "BTCUSDT":
                    pred_data["stop_loss"] = round(stop_loss, 1)
                    
            # Validate take_profit
            take_profit = pred_data.get("take_profit")
            if take_profit is None or take_profit == "N/A" or not isinstance(take_profit, (int, float)) or take_profit <= 0:
                # Set default take profit based on action
                try:
                    price_data = self._market_data.get("timeframes", {}).get("5m", {}).get("price", {})
                    current_price = price_data.get("close", 0)
                    
                    if current_price > 0:
                        # For BUY: 3% above current price
                        # For SELL: 3% below current price
                        default_tp = current_price * (1.03 if action == "BUY" else 0.97)
                        
                        # Apply appropriate precision based on symbol
                        if symbol == "BTCUSDT":
                            default_tp = round(default_tp, 1)  # 1 decimal place for BTC
                            
                        logger.warning(f"Invalid take_profit for {action}. Setting default: {default_tp}")
                        pred_data["take_profit"] = default_tp
                    else:
                        logger.error("Could not set default take_profit: current price unavailable")
                except Exception as e:
                    logger.error(f"Error setting default take_profit: {str(e)}")
            elif take_profit is not None and isinstance(take_profit, (int, float)) and take_profit > 0:
                # Format existing take profit to correct precision
                if symbol == "BTCUSDT":
                    pred_data["take_profit"] = round(take_profit, 1)
        
        # For HOLD actions, stop_loss and take_profit can be None
        if action == "HOLD":
            # If HOLD but stop_loss and take_profit are provided, do basic validation
            if "stop_loss" in pred_data and pred_data["stop_loss"] is not None:
                stop_loss = pred_data["stop_loss"]
                if not isinstance(stop_loss, (int, float)) or stop_loss <= 0:
                    pred_data["stop_loss"] = None
                elif symbol == "BTCUSDT":
                    # Ensure proper precision
                    pred_data["stop_loss"] = round(stop_loss, 1)
                    
            if "take_profit" in pred_data and pred_data["take_profit"] is not None:
                take_profit = pred_data["take_profit"]
                if not isinstance(take_profit, (int, float)) or take_profit <= 0:
                    pred_data["take_profit"] = None
                elif symbol == "BTCUSDT":
                    # Ensure proper precision
                    pred_data["take_profit"] = round(take_profit, 1)
    
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
        
        # Get symbol from prediction instead of payload
        # The prediction object has the correct symbol, whereas payload might not have it
        symbol = prediction.get("symbol", "UNKNOWN")
        if symbol == "UNKNOWN" and "prediction" in prediction and isinstance(prediction["prediction"], dict):
            # Sometimes the symbol is nested inside the prediction object
            symbol = prediction["prediction"].get("symbol", "UNKNOWN")
            
        # If still unknown, try to get from market data
        if symbol == "UNKNOWN" and hasattr(self, '_market_data') and self._market_data:
            symbol = self._market_data.get("symbol", "UNKNOWN")
        
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
        """Generate a mock prediction when the API fails"""
        current_price = float(data.get('current_price', 0.0))
        
        mock = {
            'prediction': {
                'action': 'HOLD',
                'confidence': 0.64,
                'stop_loss': current_price * 0.98 if current_price > 0 else None,
                'take_profit': current_price * 1.02 if current_price > 0 else None,
                'reasoning': 'Mock prediction due to API error',
                'risk_reward_ratio': 1.0
            },
            'timestamp': int(time.time() * 1000),
            'symbol': data.get('symbol', 'BTCUSDT')
        }
        
        return mock

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
                timeframe_5m["indicators"]["Bollinger_Bands_Width"] = (upper_band.iloc[-1] - lower_band.iloc[-1]) / sma.iloc[-1]
                
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