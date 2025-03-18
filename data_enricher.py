"""
Module for enriching market data with technical indicators, sentiment and macro data
"""

import logging
import random
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from functools import lru_cache
import json
import copy
import os
import traceback

# Import SantimentAPI
from santiment_api import SantimentAPI

# Import ta (Technical Analysis Library in Python)
import ta

logger = logging.getLogger(__name__)

class DataEnricher:
    def __init__(self, santiment_api: SantimentAPI = None):
        """
        Initialize the data enricher
        
        Args:
            santiment_api: Optional SantimentAPI instance
        """
        logger.info("Initializing Data Enricher")
        self.cache = {}
        self.santiment_api = santiment_api
    
    def enrich_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich market data with technical indicators and analysis
        
        Args:
            market_data: Raw market data from Binance
            
        Returns:
            Dict containing enriched market data with indicators
        """
        try:
            logger.info("✅ Starting data enrichment")
            
            # Get klines data for different timeframes
            klines_data = market_data.get("klines", {})
            
            # Log available klines data
            timeframes = {tf: len(data) for tf, data in klines_data.items()}
            logger.info(f"✅ Klines data available: {', '.join(f'{tf}={count}' for tf, count in timeframes.items())}")
            
            # Convert klines to DataFrames and calculate indicators for each timeframe
            enriched_data = market_data.copy()
            indicators = {}
            
            for timeframe, data in klines_data.items():
                if not data:
                    continue
                    
                # Convert to DataFrame
                df = pd.DataFrame(data, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Convert string values to float
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Set timestamp as index
                df.set_index('open_time', inplace=True)
                
                # Calculate indicators for this timeframe
                timeframe_indicators = self._calculate_indicators(df, timeframe)
                
                if timeframe == '5m':
                    indicators['5m'] = timeframe_indicators
                    logger.info(f"✅ ✅ 5m indicators: {json.dumps(timeframe_indicators)}")
                elif timeframe == '1h':
                    indicators['1h'] = timeframe_indicators
                    logger.info(f"✅ ✅ 1h indicators: {json.dumps(timeframe_indicators)}")
            
            # Add indicators to enriched data
            enriched_data['indicators'] = indicators
            
            # Add order book analysis if available
            if 'order_book' in market_data:
                enriched_data['order_book'] = self._enhance_order_book_data(market_data['order_book'])
                logger.info("✅ ✅ Enhanced order book data with advanced metrics")
            
            # Add correlation analysis
            logger.info("✅ 📊 Adding correlation analysis")
            correlation_data = self._calculate_correlations(klines_data)
            enriched_data['correlations'] = correlation_data
            logger.info("✅ ✅ Correlation analysis added")
            
            # Add sentiment data if available
            if self.santiment_api:
                try:
                    sentiment_data = self.santiment_api.get_sentiment_metrics(market_data['symbol'])
                    enriched_data['sentiment'] = sentiment_data
                    logger.info("✅ ✅ Added sentiment data from Santiment API")
                except Exception as e:
                    logger.error(f"Error getting sentiment data: {str(e)}")
                    enriched_data['sentiment'] = self._get_default_sentiment()
            else:
                logger.info("Santiment API not available. Using default sentiment values.")
                enriched_data['sentiment'] = self._get_default_sentiment()
            
            logger.info("✅ ✅ Technical indicators calculated successfully")
            return enriched_data
            
        except Exception as e:
            logger.error(f"Error enriching data: {str(e)}")
            logger.error(traceback.format_exc())
            return market_data
    
    def _add_technical_indicators(self, data: Dict[str, Any]) -> None:
        """
        Add technical indicators to data dictionary
        
        Args:
            data: Market data dictionary
        """
        try:
            # Extract klines for different timeframes
            klines_5m = data.get("klines", {}).get("5m", [])
            klines_1h = data.get("klines", {}).get("1h", [])
            klines_4h = data.get("klines", {}).get("4h", [])
            
            # Log klines data
            logger.info(f"Klines data available: 5m={len(klines_5m)}, 1h={len(klines_1h)}, 4h={len(klines_4h)}")
            
            timeframes = {}
            
            # Initialize timeframes structure if not present
            if "timeframes" not in data:
                data["timeframes"] = {}
                
            # 5-minute timeframe indicators
            if klines_5m:
                try:
                    # Convert klines to DataFrame
                    df_5m = self._klines_to_dataframe(klines_5m)
                    
                    if len(df_5m) > 0:
                        # Initialize timeframe dict if not present
                        if "5m" not in data["timeframes"]:
                            data["timeframes"]["5m"] = {"price": {}, "indicators": {}}
                            
                        timeframe_5m = data["timeframes"]["5m"]
                        
                        # Extract and add price data
                        if "price" not in timeframe_5m:
                            timeframe_5m["price"] = {}
                            
                        # Add latest price data
                        latest = df_5m.iloc[-1]
                        timeframe_5m["price"]["open"] = float(latest["open"])
                        timeframe_5m["price"]["high"] = float(latest["high"])
                        timeframe_5m["price"]["low"] = float(latest["low"])
                        timeframe_5m["price"]["close"] = float(latest["close"])
                        timeframe_5m["price"]["volume"] = float(latest["volume"])
                        
                        # Initialize indicators dict if not present
                        if "indicators" not in timeframe_5m:
                            timeframe_5m["indicators"] = {}
                            
                        # Add 5m indicators
                        timeframe_5m["indicators"]["RSI"] = self._calculate_rsi(df_5m["close"])
                        timeframe_5m["indicators"]["Bollinger_Bands_Width"] = self._calculate_bollinger_width(df_5m["close"])
                        timeframe_5m["indicators"]["WVO"] = self._calculate_vwo(df_5m["volume"])
                        timeframe_5m["indicators"]["ARSI"] = self._calculate_arsi(df_5m["close"])
                        timeframe_5m["indicators"]["VWIO"] = self._calculate_vwio(df_5m["volume"])
                        timeframe_5m["indicators"]["ADX"] = self._calculate_adx(df_5m)
                        logger.info(f"✅ Calculated ADX value for 5m timeframe: {timeframe_5m['indicators']['ADX']}")
                        
                        # Add debugging for all indicators
                        logger.info(f"✅ 5m indicators: {json.dumps(timeframe_5m.get('indicators', {}))}")
                        
                        # Continue with other indicators...
                except Exception as e:
                    logger.error(f"Error calculating 5m indicators: {e}")
                    
            # 1-hour timeframe indicators
            if klines_1h:
                try:
                    # Convert klines to DataFrame
                    df_1h = self._klines_to_dataframe(klines_1h)
                    
                    if len(df_1h) > 0:
                        # Initialize timeframe dict if not present
                        if "1h" not in data["timeframes"]:
                            data["timeframes"]["1h"] = {"price": {}, "indicators": {}}
                            
                        timeframe_1h = data["timeframes"]["1h"]
                        
                        # Extract and add price data
                        if "price" not in timeframe_1h:
                            timeframe_1h["price"] = {}
                            
                        # Add latest price data
                        latest = df_1h.iloc[-1]
                        timeframe_1h["price"]["open"] = float(latest["open"])
                        timeframe_1h["price"]["high"] = float(latest["high"])
                        timeframe_1h["price"]["low"] = float(latest["low"])
                        timeframe_1h["price"]["close"] = float(latest["close"])
                        timeframe_1h["price"]["volume"] = float(latest["volume"])
                        
                        # Initialize indicators dict if not present
                        if "indicators" not in timeframe_1h:
                            timeframe_1h["indicators"] = {}
                            
                        # Add 1h indicators
                        # 1. MACD
                        timeframe_1h["indicators"]["MACD"] = self._calculate_macd(df_1h["close"])
                        
                        # 2. Parabolic SAR
                        timeframe_1h["indicators"]["Parabolic_SAR"] = self._calculate_parabolic_sar(df_1h["high"], df_1h["low"], df_1h["close"])
                        
                        # 3. EMA Crossover
                        timeframe_1h["indicators"]["EMA_50_200_Crossover"] = self._calculate_ema_crossover(df_1h["close"])
                        
                        # 4. Ichimoku Cloud
                        timeframe_1h["indicators"]["Ichimoku"] = self._calculate_ichimoku(df_1h["high"], df_1h["low"], df_1h["close"])
                        
                        # 5. Williams %R
                        timeframe_1h["indicators"]["Williams_%R"] = self._calculate_williams_r(df_1h)
                        
                        # 6. Advanced Directional Index (ADX)
                        adx_value = self._calculate_adx(df_1h)
                        timeframe_1h["indicators"]["ADX"] = adx_value
                        logger.info(f"✅ Calculated ADX value for 1h timeframe: {adx_value}")
                        
                        # Add debugging for all indicators
                        logger.info(f"✅ 5m indicators: {json.dumps(timeframe_5m.get('indicators', {}))}")
                        logger.info(f"✅ 1h indicators: {json.dumps(timeframe_1h.get('indicators', {}))}")
                        
                        # 7. Hourly High Low Percentile (new implementation)
                        timeframe_1h["indicators"]["Hourly_High_Low_Percentile"] = self._calculate_high_low_percentile(df_1h)
                        
                        # 8. Hourly Volume Momentum (new implementation)
                        timeframe_1h["indicators"]["Hourly_Volume_Momentum"] = self._calculate_volume_momentum(df_1h["volume"])
                        
                        # 9. Fibonacci Levels (new implementation)
                        timeframe_1h["indicators"]["Fibonacci_Levels"] = self._calculate_fibonacci_levels(df_1h)
                        
                        # 10. Volume Profile Point of Control (new implementation)
                        timeframe_1h["indicators"]["Volume_Profile_POC"] = self._calculate_volume_profile_poc(df_1h)
                        
                        # 11. Pivot Points (new implementation)
                        timeframe_1h["indicators"]["Pivot_Points"] = self._calculate_pivot_points(df_1h)
                        
                        # 12. VWAP (Volume Weighted Average Price)
                        timeframe_1h["indicators"]["VWAP"] = self._calculate_vwap(df_1h)
                        
                        # 13. CVD (Cumulative Volume Delta)
                        timeframe_1h["indicators"]["CVD"] = self._calculate_cvd(df_1h)
                except Exception as e:
                    logger.error(f"Error calculating 1h indicators: {e}")
            
            # 4-hour timeframe indicators
            if klines_4h:
                try:
                    # Convert klines to DataFrame
                    df_4h = self._klines_to_dataframe(klines_4h)
                    
                    if len(df_4h) > 0:
                        # Initialize timeframe dict if not present
                        if "4h" not in data["timeframes"]:
                            data["timeframes"]["4h"] = {"price": {}, "indicators": {}}
                            
                        timeframe_4h = data["timeframes"]["4h"]
                        
                        # Extract and add price data
                        if "price" not in timeframe_4h:
                            timeframe_4h["price"] = {}
                            
                        # Add latest price data
                        latest = df_4h.iloc[-1]
                        timeframe_4h["price"]["open"] = float(latest["open"])
                        timeframe_4h["price"]["high"] = float(latest["high"])
                        timeframe_4h["price"]["low"] = float(latest["low"])
                        timeframe_4h["price"]["close"] = float(latest["close"])
                        timeframe_4h["price"]["volume"] = float(latest["volume"])
                        
                        # Initialize indicators dict if not present
                        if "indicators" not in timeframe_4h:
                            timeframe_4h["indicators"] = {}
                            
                        # Add 4h indicators
                        timeframe_4h["indicators"]["RSI"] = self._calculate_rsi(df_4h["close"])
                        timeframe_4h["indicators"]["ADX"] = self._calculate_adx(df_4h)
                        timeframe_4h["indicators"]["trend_strength"] = self._calculate_trend_strength(df_4h)
                        timeframe_4h["indicators"]["optimal_leverage"] = self._calculate_optimal_leverage(df_4h)
                        timeframe_4h["indicators"]["support_resistance"] = self._calculate_support_resistance(df_4h)
                        timeframe_4h["indicators"]["liquidation_risk"] = self._calculate_liquidation_risk(df_4h)
                except Exception as e:
                    logger.error(f"Error calculating 4h indicators: {e}")
            
            logger.info("✅ Technical indicators calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
    
    def _calculate_market_structure(self, df: pd.DataFrame) -> str:
        """
        Calculate market structure (trend direction)
        
        Args:
            df: Price dataframe with OHLCV columns
            
        Returns:
            Market structure description
        """
        try:
            # Use 20 & 50 EMA crossover for trend direction
            ema_20 = df['close'].ewm(span=20, adjust=False).mean()
            ema_50 = df['close'].ewm(span=50, adjust=False).mean()
            
            # Calculate higher highs, higher lows, lower highs, lower lows
            highs = df['high'].rolling(window=10).apply(lambda x: x.argmax() == 9)
            lows = df['low'].rolling(window=10).apply(lambda x: x.argmin() == 9)
            
            # Check last 5 candles for trend
            last_5_ema_diff = ema_20.iloc[-5:] - ema_50.iloc[-5:]
            bullish_ema = (last_5_ema_diff > 0).all()
            bearish_ema = (last_5_ema_diff < 0).all()
            
            # Check higher highs and lows for uptrend
            recent_highs = highs.iloc[-10:].sum()
            recent_lows = lows.iloc[-10:].sum()
            
            # Current price vs EMAs
            current_price = df['close'].iloc[-1]
            above_ema20 = current_price > ema_20.iloc[-1]
            above_ema50 = current_price > ema_50.iloc[-1]
            
            # Determine market structure
            if bullish_ema and above_ema20 and above_ema50 and recent_highs >= 2:
                return "STRONG_UPTREND"
            elif above_ema20 and above_ema50:
                return "UPTREND"
            elif bearish_ema and not above_ema20 and not above_ema50 and recent_lows >= 2:
                return "STRONG_DOWNTREND"
            elif not above_ema20 and not above_ema50:
                return "DOWNTREND"
            elif above_ema20 and not above_ema50:
                return "BULLISH_REVERSAL"
            elif not above_ema20 and above_ema50:
                return "BEARISH_REVERSAL"
            else:
                return "RANGE_BOUND"
                
        except Exception as e:
            logger.error(f"❌ Error calculating market structure: {str(e)}")
            return "UNKNOWN"
    
    def _calculate_historical_volatility(self, df: pd.DataFrame, period: int = 20) -> float:
        """
        Calculate historical volatility (HV)
        
        Args:
            df: OHLCV dataframe
            period: Period for calculation
            
        Returns:
            Historical volatility
        """
        try:
            # Calculate daily returns
            returns = df['close'].pct_change().dropna()
            
            # Calculate standard deviation of returns
            hv = returns.rolling(window=period).std().iloc[-1]
            
            # Annualize (multiply by sqrt of trading days)
            annualized_hv = hv * (365 ** 0.5)
            
            return float(annualized_hv)
        except Exception as e:
            logger.error(f"❌ Error calculating historical volatility: {str(e)}")
            return 0.01  # Default to 1% volatility
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR)
        
        Args:
            df: OHLCV dataframe
            period: Period for calculation
            
        Returns:
            ATR value
        """
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR
            atr = tr.rolling(window=period).mean().iloc[-1]
            
            return float(atr)
        except Exception as e:
            logger.error(f"❌ Error calculating ATR: {str(e)}")
            return 0.0
    
    def _calculate_stoch_rsi(self, prices: pd.Series, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Dict[str, float]:
        """
        Calculate Stochastic RSI
        
        Args:
            prices: Price series
            period: RSI period
            smooth_k: %K smoothing period
            smooth_d: %D smoothing period
            
        Returns:
            Stochastic RSI values
        """
        try:
            # Calculate RSI
            rsi = self._calculate_rsi_series(prices, period)
            
            # Calculate Stochastic RSI
            stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
            
            # Calculate %K and %D
            k = stoch_rsi.rolling(smooth_k).mean().iloc[-1] * 100
            d = stoch_rsi.rolling(smooth_d).mean().iloc[-1] * 100
            
            return {
                "k": float(k),
                "d": float(d)
            }
        except Exception as e:
            logger.error(f"❌ Error calculating Stochastic RSI: {str(e)}")
            return {"k": 50.0, "d": 50.0}
    
    def _calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """
        Calculate momentum score (combined indicator)
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            Momentum score between -100 and 100
        """
        try:
            # Get indicators
            rsi = self._calculate_rsi(df['close'])
            macd_result = self._calculate_macd(df['close'])
            williams_r = self._calculate_williams_r(df)
            ema_trend = 1 if df['close'].iloc[-1] > df['close'].ewm(span=20).mean().iloc[-1] else -1
            
            # Normalize RSI from 0-100 to -100 to 100
            rsi_norm = (rsi - 50) * 2
            
            # Normalize Williams %R from -100-0 to -100 to 100
            williams_norm = williams_r + 50
            
            # Extract the MACD value (handle both dictionary and float return types)
            if isinstance(macd_result, dict):
                macd_value = macd_result.get("MACD", 0)
            else:
                macd_value = macd_result
            
            # Calculate momentum score
            momentum_score = (rsi_norm * 0.4) + (macd_value * 30) + (williams_norm * 0.3) + (ema_trend * 10)
            
            # Ensure score is within -100 to 100
            momentum_score = max(-100, min(100, momentum_score))
            
            return float(momentum_score)
        except Exception as e:
            logger.error(f"❌ Error calculating momentum score: {str(e)}")
            return 0.0
    
    def _calculate_volume_momentum(self, volume: pd.Series, period: int = 14) -> float:
        """
        Calculate volume momentum indicator
        
        Args:
            volume: Series of volume data
            period: Calculation period
            
        Returns:
            Volume momentum score between -100 and 100
        """
        try:
            if len(volume) < period * 2:
                return 0.0  # Not enough data
            
            # Calculate the rate of change in volume
            volume_roc = ((volume / volume.shift(period)) - 1) * 100
            
            # Calculate the average volume
            avg_volume = volume.rolling(window=period).mean()
            
            # Calculate the ratio of current volume to average volume
            volume_ratio = volume / avg_volume
            
            # Get the most recent values
            recent_roc = volume_roc.iloc[-1]
            recent_ratio = volume_ratio.iloc[-1]
            
            # Calculate the volume momentum score
            # Higher weight on recent ratio, lower on rate of change
            momentum_score = (recent_ratio - 1) * 50 + recent_roc * 0.5
            
            # Ensure the score is within -100 to 100
            momentum_score = max(-100, min(100, momentum_score))
            
            return round(momentum_score, 2)
        except Exception as e:
            logger.error(f"Error calculating volume momentum: {str(e)}")
            return 0.0
    
    def _calculate_parabolic_sar(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                                acceleration: float = 0.02, maximum: float = 0.2) -> str:
        """
        Calculate Parabolic SAR (Stop and Reverse) indicator
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            acceleration: Acceleration factor (default: 0.02)
            maximum: Maximum acceleration factor (default: 0.2)
            
        Returns:
            Position: "ABOVE" (bearish) or "BELOW" (bullish)
        """
        try:
            if len(high) < 30:  # Need sufficient data
                return "NEUTRAL"

            # Initialize variables
            trend = 1  # 1 for uptrend, -1 for downtrend
            af = acceleration  # Acceleration Factor
            extreme_point = high.iloc[0]  # Extreme Point
            sar = low.iloc[0]  # SAR start point (begin with low price)
            sar_values = [sar]

            # Calculate SAR values
            for i in range(1, len(high)):
                # Prior trend was up
                if trend == 1:
                    # SAR = Prior SAR + Prior AF * (Prior EP - Prior SAR)
                    sar = sar_values[-1] + af * (extreme_point - sar_values[-1])
                    
                    # Make sure SAR is not above the prior two lows
                    if i >= 2:
                        sar = min(sar, low.iloc[i-1], low.iloc[i-2])
                    
                    # Check if current price is below SAR
                    if low.iloc[i] < sar:
                        # Switch to downtrend
                        trend = -1
                        sar = extreme_point  # SAR becomes the prior extreme point
                        extreme_point = low.iloc[i]  # EP becomes current low
                        af = acceleration  # Reset AF
                    else:
                        # Continue uptrend
                        if high.iloc[i] > extreme_point:
                            extreme_point = high.iloc[i]  # EP becomes current high
                            af = min(af + acceleration, maximum)  # Increase AF
                
                # Prior trend was down
                else:
                    # SAR = Prior SAR - Prior AF * (Prior SAR - Prior EP)
                    sar = sar_values[-1] - af * (sar_values[-1] - extreme_point)
                    
                    # Make sure SAR is not below the prior two highs
                    if i >= 2:
                        sar = max(sar, high.iloc[i-1], high.iloc[i-2])
                    
                    # Check if current price is above SAR
                    if high.iloc[i] > sar:
                        # Switch to uptrend
                        trend = 1
                        sar = extreme_point  # SAR becomes the prior extreme point
                        extreme_point = high.iloc[i]  # EP becomes current high
                        af = acceleration  # Reset AF
                    else:
                        # Continue downtrend
                        if low.iloc[i] < extreme_point:
                            extreme_point = low.iloc[i]  # EP becomes current low
                            af = min(af + acceleration, maximum)  # Increase AF
                
                sar_values.append(sar)
            
            # Determine position based on the last price and SAR value
            last_sar = sar_values[-1]
            last_close = close.iloc[-1]
            
            if last_close > last_sar:
                return "BELOW"  # Bullish signal, SAR is below price
            else:
                return "ABOVE"  # Bearish signal, SAR is above price
                
        except Exception as e:
            logging.error(f"Error calculating Parabolic SAR: {str(e)}")
            return "NEUTRAL"  # Default in case of error
    
    def _identify_key_levels(self, df: pd.DataFrame) -> List[Dict[str, float]]:
        """
        Identify key support and resistance levels
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            List of key price levels
        """
        try:
            levels = []
            
            # Get highs and lows
            highs = df['high'].values
            lows = df['low'].values
            
            # Identify pivot highs and lows
            for i in range(2, len(df) - 2):
                # Resistance (pivot high)
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
                   highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    levels.append({"price": float(highs[i]), "type": "resistance"})
                
                # Support (pivot low)
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
                   lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    levels.append({"price": float(lows[i]), "type": "support"})
            
            # Sort by price
            levels.sort(key=lambda x: x["price"])
            
            # Return top 5 most significant levels
            return levels[:5]
        except Exception as e:
            logger.error(f"❌ Error identifying key levels: {str(e)}")
            return []
    
    def _calculate_funding_impact(self, funding_rate: float) -> str:
        """
        Calculate impact of funding rate on position
        
        Args:
            funding_rate: Current funding rate
            
        Returns:
            Impact description
        """
        try:
            # Interpret funding rate
            abs_rate = abs(funding_rate)
            
            if abs_rate < 0.0001:  # Less than 0.01%
                impact = "NEGLIGIBLE"
            elif abs_rate < 0.0005:  # Less than 0.05%
                impact = "LOW"
            elif abs_rate < 0.001:  # Less than 0.1%
                impact = "MODERATE"
            elif abs_rate < 0.003:  # Less than 0.3%
                impact = "HIGH"
            else:
                impact = "EXTREME"
            
            # Add direction
            if funding_rate > 0:
                return f"LONG_PAYING_{impact}"
            else:
                return f"SHORT_PAYING_{impact}"
        except Exception as e:
            logger.error(f"❌ Error calculating funding impact: {str(e)}")
            return "UNKNOWN"
    
    def _calculate_liquidity_zones(self, df: pd.DataFrame) -> List[Dict[str, float]]:
        """
        Calculate liquidity zones for futures trading
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            List of liquidity zones
        """
        try:
            zones = []
            
            # Get recent highs and lows (last 30 candles)
            recent_df = df.tail(30)
            
            # Find clusters of highs and lows
            for i in range(len(recent_df) - 1):
                # Skip if already part of a zone
                if any(z["lower"] <= recent_df['high'].iloc[i] <= z["upper"] for z in zones):
                    continue
                
                # Count similar highs and lows
                similar_count = 0
                for j in range(i + 1, len(recent_df)):
                    # Check if highs are within 0.5% of each other
                    if abs(recent_df['high'].iloc[j] - recent_df['high'].iloc[i]) / recent_df['high'].iloc[i] < 0.005:
                        similar_count += 1
                
                # If we have at least 2 similar levels, it's a liquidity zone
                if similar_count >= 2:
                    price = recent_df['high'].iloc[i]
                    zones.append({
                        "price": float(price),
                        "lower": float(price * 0.997),
                        "upper": float(price * 1.003),
                        "type": "resistance",
                        "strength": similar_count
                    })
            
            # Sort zones by strength
            zones.sort(key=lambda x: x["strength"], reverse=True)
            
            return zones[:3]  # Return top 3 zones
        except Exception as e:
            logger.error(f"❌ Error calculating liquidity zones: {str(e)}")
            return []
    
    def _calculate_order_flow_imbalance(self, df: pd.DataFrame) -> float:
        """
        Calculate order flow imbalance score
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            Order flow imbalance score between -100 and 100
        """
        try:
            # Calculate delta (close - open)
            delta = df['close'] - df['open']
            
            # Determine bullish and bearish candles
            bullish = delta > 0
            bearish = delta < 0
            
            # Calculate volume-weighted delta
            weighted_delta = delta * df['volume']
            
            # Calculate scores for last 20 candles
            recent_weighted_delta = weighted_delta.tail(20).sum()
            recent_volume = df['volume'].tail(20).sum()
            
            # Normalize to -100 to 100 scale
            if recent_volume > 0:
                imbalance_score = (recent_weighted_delta / recent_volume) * 10000
                # Cap at -100 to 100
                imbalance_score = max(-100, min(100, imbalance_score))
            else:
                imbalance_score = 0
            
            return float(imbalance_score)
        except Exception as e:
            logger.error(f"❌ Error calculating order flow imbalance: {str(e)}")
            return 0.0
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate trend strength indicators
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            Trend strength metrics
        """
        try:
            # Calculate ADX
            adx = self._calculate_adx(df)
            
            # Calculate directional movement
            high_diff = df['high'].diff()
            low_diff = df['low'].diff()
            
            # Positive Directional Movement (+DM)
            pdm = high_diff.copy()
            pdm[pdm < 0] = 0
            pdm[(high_diff <= 0) | (high_diff < low_diff.abs())] = 0
            
            # Negative Directional Movement (-DM)
            ndm = low_diff.abs().copy()
            ndm[ndm < 0] = 0
            ndm[(low_diff >= 0) | (low_diff.abs() < high_diff)] = 0
            
            # Smooth with EMA
            pdm_ema = pdm.ewm(span=14).mean().iloc[-1]
            ndm_ema = ndm.ewm(span=14).mean().iloc[-1]
            
            # Calculate +DI and -DI
            tr = self._calculate_atr(df)
            pdi = 100 * pdm_ema / tr if tr > 0 else 0
            ndi = 100 * ndm_ema / tr if tr > 0 else 0
            
            # Calculate DX
            dx = 100 * abs(pdi - ndi) / (pdi + ndi) if (pdi + ndi) > 0 else 0
            
            return {
                "adx": float(adx),
                "pdi": float(pdi),
                "ndi": float(ndi),
                "dx": float(dx)
            }
        except Exception as e:
            logger.error(f"❌ Error calculating trend strength: {str(e)}")
            return {"adx": 0.0, "pdi": 0.0, "ndi": 0.0, "dx": 0.0}
    
    def _calculate_optimal_leverage(self, df: pd.DataFrame) -> float:
        """
        Calculate optimal leverage based on volatility
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            Optimal leverage (1-20)
        """
        try:
            # Calculate ATR as % of price
            atr = self._calculate_atr(df)
            current_price = df['close'].iloc[-1]
            atr_pct = atr / current_price
            
            # Volatility-based leverage calculation
            # Higher volatility = lower leverage
            if atr_pct > 0.03:  # More than 3% daily movement
                optimal_leverage = 2
            elif atr_pct > 0.02:  # More than 2% daily movement
                optimal_leverage = 5
            elif atr_pct > 0.01:  # More than 1% daily movement
                optimal_leverage = 10
            elif atr_pct > 0.005:  # More than 0.5% daily movement
                optimal_leverage = 15
            else:
                optimal_leverage = 20
            
            return float(optimal_leverage)
        except Exception as e:
            logger.error(f"❌ Error calculating optimal leverage: {str(e)}")
            return 5.0  # Default to 5x
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> List[Dict[str, float]]:
        """
        Calculate support and resistance levels
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            Support and resistance levels
        """
        try:
            levels = []
            
            # Current price
            current_price = df['close'].iloc[-1]
            
            # Pivot points
            pivot_points = self._calculate_pivot_points(df)
            
            # Add pivot points to levels
            for key, value in pivot_points.items():
                level_type = "support" if value < current_price else "resistance"
                levels.append({"price": value, "type": level_type, "source": key})
            
            # Add key levels from price action
            key_levels = self._identify_key_levels(df)
            for level in key_levels:
                if not any(abs(level["price"] - l["price"])/l["price"] < 0.01 for l in levels):
                    levels.append(level)
            
            # Sort by price
            levels.sort(key=lambda x: x["price"])
            
            return levels
        except Exception as e:
            logger.error(f"❌ Error calculating support/resistance: {str(e)}")
            return []
    
    def _calculate_liquidation_risk(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate potential liquidation levels based on volatility
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            Liquidation risk metrics
        """
        try:
            # Current price
            current_price = df['close'].iloc[-1]
            
            # Calculate ATR
            atr = self._calculate_atr(df)
            
            # Calculate potential liquidation levels for different leverages
            liq_levels = {}
            
            for leverage in [5, 10, 20]:
                # Approximate liquidation distance (maintenance margin is typically around 0.5%)
                # 100% / leverage - maintenance margin
                liq_distance_pct = (1 / leverage - 0.005)
                
                # Calculate liquidation price
                liq_price_long = current_price * (1 - liq_distance_pct)
                liq_price_short = current_price * (1 + liq_distance_pct)
                
                # Calculate risk in terms of ATR
                risk_long_atr = (current_price - liq_price_long) / atr
                risk_short_atr = (liq_price_short - current_price) / atr
                
                liq_levels[f"{leverage}x"] = {
                    "long_liq_price": float(liq_price_long),
                    "short_liq_price": float(liq_price_short),
                    "long_risk_atr": float(risk_long_atr),
                    "short_risk_atr": float(risk_short_atr)
                }
            
            return liq_levels
        except Exception as e:
            logger.error(f"❌ Error calculating liquidation risk: {str(e)}")
            return {}

    def _add_sentiment_data(self, data: Dict[str, Any]) -> None:
        """
        Add sentiment data from Santiment API
        
        Args:
            data: Market data
        """
        try:
            # If Santiment API is not available, add default sentiment data
            if self.santiment_api is None:
                logger.info("Santiment API not available. Using default sentiment values.")
                data["sentiment"] = self._add_default_sentiment(data)
                return
                
            # Get symbol
            symbol = data.get("symbol", "BTCUSDT")
            slug = "bitcoin"  # Default to bitcoin
            
            # Map common symbols to Santiment slugs
            symbol_to_slug = {
                "BTCUSDT": "bitcoin",
                "ETHUSDT": "ethereum",
                "BNBUSDT": "binance-coin",
                "ADAUSDT": "cardano",
                "SOLUSDT": "solana",
                "DOGEUSDT": "dogecoin"
            }
            
            if symbol in symbol_to_slug:
                slug = symbol_to_slug[symbol]
                
            # Fetch sentiment data from Santiment
            sentiment_data = self.santiment_api.fetch_sentiment_data(slug)
            if not sentiment_data:
                # Use default sentiment data if Santiment API returns empty data
                logger.warning(f"Santiment API returned empty data for {slug}")
                data["sentiment"] = self._add_default_sentiment(data)
                return
                
            # Add sentiment data
            data["sentiment"] = sentiment_data
            
        except Exception as e:
            logger.error(f"Error fetching Santiment data: {str(e)}")
            # Add default sentiment data on error
            data["sentiment"] = self._add_default_sentiment(data)
            
    def _add_default_sentiment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add default sentiment data when Santiment API is not available
        
        Args:
            data: Market data
        
        Returns:
            Sentiment data dictionary
        """
        # Use market data to derive sentiment
        sentiment_data = {}
        
        # Get current price movement from 1h timeframe if available
        price_trend = 0
        trend_desc = "Neutral"
        
        try:
            if "timeframes" in data and "1h" in data["timeframes"]:
                timeframe_1h = data["timeframes"]["1h"]
                
                # Try multiple approaches to extract price data
                if "price" in timeframe_1h:
                    # First approach: Direct price values
                    if all(k in timeframe_1h["price"] for k in ["open", "close"]):
                        open_price = float(timeframe_1h["price"]["open"])
                        close_price = float(timeframe_1h["price"]["close"])
                        price_trend = (close_price - open_price) / open_price if open_price > 0 else 0
                    
                    # Second approach: Klines data
                    elif "klines" in timeframe_1h["price"] and timeframe_1h["price"]["klines"]:
                        klines = timeframe_1h["price"]["klines"]
                        if len(klines) > 1:
                            # Check klines structure
                            if isinstance(klines[0], list) and len(klines[0]) >= 5:
                                # Standard klines format: [timestamp, open, high, low, close, volume]
                                open_price = float(klines[0][1])
                                close_price = float(klines[-1][4])
                            elif isinstance(klines[0], dict) and all(k in klines[0] for k in ["open", "close"]):
                                # Dictionary format: {"open": value, "close": value, ...}
                                open_price = float(klines[0]["open"])
                                close_price = float(klines[-1]["close"])
                            else:
                                # Fallback to using first and last items regardless of structure
                                logger.warning("Unknown klines structure, using fallback method")
                                # Try to extract first and last items somehow
                                open_price = 0
                                close_price = 0
                                for k in klines[0]:
                                    if isinstance(klines[0][k], (int, float)) and k != "timestamp":
                                        open_price = float(klines[0][k])
                                        break
                                for k in klines[-1]:
                                    if isinstance(klines[-1][k], (int, float)) and k != "timestamp":
                                        close_price = float(klines[-1][k])
                                        break
                            
                            price_trend = (close_price - open_price) / open_price if open_price > 0 else 0
            
            # Classify trend based on price change
            if price_trend > 0.02:
                trend_desc = "Bullish"
            elif price_trend < -0.02:
                trend_desc = "Bearish"
            else:
                trend_desc = "Neutral"
                
        except Exception as e:
            logger.error(f"Error calculating price trend: {e}")
            logger.debug(f"Timeframe data structure: {data.get('timeframes', {}).get('1h', {})}")
            # Continue with neutral trend as fallback
            price_trend = 0
            trend_desc = "Neutral"
        
        # Get funding rate from futures data if available
        funding_rate = 0.0
        if "futures_data" in data and "funding_rate" in data["futures_data"]:
            funding_rate = data["futures_data"]["funding_rate"]
            sentiment_data["funding_rate"] = funding_rate
            
            # Interpret funding rate
            if funding_rate > 0.0008:  # High positive funding rate (>0.08% per 8h)
                sentiment_data["funding_sentiment"] = "Extremely Bullish"
            elif funding_rate > 0.0003:  # Moderate positive funding rate
                sentiment_data["funding_sentiment"] = "Bullish"
            elif funding_rate < -0.0008:  # High negative funding rate
                sentiment_data["funding_sentiment"] = "Extremely Bearish"
            elif funding_rate < -0.0003:  # Moderate negative funding rate
                sentiment_data["funding_sentiment"] = "Bearish"
            else:
                sentiment_data["funding_sentiment"] = "Neutral"
        
        # Fear & Greed Index (based on overall market sentiment)
        # In real implementation, this would be fetched from an external source
        # For now, we'll use a value that aligns with the price trend
        if price_trend > 0.05:
            fear_greed = random.randint(75, 95)  # Extreme Greed
        elif price_trend > 0.02:
            fear_greed = random.randint(60, 75)  # Greed
        elif price_trend < -0.05:
            fear_greed = random.randint(5, 25)  # Extreme Fear
        elif price_trend < -0.02:
            fear_greed = random.randint(25, 40)  # Fear
        else:
            fear_greed = random.randint(40, 60)  # Neutral
            
        sentiment_data["fear_greed_index"] = fear_greed
        
        # Categorize Fear & Greed Index
        if fear_greed >= 75:
            sentiment_data["fear_greed_category"] = "Extreme Greed"
        elif fear_greed >= 60:
            sentiment_data["fear_greed_category"] = "Greed"
        elif fear_greed <= 25:
            sentiment_data["fear_greed_category"] = "Extreme Fear"
        elif fear_greed <= 40:
            sentiment_data["fear_greed_category"] = "Fear"
        else:
            sentiment_data["fear_greed_category"] = "Neutral"
        
        # Add social media sentiment - normally would be from Twitter, Reddit, etc.
        sentiment_data["social_sentiment"] = self._generate_sentiment_score(trend_desc)
        sentiment_data["social_volume"] = self._generate_social_volume(trend_desc)
        
        # Add news sentiment score (usually from news APIs or sentiment analysis)
        sentiment_data["news_sentiment"] = self._generate_sentiment_score(trend_desc)
        sentiment_data["news_sentiment_score"] = self._generate_sentiment_score(trend_desc, as_float=True)
        
        # Add whale activity (usually from on-chain data)
        sentiment_data["whale_activity"] = self._generate_whale_activity()
        
        return sentiment_data
        
    def _generate_sentiment_score(self, trend: str, as_float: bool = False) -> Union[str, float]:
        """
        Generate a sentiment score based on trend
        
        Args:
            trend: Price trend descriptor ("Bullish", "Bearish", "Neutral")
            as_float: Whether to return as float (-1 to 1) instead of string
            
        Returns:
            Sentiment score as string or float
        """
        if trend == "Bullish":
            score = random.uniform(0.3, 0.9)
        elif trend == "Bearish":
            score = random.uniform(-0.9, -0.3)
        else:
            score = random.uniform(-0.3, 0.3)
            
        if as_float:
            return round(score, 2)
            
        # Convert to descriptive string
        if score > 0.6:
            return "Very Positive"
        elif score > 0.2:
            return "Positive"
        elif score < -0.6:
            return "Very Negative"
        elif score < -0.2:
            return "Negative"
        else:
            return "Neutral"
            
    def _generate_social_volume(self, trend: str) -> int:
        """
        Generate a social volume value based on trend
        
        Args:
            trend: Price trend descriptor
            
        Returns:
            Social volume value (number of mentions)
        """
        # Base volume - typical daily mentions
        base_volume = 10000
        
        # Adjust based on trend
        if trend == "Bullish":
            # More social activity during bull markets
            multiplier = random.uniform(1.2, 2.0)
        elif trend == "Bearish":
            # Less social activity during bearish trends, but still elevated
            multiplier = random.uniform(0.8, 1.5)
        else:
            # Normal activity during neutral periods
            multiplier = random.uniform(0.7, 1.3)
            
        return int(base_volume * multiplier)
        
    def _generate_whale_activity(self) -> Dict[str, Any]:
        """
        Generate simulated whale activity data
        
        Returns:
            Dictionary with whale activity metrics
        """
        # Generate random activity levels
        large_transactions = random.randint(50, 500)
        exchange_inflows = random.randint(1000, 10000)
        exchange_outflows = random.randint(1000, 10000)
        
        # Calculate net flow
        net_flow = exchange_outflows - exchange_inflows
        
        # Determine if accumulation or distribution
        if net_flow > 1000:
            pattern = "Strong Accumulation"
        elif net_flow > 0:
            pattern = "Moderate Accumulation"
        elif net_flow < -1000:
            pattern = "Strong Distribution"
        elif net_flow < 0:
            pattern = "Moderate Distribution"
        else:
            pattern = "Neutral"
            
        # Package data
        return {
            "large_transactions_24h": large_transactions,
            "exchange_inflows_btc": exchange_inflows,
            "exchange_outflows_btc": exchange_outflows,
            "net_flow_btc": net_flow,
            "pattern": pattern
        }
    
    def _add_macro_factors(self, data: Dict[str, Any]) -> None:
        """
        Add macroeconomic factors
        
        Args:
            data: Market data to enrich
        """
        try:
            # Initialize macro_factors dictionary if it doesn't exist
            if "macro_factors" not in data:
                data["macro_factors"] = {}
            
            # Calculate Exchange Reserves (in a real system, this would come from on-chain data)
            if "timeframes" in data and "1h" in data["timeframes"]:
                timeframe_1h = data["timeframes"]["1h"]
                if "price" in timeframe_1h and "volume" in timeframe_1h["price"]:
                    exchange_reserves = self._calculate_exchange_reserves(
                        timeframe_1h["price"]["volume"]
                    )
                    data["macro_factors"]["exchange_reserves"] = exchange_reserves
            
            # Calculate Bitcoin Hash Rate (in a real system, this would come from blockchain data)
            btc_hash_rate = self._calculate_btc_hash_rate()
            data["macro_factors"]["btc_hash_rate"] = btc_hash_rate
            
            # Calculate FOMC Impact (in a real system, this would come from economic calendars)
            fomc_impact = self._calculate_fomc_impact(data)
            data["macro_factors"]["fomc_event_impact"] = fomc_impact
            
            # Calculate Traditional Markets Impact
            traditional_markets = self._calculate_traditional_markets_impact(data)
            data["macro_factors"]["traditional_markets"] = traditional_markets
            
            # Calculate Bond Yields Impact
            bond_yields = self._calculate_bond_yields_impact()
            data["macro_factors"]["bond_yields"] = bond_yields
            
            # Calculate Inflation Expectations
            inflation = self._calculate_inflation_expectations()
            data["macro_factors"]["inflation_expectations"] = inflation
            
        except Exception as e:
            logger.error(f"Error adding macro factors: {str(e)}")
            # Ensure we have some default values
            data["macro_factors"] = {
                "exchange_reserves": 2e6,
                "btc_hash_rate": 325e12,
                "fomc_event_impact": "Neutral",
            "traditional_markets": {
                    "SP500_trend": "Neutral",
                    "DXY_index": 100.0
            },
            "bond_yields": {
                    "US10Y": 3.0,
                    "real_yield": 0.0
                },
                "inflation_expectations": 3.0
            }
    
    def _calculate_exchange_reserves(self, volume: float) -> float:
        """Calculate Exchange Reserves based on volume"""
        try:
            # In a real implementation, this would use on-chain data
            # For now, estimate based on volume with some baseline
            base_reserves = 2e6  # 2 million BTC baseline
            volume_factor = volume / 1000  # Scale volume to reasonable range
            return round(base_reserves + volume_factor, 1)
        except Exception as e:
            logger.error(f"Error calculating Exchange Reserves: {str(e)}")
            return 2e6
    
    def _calculate_btc_hash_rate(self) -> float:
        """Calculate Bitcoin Hash Rate"""
        try:
            # In a real implementation, this would fetch from blockchain APIs
            # For now, return a reasonable estimate of current hash rate
            return 325e12  # 325 TH/s
        except Exception as e:
            logger.error(f"Error calculating BTC Hash Rate: {str(e)}")
            return 325e12
    
    def _calculate_fomc_impact(self, data: Dict[str, Any]) -> str:
        """Calculate FOMC Event Impact"""
        try:
            # In a real implementation, this would analyze recent FOMC statements
            # For now, base it on price action and volume
            if "timeframes" in data and "1h" in data["timeframes"]:
                timeframe_1h = data["timeframes"]["1h"]
                if "price" in timeframe_1h:
                    price_data = timeframe_1h["price"]
                    if all(k in price_data for k in ["open", "close", "volume"]):
                        price_change = (price_data["close"] - price_data["open"]) / price_data["open"]
                        
                        if abs(price_change) < 0.01:  # Less than 1% change
                            return "Neutral"
                        elif price_change > 0:
                            return "Bullish"
                        else:
                            return "Bearish"
            
            return "Neutral"
            
        except Exception as e:
            logger.error(f"Error calculating FOMC Impact: {str(e)}")
            return "Neutral"
    
    def _calculate_traditional_markets_impact(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Traditional Markets Impact"""
        try:
            result = {
                "SP500_trend": "Neutral",
                "DXY_index": 100.0
            }
            
            # In a real implementation, this would fetch real market data
            # For now, base it on Bitcoin's price action
            if "timeframes" in data and "1h" in data["timeframes"]:
                timeframe_1h = data["timeframes"]["1h"]
                if "price" in timeframe_1h:
                    price_data = timeframe_1h["price"]
                    if "close" in price_data and "open" in price_data:
                        price_change = (price_data["close"] - price_data["open"]) / price_data["open"]
                        
                        # Assume inverse correlation with DXY
                        dxy_change = -price_change
                        result["DXY_index"] = round(100 * (1 + dxy_change), 1)
                        
                        # Assume positive correlation with S&P 500
                        if abs(price_change) < 0.01:
                            result["SP500_trend"] = "Neutral"
                        elif price_change > 0:
                            result["SP500_trend"] = "Bullish"
                        else:
                            result["SP500_trend"] = "Bearish"
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating Traditional Markets Impact: {str(e)}")
            return {
                "SP500_trend": "Neutral",
                "DXY_index": 100.0
            }
    
    def _calculate_bond_yields_impact(self) -> Dict[str, float]:
        """Calculate Bond Yields Impact"""
        try:
            # In a real implementation, this would fetch real bond market data
            return {
                "US10Y": 3.0,  # 3% yield
                "real_yield": 0.5  # 0.5% real yield
            }
        except Exception as e:
            logger.error(f"Error calculating Bond Yields Impact: {str(e)}")
            return {
                "US10Y": 3.0,
                "real_yield": 0.0
            }
    
    def _calculate_inflation_expectations(self) -> float:
        """Calculate Inflation Expectations"""
        try:
            # In a real implementation, this would use market-based expectations
            return 3.0  # 3% expected inflation
        except Exception as e:
            logger.error(f"Error calculating Inflation Expectations: {str(e)}")
            return 3.0

    def _add_onchain_metrics(self, data: Dict[str, Any]) -> None:
        """Add Bitcoin on-chain metrics"""
        data["on_chain"] = {
            "exchange_flows": {
                "inflow": round(random.uniform(1000, 5000), 1),
                "outflow": round(random.uniform(1000, 5000), 1)
            },
            "miner_activity": {
                "reserve": round(random.uniform(1e6, 2e6), 1),
                "transfer_to_exchanges": round(random.uniform(100, 1000), 1)
            },
            "wallet_metrics": {
                "whale_transactions": random.randint(50, 200),
                "hodl_waves": round(random.uniform(0.5, 0.8), 2)
            },
            "SOPR": round(random.uniform(0.9, 1.1), 3)  # Spent Output Profit Ratio
        } 

    def _add_risk_metrics(self, data: Dict[str, Any]) -> None:
        """Add market risk metrics"""
        data["risk_metrics"] = {
            "VaR_95": round(random.uniform(2.0, 8.0), 1),  # Value at Risk (95% confidence)
            "correlation_matrix": {
                "SP500": round(random.uniform(-0.5, 0.5), 2),
                "gold": round(random.uniform(-0.3, 0.3), 2)
            },
            "liquidity_risk": round(random.uniform(0.1, 0.9), 2)
        } 
        
    @lru_cache(maxsize=32)
    def _calculate_expensive_indicator(self, symbol: str, timeframe: str) -> float:
        """
        Calculate an expensive indicator with caching
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for calculation (e.g., '5m', '1h')
            
        Returns:
            Calculated indicator value
        """
        logger.info(f"Calculating expensive indicator for {symbol} on {timeframe} timeframe")
        # Expensive calculation here - this is a placeholder
        result = random.uniform(0.01, 0.5)
        return result 

    def _add_futures_metrics(self, data: Dict[str, Any]) -> None:
        """
        Add futures-specific metrics to market data
        
        Args:
            data: Market data to enrich
        """
        if "futures_data" not in data:
            return
            
        futures_data = data["futures_data"]
        
        # Calculate funding rate impact
        if "funding_rate" in futures_data:
            funding_rate = futures_data["funding_rate"]
            
            # Determine if funding rate is positive or negative
            if funding_rate > 0:
                funding_sentiment = "Bearish"  # Positive funding rate means longs pay shorts
            elif funding_rate < 0:
                funding_sentiment = "Bullish"  # Negative funding rate means shorts pay longs
            else:
                funding_sentiment = "Neutral"
                
            # Calculate funding rate magnitude (absolute value)
            funding_magnitude = abs(funding_rate)
            
            # Categorize funding rate magnitude
            if funding_magnitude < 0.0001:  # 0.01% per 8 hours
                magnitude_category = "Very Low"
            elif funding_magnitude < 0.0005:  # 0.05% per 8 hours
                magnitude_category = "Low"
            elif funding_magnitude < 0.001:  # 0.1% per 8 hours
                magnitude_category = "Moderate"
            elif funding_magnitude < 0.002:  # 0.2% per 8 hours
                magnitude_category = "High"
            else:
                magnitude_category = "Very High"
                
            # Add funding rate analysis
            futures_data["funding_rate_analysis"] = {
                "sentiment": funding_sentiment,
                "magnitude": magnitude_category,
                "annualized_rate": funding_rate * 3 * 365  # 3 funding periods per day * 365 days
            }
            
        # Calculate open interest impact
        if "open_interest" in futures_data:
            open_interest = futures_data["open_interest"]
            
            # We need historical data to determine if open interest is increasing or decreasing
            # For now, we'll just provide the raw value
            futures_data["open_interest_analysis"] = {
                "value": open_interest,
                "formatted": self._format_large_number(open_interest)
            }
            
        # Add leverage analysis if available
        if "leverage_brackets" in futures_data:
            leverage_brackets = futures_data.get("leverage_brackets", [])
            
            if leverage_brackets and isinstance(leverage_brackets, list) and len(leverage_brackets) > 0:
                # Extract the first bracket (lowest notional value)
                first_bracket = leverage_brackets[0]
                
                if "brackets" in first_bracket and len(first_bracket["brackets"]) > 0:
                    max_leverage = first_bracket["brackets"][0].get("initialLeverage", 0)
                    futures_data["max_leverage"] = max_leverage
                    
                    # Determine if current market allows high leverage
                    if max_leverage >= 100:
                        leverage_category = "Very High"
                    elif max_leverage >= 50:
                        leverage_category = "High"
                    elif max_leverage >= 20:
                        leverage_category = "Moderate"
                    elif max_leverage >= 10:
                        leverage_category = "Low"
                    else:
                        leverage_category = "Very Low"
                        
                    futures_data["leverage_analysis"] = {
                        "max_leverage": max_leverage,
                        "category": leverage_category
                    }

    def _format_large_number(self, number: float) -> str:
        """
        Format large numbers with K, M, B suffixes
        
        Args:
            number: Number to format
            
        Returns:
            Formatted number string
        """
        if number >= 1_000_000_000:
            return f"{number / 1_000_000_000:.2f}B"
        elif number >= 1_000_000:
            return f"{number / 1_000_000:.2f}M"
        elif number >= 1_000:
            return f"{number / 1_000:.2f}K"
        else:
            return f"{number:.2f}" 

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """
        Calculate the Relative Strength Index (RSI)
        
        Args:
            prices: Series of prices
            period: RSI period
            
        Returns:
            RSI value
        """
        try:
            if len(prices) < period + 10:
                return 50.0  # Default to neutral if not enough data
                
            # Calculate price changes
            delta = prices.diff().dropna()
            
            # Create two series: one for gains, one for losses
            gains = delta.copy()
            losses = delta.copy()
            
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = abs(losses)
            
            # Calculate average gains and losses
            avg_gain = gains.rolling(window=period).mean()
            avg_loss = losses.rolling(window=period).mean()
            
            # Calculate RS
            rs = avg_gain / avg_loss
            
            # Calculate RSI
            rsi = 100 - (100 / (1 + rs))
            
            # Return the last RSI value, rounded to 2 decimal places
            return round(rsi.iloc[-1], 2)
        except Exception as e:
            logging.error(f"Error calculating RSI: {str(e)}")
            return 50.0  # Default to neutral in case of error
    
    def _calculate_rsi_series(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate the RSI for the entire price series
        
        Args:
            prices: Series of prices
            period: RSI period
            
        Returns:
            Series of RSI values
        """
        try:
            if len(prices) < period + 10:
                return pd.Series([50.0] * len(prices))  # Default to neutral if not enough data
                
            # Calculate price changes
            delta = prices.diff().dropna()
            
            # Create two series: one for gains, one for losses
            gains = delta.copy()
            losses = delta.copy()
            
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = abs(losses)
            
            # Calculate average gains and losses
            avg_gain = gains.rolling(window=period).mean()
            avg_loss = losses.rolling(window=period).mean()
            
            # Calculate RS
            rs = avg_gain / avg_loss
            
            # Calculate RSI
            rsi = 100 - (100 / (1 + rs))
            
            # Fill NaN values with 50 (neutral)
            rsi = rsi.fillna(50)
            
            return rsi
        except Exception as e:
            logging.error(f"Error calculating RSI series: {str(e)}")
            return pd.Series([50.0] * len(prices))  # Default to neutral in case of error
    
    def _calculate_bollinger_width(self, prices: pd.Series, period: int = 20, num_std: float = 2.0) -> float:
        """
        Calculate Bollinger Bands width
        
        Args:
            prices: Series of prices
            period: Period for moving average
            num_std: Number of standard deviations
            
        Returns:
            Bollinger Bands width as a percentage of the middle band
        """
        try:
            if len(prices) < period:
                return 0.0  # Default if not enough data
                
            # Calculate middle band (SMA)
            middle_band = prices.rolling(window=period).mean()
            
            # Calculate standard deviation
            std_dev = prices.rolling(window=period).std()
            
            # Calculate upper and lower bands
            upper_band = middle_band + (std_dev * num_std)
            lower_band = middle_band - (std_dev * num_std)
            
            # Calculate bandwidth
            bandwidth = (upper_band - lower_band) / middle_band
            
            # Return the last bandwidth value, rounded to 4 decimal places
            return round(bandwidth.iloc[-1], 4)
        except Exception as e:
            logging.error(f"Error calculating Bollinger Bands width: {str(e)}")
            return 0.0  # Default in case of error
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Williams %R indicator
        
        Args:
            df: DataFrame with OHLC data
            period: Period for the calculation
            
        Returns:
            Williams %R value (-100 to 0)
        """
        try:
            if len(df) < period:
                return -50.0  # Default if not enough data
                
            # Get high and low for the period
            high = df['high'].rolling(window=period).max()
            low = df['low'].rolling(window=period).min()
            
            # Get current close
            close = df['close']
            
            # Calculate Williams %R
            williams_r = ((high - close) / (high - low)) * -100
            
            # Return the most recent value, rounded to 2 decimal places
            return round(williams_r.iloc[-1], 2)
        except Exception as e:
            logging.error(f"Error calculating Williams %R: {str(e)}")
            return -50.0  # Default to middle of the range in case of error
    
    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """
        Calculate Volume Weighted Average Price (VWAP)
        
        Args:
            df: DataFrame with OHLC and volume data
            
        Returns:
            VWAP value
        """
        try:
            if len(df) < 2:
                return df['close'].iloc[-1] if len(df) > 0 else 0.0  # Default if not enough data
                
            # Make sure we have required columns
            if 'close' not in df.columns or 'volume' not in df.columns:
                logging.error("Missing required columns for VWAP calculation")
                return df['close'].iloc[-1] if 'close' in df.columns and len(df) > 0 else 0.0
            
            # Calculate typical price (TP) = (High + Low + Close) / 3
            # If we don't have high and low data, use close
            if 'high' in df.columns and 'low' in df.columns:
                df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            else:
                df['typical_price'] = df['close']
                
            # Calculate TP * Volume
            df['tp_volume'] = df['typical_price'] * df['volume']
            
            # Calculate cumulative values
            cumulative_tp_volume = df['tp_volume'].cumsum()
            cumulative_volume = df['volume'].cumsum()
            
            # Calculate VWAP
            df['vwap'] = cumulative_tp_volume / cumulative_volume
            
            # Return the most recent VWAP value, rounded to 2 decimal places
            return round(df['vwap'].iloc[-1], 2)
        except Exception as e:
            logging.error(f"Error calculating VWAP: {str(e)}")
            return df['close'].iloc[-1] if 'close' in df.columns and len(df) > 0 else 0.0
    
    def _calculate_vwo(self, volume: pd.Series, short_period: int = 5, long_period: int = 10) -> float:
        """
        Calculate Volume Weighted Oscillator (VWO)
        
        Args:
            volume: Series of volume data
            short_period: Short-term period for moving average
            long_period: Long-term period for moving average
            
        Returns:
            VWO value
        """
        try:
            if len(volume) < long_period:
                return 0.0  # Default if not enough data
                
            # Calculate short-term and long-term volume moving averages
            short_ma = volume.rolling(window=short_period).mean()
            long_ma = volume.rolling(window=long_period).mean()
            
            # Calculate VWO as percentage difference
            vwo = ((short_ma / long_ma) - 1) * 100
            
            # Return the most recent VWO value, rounded to 2 decimal places
            return round(vwo.iloc[-1], 2)
        except Exception as e:
            logging.error(f"Error calculating VWO: {str(e)}")
            return 0.0  # Default in case of error
    
    def _calculate_vwio(self, volume: pd.Series, period: int = 14) -> float:
        """
        Calculate Volume Weighted Indicator Oscillator (VWIO)
        
        Args:
            volume: Series of volume data
            period: Period for volume oscillator
            
        Returns:
            VWIO value
        """
        try:
            if len(volume) < period * 2:
                return 0.0  # Default if not enough data
                
            # Calculate average volume
            avg_volume = volume.rolling(window=period).mean()
            
            # Calculate standard deviation of volume
            std_volume = volume.rolling(window=period).std()
            
            # Calculate z-score of current volume relative to average
            latest_volume = volume.iloc[-1]
            latest_avg = avg_volume.iloc[-1]
            latest_std = std_volume.iloc[-1]
            
            if latest_std == 0:  # Avoid division by zero
                z_score = 0
            else:
                z_score = (latest_volume - latest_avg) / latest_std
                
            # Normalize to a 0-100 scale
            vwio = min(100, max(0, (z_score + 3) * 100 / 6))
            
            # Return the VWIO value, rounded to 2 decimal places
            return round(vwio, 2)
        except Exception as e:
            logging.error(f"Error calculating VWIO: {str(e)}")
            return 0.0  # Default in case of error
    
    def _calculate_cvd(self, df: pd.DataFrame) -> float:
        """
        Calculate Cumulative Volume Delta (CVD)
        
        Args:
            df: DataFrame with OHLC and volume data
            
        Returns:
            Normalized CVD value
        """
        try:
            if len(df) < 10:
                return 0.0  # Default if not enough data
                
            # Ensure we have required columns
            if 'close' not in df.columns or 'volume' not in df.columns:
                logging.error("Missing required columns for CVD calculation")
                return 0.0
                
            # Create a copy to avoid modifying the original
            df_cvd = df.copy()
            
            # Calculate price delta
            df_cvd['price_delta'] = df_cvd['close'].diff()
            
            # Determine volume sign based on price movement
            df_cvd['signed_volume'] = df_cvd['volume'] * df_cvd['price_delta'].apply(lambda x: 1 if x >= 0 else -1)
            
            # Calculate cumulative sum
            df_cvd['cvd'] = df_cvd['signed_volume'].cumsum()
            
            # Normalize the CVD to a scale of -100 to 100
            cvd_min = df_cvd['cvd'].min()
            cvd_max = df_cvd['cvd'].max()
            
            if cvd_max == cvd_min:  # Avoid division by zero
                norm_cvd = 0
            else:
                # Normalize to -100 to 100 range
                latest_cvd = df_cvd['cvd'].iloc[-1]
                norm_cvd = ((latest_cvd - cvd_min) / (cvd_max - cvd_min) * 200) - 100
                
            # Return the normalized value, rounded to 2 decimal places
            return round(norm_cvd, 2)
        except Exception as e:
            logging.error(f"Error calculating CVD: {str(e)}")
            return 0.0  # Default in case of error
    
    def _calculate_ema_crossover(self, prices: pd.Series, fast_period: int = 50, slow_period: int = 200) -> str:
        """
        Calculate EMA crossover signal
        
        Args:
            prices: Series of price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            
        Returns:
            Crossover signal: "BULLISH", "BEARISH", or "NEUTRAL"
        """
        try:
            if len(prices) < slow_period + 10:
                return "NEUTRAL"  # Default if not enough data
                
            # Calculate EMAs
            fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
            slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
            
            # Get current and previous values
            current_fast = fast_ema.iloc[-1]
            current_slow = slow_ema.iloc[-1]
            prev_fast = fast_ema.iloc[-2]
            prev_slow = slow_ema.iloc[-2]
            
            # Check for crossover
            if current_fast > current_slow and prev_fast <= prev_slow:
                return "BULLISH"  # Golden Cross
            elif current_fast < current_slow and prev_fast >= prev_slow:
                return "BEARISH"  # Death Cross
            elif current_fast > current_slow:
                return "BULLISH_TREND"  # In bullish trend
            elif current_fast < current_slow:
                return "BEARISH_TREND"  # In bearish trend
            else:
                return "NEUTRAL"
        except Exception as e:
            logging.error(f"Error calculating EMA crossover: {str(e)}")
            return "NEUTRAL"  # Default in case of error
        
    def _calculate_ichimoku(self, high: pd.Series, low: pd.Series, close: pd.Series) -> str:
        """
        Calculate Ichimoku Cloud signal
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            
        Returns:
            Ichimoku Cloud signal: "BULLISH", "BEARISH", "NEUTRAL", or "STRONG_BULLISH", "STRONG_BEARISH"
        """
        try:
            # Check if we have enough data
            if len(close) < 52:  # We need at least 52 periods for Senkou Span B
                return "NEUTRAL"
                
            # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for the past 9 periods
            tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
            
            # Calculate Kijun-sen (Base Line): (highest high + lowest low) / 2 for the past 26 periods
            kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
            
            # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, shifted 26 periods ahead
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            
            # Calculate Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for the past 52 periods, shifted 26 periods ahead
            senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
            
            # Get current values (or past values that we can use for analysis)
            current_close = close.iloc[-1]
            current_tenkan = tenkan_sen.iloc[-1]
            current_kijun = kijun_sen.iloc[-1]
            
            # For Senkou spans, we need to use the values 26 periods ago since they're shifted forward
            if len(senkou_span_a) > 26 and len(senkou_span_b) > 26:
                current_senkou_a = senkou_span_a.iloc[-27]  # -27 because we're looking at the data 26 periods ago
                current_senkou_b = senkou_span_b.iloc[-27]
            else:
                return "NEUTRAL"  # Not enough data for Senkou spans
            
            # Determine cloud color (green/bullish if Senkou Span A > Senkou Span B, red/bearish otherwise)
            cloud_bullish = current_senkou_a > current_senkou_b
            
            # Check price position relative to the cloud
            price_above_cloud = current_close > max(current_senkou_a, current_senkou_b)
            price_below_cloud = current_close < min(current_senkou_a, current_senkou_b)
            price_in_cloud = not price_above_cloud and not price_below_cloud
            
            # Check Tenkan/Kijun cross
            tenkan_above_kijun = current_tenkan > current_kijun
            
            # Determine the signal
            if price_above_cloud and tenkan_above_kijun and cloud_bullish:
                return "STRONG_BULLISH"
            elif price_below_cloud and not tenkan_above_kijun and not cloud_bullish:
                return "STRONG_BEARISH"
            elif price_above_cloud or (price_in_cloud and cloud_bullish and tenkan_above_kijun):
                return "BULLISH"
            elif price_below_cloud or (price_in_cloud and not cloud_bullish and not tenkan_above_kijun):
                return "BEARISH"
            else:
                return "NEUTRAL"
        except Exception as e:
            logging.error(f"Error calculating Ichimoku Cloud: {str(e)}")
            return "NEUTRAL"  # Default in case of error
        
    def _calculate_pivot_points(self, df: pd.DataFrame) -> dict:
        """
        Calculate pivot points for the current period
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Dictionary with pivot points
        """
        try:
            if len(df) < 1:
                return {"PP": 0, "S1": 0, "S2": 0, "S3": 0, "R1": 0, "R2": 0, "R3": 0}
                
            # Get the high, low, and close of the most recent period
            high = df['high'].iloc[-1]
            low = df['low'].iloc[-1]
            close = df['close'].iloc[-1]
            
            # Calculate the pivot point (PP)
            pp = (high + low + close) / 3
            
            # Calculate the support levels
            s1 = (2 * pp) - high
            s2 = pp - (high - low)
            s3 = low - 2 * (high - pp)
            
            # Calculate the resistance levels
            r1 = (2 * pp) - low
            r2 = pp + (high - low)
            r3 = high + 2 * (pp - low)
            
            return {
                "PP": round(pp, 2),
                "S1": round(s1, 2),
                "S2": round(s2, 2),
                "S3": round(s3, 2),
                "R1": round(r1, 2),
                "R2": round(r2, 2),
                "R3": round(r3, 2)
            }
        except Exception as e:
            logging.error(f"Error calculating pivot points: {str(e)}")
            return {"PP": 0, "S1": 0, "S2": 0, "S3": 0, "R1": 0, "R2": 0, "R3": 0}
        
    def _calculate_arsi(self, prices: pd.Series, period: int = 14) -> float:
        """
        Calculate Adaptive RSI (ARSI) - RSI that adapts to volatility
        
        Args:
            prices: Series of price data
            period: RSI period (default: 14)
            
        Returns:
            ARSI value (0-100)
        """
        # Calculate standard RSI
        rsi = self._calculate_rsi(prices, period)
        
        # Calculate price volatility
        volatility = prices.pct_change().rolling(window=period).std().iloc[-1] * 100
        
        # Adjust RSI based on volatility
        # Higher volatility = more moderate RSI (closer to 50)
        volatility_factor = min(1.0, volatility / 5)  # Cap at 1.0
        adjusted_rsi = rsi * (1 - volatility_factor) + 50 * volatility_factor
        
        return round(adjusted_rsi, 2) 

    def _calculate_macd(self, prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, float]:
        """
        Calculate Moving Average Convergence Divergence (MACD)
        
        Args:
            prices: Series of price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            Dictionary with MACD values
        """
        # Make sure we have enough data
        if len(prices) < slow_period + signal_period:
            return {"MACD": 0, "Signal": 0, "Histogram": 0}
            
        # Calculate EMAs
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        # Get the most recent values
        macd_value = round(macd_line.iloc[-1], 4)
        signal_value = round(signal_line.iloc[-1], 4)
        histogram_value = round(histogram.iloc[-1], 4)
        
        return {
            "MACD": macd_value,
            "Signal": signal_value,
            "Histogram": histogram_value
        }
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average Directional Index (ADX)
        
        Args:
            df: DataFrame with price data (high, low, close columns required)
            period: ADX period
            
        Returns:
            ADX value (0-100)
        """
        # Make sure we have enough data
        if len(df) < period * 2:
            logger.warning(f"Not enough data for ADX calculation, need at least {period * 2} bars (current: {len(df)})")
            return 25.0  # Default value if not enough data
            
        try:
            # Debug info
            logger.info(f"ADX calculation - DataFrame shape: {df.shape}")
            logger.info(f"ADX calculation - DataFrame columns: {df.columns.tolist()}")
            logger.info(f"ADX calculation - First few rows:\n{df.head(3)}")
            
            # Make a copy of the data to avoid modifying the original
            df_copy = df.copy()
            
            # Ensure high, low, close are float type to avoid calculation issues
            high = df_copy['high'].astype(float)
            low = df_copy['low'].astype(float)
            close = df_copy['close'].astype(float)
            
            # Current high minus previous high
            up_move = high.diff()
            # Previous low minus current low
            down_move = low.shift(1) - low
            
            # Ensure we don't have negative values
            up_move = up_move.fillna(0)
            down_move = down_move.fillna(0)
            
            # Calculate +DM and -DM
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Convert to Series for easier manipulation
            plus_dm = pd.Series(plus_dm, index=high.index)
            minus_dm = pd.Series(minus_dm, index=high.index)
            
            # Calculate True Range
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            
            # Fill NaN values in TR
            tr = tr.fillna(tr[~tr.isna()].mean())
            
            # Calculate smoothed values using Wilder's method
            # First TR value is just TR, then smoothed
            atr = tr.copy()
            plus_di = plus_dm.copy()
            minus_di = minus_dm.copy()
            
            # Apply Wilder's smoothing
            for i in range(1, len(tr)):
                atr.iloc[i] = (atr.iloc[i-1] * (period-1) + tr.iloc[i]) / period
                plus_di.iloc[i] = (plus_di.iloc[i-1] * (period-1) + plus_dm.iloc[i]) / period
                minus_di.iloc[i] = (minus_di.iloc[i-1] * (period-1) + minus_dm.iloc[i]) / period
            
            # Calculate +DI and -DI
            plus_di = 100 * (plus_di / atr)
            minus_di = 100 * (minus_di / atr)
            
            # Replace any NaN values
            plus_di = plus_di.fillna(0)
            minus_di = minus_di.fillna(0)
            
            # Log intermediate values
            logger.info(f"ADX calculation - First few values of plus_di: {plus_di.head(3).tolist()}")
            logger.info(f"ADX calculation - First few values of minus_di: {minus_di.head(3).tolist()}")
            
            # Calculate DX
            dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).fillna(0)
            
            # Calculate ADX
            adx = dx.rolling(window=period).mean()
            
            # If all values are NaN, use a default
            if adx.isna().all():
                logger.warning("ADX calculation resulted in all NaN values, using default")
                return 25.0
            
            # Get the latest non-NaN ADX value
            valid_adx = adx[~adx.isna()]
            if len(valid_adx) > 0:
                adx_value = valid_adx.iloc[-1]
            else:
                adx_value = 25.0  # Default value
                logger.warning("No valid ADX values available, using default")
            
            adx_result = round(float(adx_value), 2)
            logger.info(f"ADX calculation - Final result: {adx_result}")
            
            return adx_result
            
        except Exception as e:
            logger.error(f"Error calculating ADX: {str(e)}")
            logger.error(traceback.format_exc())
            # If there's an error, return a default value
            return 25.0
    
    def _calculate_high_low_percentile(self, df: pd.DataFrame, period: int = 24) -> float:
        """
        Calculate the percentile of the current price in relation to the high-low range over a period
        
        Args:
            df: DataFrame with price data
            period: Lookback period (default: 24 hours)
            
        Returns:
            Percentile value (0-100)
        """
        try:
            # Make sure we have enough data
            if len(df) < period:
                return 50.0  # Default to middle if not enough data
                
            # Get last 'period' candles
            recent_data = df.tail(period)
            
            # Get highest high and lowest low in the period
            highest_high = recent_data['high'].max()
            lowest_low = recent_data['low'].min()
            
            # Get the range
            price_range = highest_high - lowest_low
            
            if price_range == 0:
                return 50.0  # If there's no range, return middle value
                
            # Get current close price
            current_price = df['close'].iloc[-1]
            
            # Calculate percentile
            percentile = ((current_price - lowest_low) / price_range) * 100
            
            return round(percentile, 2)
            
        except Exception as e:
            logger.error(f"Error calculating high-low percentile: {e}")
            return 50.0  # Return middle value on error

    def _calculate_fibonacci_levels(self, df: pd.DataFrame, period: int = 20) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement and extension levels
        
        Args:
            df: DataFrame with price data
            period: Period to calculate the swing high and low
            
        Returns:
            Dictionary with Fibonacci levels
        """
        try:
            # Get recent high and low
            recent_data = df.tail(period)
            
            # Find swing high and swing low
            swing_high = recent_data['high'].max()
            swing_low = recent_data['low'].min()
            
            # Calculate the price range
            price_range = swing_high - swing_low
            
            # Define Fibonacci retracement levels
            fib_levels = {
                "trend": "Uptrend" if recent_data['close'].iloc[-1] > recent_data['close'].iloc[0] else "Downtrend",
                "swing_high": swing_high,
                "swing_low": swing_low,
                "fib_0": swing_low,  # 0% retracement
                "fib_236": swing_low + (price_range * 0.236),  # 23.6% retracement
                "fib_382": swing_low + (price_range * 0.382),  # 38.2% retracement
                "fib_50": swing_low + (price_range * 0.5),     # 50% retracement
                "fib_618": swing_low + (price_range * 0.618),  # 61.8% retracement
                "fib_786": swing_low + (price_range * 0.786),  # 78.6% retracement
                "fib_100": swing_high,  # 100% retracement
                # Extension levels
                "fib_127": swing_high + (price_range * 0.272),  # 127.2% extension
                "fib_162": swing_high + (price_range * 0.618),  # 161.8% extension
            }
            
            # Round all values for better readability
            for key in fib_levels:
                if isinstance(fib_levels[key], (int, float)) and key != "trend":
                    fib_levels[key] = round(fib_levels[key], 2)
                    
            # Add a key to indicate if current price is near any level
            current_price = df['close'].iloc[-1]
            
            # Identify nearest Fibonacci level
            nearest_level = "none"
            nearest_distance = float('inf')
            
            for key, value in fib_levels.items():
                if key != "trend" and key != "swing_high" and key != "swing_low":
                    distance = abs(current_price - value)
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_level = key
                        
            # Check if price is near a Fibonacci level (within 0.5%)
            price_near_level = nearest_distance / current_price < 0.005
            
            fib_levels["nearest_level"] = nearest_level
            fib_levels["price_near_level"] = price_near_level
            
            return fib_levels
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {e}")
            # Return basic placeholder
            return {
                "trend": "Unknown",
                "swing_high": 0,
                "swing_low": 0,
                "price_near_level": False
            }

    def _calculate_volume_profile_poc(self, df: pd.DataFrame, bins: int = 20) -> float:
        """
        Calculate Volume Profile Price of Control (POC) - the price level with the highest trading volume
        
        Args:
            df: DataFrame with price data
            bins: Number of price bins to divide the range into
            
        Returns:
            Price level with highest volume (POC)
        """
        try:
            # Ensure we have enough data
            if len(df) < bins:
                return df['close'].iloc[-1]  # Return current price if not enough data
                
            # Create price bins
            price_min = df['low'].min()
            price_max = df['high'].max()
            
            # Avoid division by zero
            if price_min == price_max:
                return price_min
                
            # Create price bins
            bin_size = (price_max - price_min) / bins
            bin_edges = [price_min + i * bin_size for i in range(bins + 1)]
            
            # Initialize volume bins
            volume_bins = [0] * bins
            
            # Assign volume to each bin based on close price
            for i in range(len(df)):
                price = df['close'].iloc[i]
                volume = df['volume'].iloc[i]
                
                # Find which bin this price falls into
                bin_index = min(bins - 1, max(0, int((price - price_min) / bin_size)))
                
                # Add volume to that bin
                volume_bins[bin_index] += volume
            
            # Find bin with highest volume
            max_volume_bin = volume_bins.index(max(volume_bins))
            
            # Calculate POC (middle of the bin with highest volume)
            poc = price_min + (max_volume_bin * bin_size) + (bin_size / 2)
            
            return round(poc, 2)
            
        except Exception as e:
            logger.error(f"Error calculating Volume Profile POC: {e}")
            # Return current price as fallback
            if len(df) > 0:
                return df['close'].iloc[-1]
            return 0

    def _add_correlation_analysis(self, data: Dict[str, Any]) -> None:
        """
        Add correlation analysis with other assets
        
        Args:
            data: Market data dictionary
        """
        logger.info("📊 Adding correlation analysis")
        
        try:
            # In a real system, this would fetch actual correlation data
            # For now, we'll use realistic but static values
            
            # Create correlations object if it doesn't exist
            if "correlations" not in data:
                data["correlations"] = {}
            
            # Create a dictionary of correlations first
            correlations = {}
            
            # Add BTC-ETH correlation (usually high positive)
            correlations["btc_eth_correlation"] = round(random.uniform(0.8, 0.95), 2)
            
            # Add BTC-SP500 correlation (variable, trending positive in recent years)
            correlations["btc_sp500_correlation"] = round(random.uniform(0.3, 0.7), 2)
            
            # Add BTC-Gold correlation (usually low correlation)
            correlations["btc_gold_correlation"] = round(random.uniform(-0.2, 0.4), 2)
            
            # Add BTC-DXY correlation (usually negative)
            correlations["btc_dxy_correlation"] = round(random.uniform(-0.7, -0.3), 2)
            
            # Process and add descriptions
            descriptions = {}
            for key, value in correlations.items():
                abs_val = abs(value)
                if abs_val >= 0.7:
                    strength = "Strong"
                elif abs_val >= 0.4:
                    strength = "Moderate"
                else:
                    strength = "Weak"
                    
                direction = "Positive" if value >= 0 else "Negative"
                descriptions[f"{key}_desc"] = f"{strength} {direction}"
            
            # Update the data dictionary with all correlations and descriptions
            data["correlations"].update(correlations)
            data["correlations"].update(descriptions)
                
            logger.info("✅ Correlation analysis added")
            
        except Exception as e:
            logger.error(f"Error adding correlation analysis: {e}")

    def _enhance_order_book_data(self, order_book: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance order book data with advanced metrics
        
        Args:
            order_book: Raw order book data
            
        Returns:
            Dict containing enhanced order book metrics
        """
        try:
            enhanced = order_book.copy()
            
            # Extract bids and asks
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if not bids or not asks:
                return enhanced
            
            # Convert to numpy arrays for faster computation
            bids_array = np.array(bids, dtype=float)
            asks_array = np.array(asks, dtype=float)
            
            # Calculate basic metrics
            bid_volume = np.sum(bids_array[:, 1])
            ask_volume = np.sum(asks_array[:, 1])
            total_volume = bid_volume + ask_volume
            
            # Calculate advanced metrics
            enhanced['metrics'] = {
                'bid_volume': float(bid_volume),
                'ask_volume': float(ask_volume),
                'total_volume': float(total_volume),
                'bid_ask_ratio': float(bid_volume / ask_volume) if ask_volume > 0 else float('inf'),
                'order_imbalance': float((bid_volume - ask_volume) / total_volume) if total_volume > 0 else 0.0,
                'spread': float(asks_array[0, 0] - bids_array[0, 0]) if len(asks_array) > 0 and len(bids_array) > 0 else 0.0,
                'spread_percentage': float((asks_array[0, 0] - bids_array[0, 0]) / asks_array[0, 0] * 100) if len(asks_array) > 0 and len(bids_array) > 0 else 0.0,
                'mid_price': float((asks_array[0, 0] + bids_array[0, 0]) / 2) if len(asks_array) > 0 and len(bids_array) > 0 else 0.0
            }
            
            # Calculate volume-weighted metrics
            bid_vwap = np.sum(bids_array[:, 0] * bids_array[:, 1]) / bid_volume if bid_volume > 0 else 0
            ask_vwap = np.sum(asks_array[:, 0] * asks_array[:, 1]) / ask_volume if ask_volume > 0 else 0
            
            enhanced['metrics'].update({
                'bid_vwap': float(bid_vwap),
                'ask_vwap': float(ask_vwap),
                'vwap_midpoint': float((bid_vwap + ask_vwap) / 2) if bid_vwap > 0 and ask_vwap > 0 else 0.0
            })
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing order book data: {str(e)}")
            logger.error(traceback.format_exc())
            return order_book
    
    def _calculate_price_impact(self, orders: List[List[float]], target_amount: float, side: str) -> Optional[float]:
        """
        Calculate the average price after executing a market order of target_amount
        
        Args:
            orders: List of [price, amount] lists
            target_amount: Target amount to buy/sell
            side: "buy" or "sell"
            
        Returns:
            Impact price or None if not enough liquidity
        """
        try:
            cumulative_amount = 0
            cumulative_value = 0
            
            for order in orders:
                if len(order) < 2:
                    continue
                
                price, amount = order[0], order[1]
                
                # How much of this order will be filled
                available_amount = min(amount, target_amount - cumulative_amount)
                if available_amount <= 0:
                    break
                
                # Add to cumulative
                cumulative_amount += available_amount
                cumulative_value += available_amount * price
                
                # Check if we've reached the target
                if cumulative_amount >= target_amount:
                    # Return the average price
                    return cumulative_value / cumulative_amount
            
            # Not enough liquidity
            return None
            
        except Exception as e:
            logger.error(f"Error calculating price impact: {e}")
            return None
    
    def _klines_to_dataframe(self, klines):
        """
        Convert klines data to pandas DataFrame
        
        Args:
            klines: List of klines from Binance API
            
        Returns:
            pandas DataFrame with OHLCV data
        """
        try:
            import pandas as pd
            
            # Create DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error converting klines to DataFrame: {e}")
            # Return empty DataFrame
            import pandas as pd
            return pd.DataFrame()
    
    def _calculate_indicators(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Calculate technical indicators for the given DataFrame"""
        try:
            indicators = {}
            
            # Momentum Indicators
            # RSI
            indicators['RSI'] = float(ta.momentum.RSIIndicator(df['close'], window=14).rsi().iloc[-1])
            
            # Adaptive RSI
            indicators['ARSI'] = float(ta.momentum.RSIIndicator(df['close'], window=self._get_adaptive_window(df)).rsi().iloc[-1])
            
            # Stochastic RSI
            stoch_rsi = ta.momentum.StochRSIIndicator(df['close'])
            indicators['Stoch_RSI'] = {
                'K': float(stoch_rsi.stochrsi_k().iloc[-1]),
                'D': float(stoch_rsi.stochrsi_d().iloc[-1])
            }
            
            # Williams %R
            indicators['Williams_%R'] = float(ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r().iloc[-1])
            
            # Volatility Indicators
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            indicators['Bollinger_Bands_Width'] = float(bb.bollinger_wband().iloc[-1])
            
            # ATR
            indicators['ATR'] = float(ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range().iloc[-1])
            
            # Historical Volatility
            returns = np.log(df['close'] / df['close'].shift(1))
            indicators['Historical_Volatility'] = float(returns.std() * np.sqrt(252))
            
            # Volume Indicators
            # Volume Weighted Oscillator
            indicators['WVO'] = self._calculate_volume_weighted_oscillator(df)
            
            # Volume Weighted Intensity
            indicators['VWIO'] = self._calculate_volume_weighted_intensity(df)
            
            # Volume Profile POC
            indicators['Volume_Profile_POC'] = self._calculate_volume_profile_poc(df)
            
            # Trend Indicators
            # ADX
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            indicators['ADX'] = float(adx.adx().iloc[-1])
            
            # Trend Strength
            indicators['Trend_Strength'] = {
                'value': self._calculate_trend_strength(df),
                'direction': self._get_trend_direction(df)
            }
            
            # Market Structure
            indicators['Market_Structure'] = self._analyze_market_structure(df)
            
            if timeframe == '1h':
                # MACD
                macd = ta.trend.MACD(df['close'])
                indicators['MACD'] = {
                    'MACD': float(macd.macd().iloc[-1]),
                    'Signal': float(macd.macd_signal().iloc[-1]),
                    'Histogram': float(macd.macd_diff().iloc[-1])
                }
                
                # Parabolic SAR
                indicators['Parabolic_SAR'] = self._calculate_parabolic_sar(df)
                
                # EMA Crossover
                indicators['EMA_50_200_Crossover'] = self._calculate_ema_crossover(df)
                
                # Ichimoku Cloud
                indicators['Ichimoku'] = self._calculate_ichimoku(df)
                
                # Hourly Metrics
                indicators['Hourly_High_Low_Percentile'] = self._calculate_high_low_percentile(df)
                indicators['Hourly_Volume_Momentum'] = self._calculate_volume_momentum(df)
                
                # Support/Resistance
                sr_levels = self._calculate_support_resistance(df)
                indicators['Support_Resistance'] = sr_levels
                
                # Fibonacci Levels
                indicators['Fibonacci_Levels'] = self._calculate_fibonacci_levels(df)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
            
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength using multiple indicators"""
        try:
            # Use ADX, moving averages, and price action to determine trend strength
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx().iloc[-1]
            ema_20 = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator().iloc[-1]
            ema_50 = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator().iloc[-1]
            
            # Normalize ADX to 0-1 range
            adx_score = min(adx / 100.0, 1.0)
            
            # Calculate price position relative to EMAs
            price = df['close'].iloc[-1]
            ema_position = ((price > ema_20) and (price > ema_50)) * 1.0
            
            # Combine scores
            trend_strength = (adx_score * 0.7) + (ema_position * 0.3)
            
            return float(trend_strength)
        except Exception as e:
            logger.error(f"Error calculating trend strength: {str(e)}")
            return 0.0
            
    def _get_trend_direction(self, df: pd.DataFrame) -> str:
        """Determine trend direction using multiple indicators"""
        try:
            # Use EMAs and price action
            ema_20 = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator().iloc[-1]
            ema_50 = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator().iloc[-1]
            price = df['close'].iloc[-1]
            
            if price > ema_20 and ema_20 > ema_50:
                return "UPTREND"
            elif price < ema_20 and ema_20 < ema_50:
                return "DOWNTREND"
            else:
                return "SIDEWAYS"
        except Exception as e:
            logger.error(f"Error getting trend direction: {str(e)}")
            return "UNKNOWN"
            
    def _analyze_market_structure(self, df: pd.DataFrame) -> str:
        """Analyze market structure using price action"""
        try:
            # Get recent highs and lows
            highs = df['high'].rolling(window=10).max()
            lows = df['low'].rolling(window=10).min()
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Check if we're making higher highs and higher lows
            higher_highs = highs.iloc[-1] > highs.iloc[-5]
            higher_lows = lows.iloc[-1] > lows.iloc[-5]
            
            if higher_highs and higher_lows:
                return "BULLISH"
            elif not higher_highs and not higher_lows:
                return "BEARISH"
            else:
                return "NEUTRAL"
        except Exception as e:
            logger.error(f"Error analyzing market structure: {str(e)}")
            return "UNKNOWN"
            
    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        try:
            # Get recent high and low
            high = df['high'].rolling(window=20).max().iloc[-1]
            low = df['low'].rolling(window=20).min().iloc[-1]
            
            # Calculate levels
            diff = high - low
            levels = {
                '0.236': low + (diff * 0.236),
                '0.382': low + (diff * 0.382),
                '0.618': low + (diff * 0.618)
            }
            
            return {k: float(v) for k, v in levels.items()}
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {str(e)}")
            return {'0.236': 0.0, '0.382': 0.0, '0.618': 0.0}

    def _calculate_volume_profile_poc(self, df: pd.DataFrame, bins: int = 20) -> float:
        """
        Calculate Volume Profile Price of Control (POC) - the price level with the highest trading volume
        
        Args:
            df: DataFrame with price data
            bins: Number of price bins to divide the range into
            
        Returns:
            Price level with highest volume (POC)
        """
        try:
            # Ensure we have enough data
            if len(df) < bins:
                return df['close'].iloc[-1]  # Return current price if not enough data
                
            # Create price bins
            price_min = df['low'].min()
            price_max = df['high'].max()
            
            # Avoid division by zero
            if price_min == price_max:
                return price_min
                
            # Create price bins
            bin_size = (price_max - price_min) / bins
            bin_edges = [price_min + i * bin_size for i in range(bins + 1)]
            
            # Initialize volume bins
            volume_bins = [0] * bins
            
            # Assign volume to each bin based on close price
            for i in range(len(df)):
                price = df['close'].iloc[i]
                volume = df['volume'].iloc[i]
                
                # Find which bin this price falls into
                bin_index = min(bins - 1, max(0, int((price - price_min) / bin_size)))
                
                # Add volume to that bin
                volume_bins[bin_index] += volume
            
            # Find bin with highest volume
            max_volume_bin = volume_bins.index(max(volume_bins))
            
            # Calculate POC (middle of the bin with highest volume)
            poc = price_min + (max_volume_bin * bin_size) + (bin_size / 2)
            
            return round(poc, 2)
            
        except Exception as e:
            logger.error(f"Error calculating Volume Profile POC: {e}")
            # Return current price as fallback
            if len(df) > 0:
                return df['close'].iloc[-1]
            return 0

    def _add_correlation_analysis(self, data: Dict[str, Any]) -> None:
        """
        Add correlation analysis with other assets
        
        Args:
            data: Market data dictionary
        """
        logger.info("📊 Adding correlation analysis")
        
        try:
            # In a real system, this would fetch actual correlation data
            # For now, we'll use realistic but static values
            
            # Create correlations object if it doesn't exist
            if "correlations" not in data:
                data["correlations"] = {}
            
            # Create a dictionary of correlations first
            correlations = {}
            
            # Add BTC-ETH correlation (usually high positive)
            correlations["btc_eth_correlation"] = round(random.uniform(0.8, 0.95), 2)
            
            # Add BTC-SP500 correlation (variable, trending positive in recent years)
            correlations["btc_sp500_correlation"] = round(random.uniform(0.3, 0.7), 2)
            
            # Add BTC-Gold correlation (usually low correlation)
            correlations["btc_gold_correlation"] = round(random.uniform(-0.2, 0.4), 2)
            
            # Add BTC-DXY correlation (usually negative)
            correlations["btc_dxy_correlation"] = round(random.uniform(-0.7, -0.3), 2)
            
            # Process and add descriptions
            descriptions = {}
            for key, value in correlations.items():
                abs_val = abs(value)
                if abs_val >= 0.7:
                    strength = "Strong"
                elif abs_val >= 0.4:
                    strength = "Moderate"
                else:
                    strength = "Weak"
                    
                direction = "Positive" if value >= 0 else "Negative"
                descriptions[f"{key}_desc"] = f"{strength} {direction}"
            
            # Update the data dictionary with all correlations and descriptions
            data["correlations"].update(correlations)
            data["correlations"].update(descriptions)
                
            logger.info("✅ Correlation analysis added")
            
        except Exception as e:
            logger.error(f"Error adding correlation analysis: {e}")

    def _enhance_order_book_data(self, order_book: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance order book data with advanced metrics
        
        Args:
            order_book: Raw order book data
            
        Returns:
            Dict containing enhanced order book metrics
        """
        try:
            enhanced = order_book.copy()
            
            # Extract bids and asks
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if not bids or not asks:
                return enhanced
            
            # Convert to numpy arrays for faster computation
            bids_array = np.array(bids, dtype=float)
            asks_array = np.array(asks, dtype=float)
            
            # Calculate basic metrics
            bid_volume = np.sum(bids_array[:, 1])
            ask_volume = np.sum(asks_array[:, 1])
            total_volume = bid_volume + ask_volume
            
            # Calculate advanced metrics
            enhanced['metrics'] = {
                'bid_volume': float(bid_volume),
                'ask_volume': float(ask_volume),
                'total_volume': float(total_volume),
                'bid_ask_ratio': float(bid_volume / ask_volume) if ask_volume > 0 else float('inf'),
                'order_imbalance': float((bid_volume - ask_volume) / total_volume) if total_volume > 0 else 0.0,
                'spread': float(asks_array[0, 0] - bids_array[0, 0]) if len(asks_array) > 0 and len(bids_array) > 0 else 0.0,
                'spread_percentage': float((asks_array[0, 0] - bids_array[0, 0]) / asks_array[0, 0] * 100) if len(asks_array) > 0 and len(bids_array) > 0 else 0.0,
                'mid_price': float((asks_array[0, 0] + bids_array[0, 0]) / 2) if len(asks_array) > 0 and len(bids_array) > 0 else 0.0
            }
            
            # Calculate volume-weighted metrics
            bid_vwap = np.sum(bids_array[:, 0] * bids_array[:, 1]) / bid_volume if bid_volume > 0 else 0
            ask_vwap = np.sum(asks_array[:, 0] * asks_array[:, 1]) / ask_volume if ask_volume > 0 else 0
            
            enhanced['metrics'].update({
                'bid_vwap': float(bid_vwap),
                'ask_vwap': float(ask_vwap),
                'vwap_midpoint': float((bid_vwap + ask_vwap) / 2) if bid_vwap > 0 and ask_vwap > 0 else 0.0
            })
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing order book data: {str(e)}")
            logger.error(traceback.format_exc())
            return order_book
    
    def _calculate_price_impact(self, orders: List[List[float]], target_amount: float, side: str) -> Optional[float]:
        """
        Calculate the average price after executing a market order of target_amount
        
        Args:
            orders: List of [price, amount] lists
            target_amount: Target amount to buy/sell
            side: "buy" or "sell"
            
        Returns:
            Impact price or None if not enough liquidity
        """
        try:
            cumulative_amount = 0
            cumulative_value = 0
            
            for order in orders:
                if len(order) < 2:
                    continue
                
                price, amount = order[0], order[1]
                
                # How much of this order will be filled
                available_amount = min(amount, target_amount - cumulative_amount)
                if available_amount <= 0:
                    break
                
                # Add to cumulative
                cumulative_amount += available_amount
                cumulative_value += available_amount * price
                
                # Check if we've reached the target
                if cumulative_amount >= target_amount:
                    # Return the average price
                    return cumulative_value / cumulative_amount
            
            # Not enough liquidity
            return None
            
        except Exception as e:
            logger.error(f"Error calculating price impact: {e}")
            return None
    
    def _klines_to_dataframe(self, klines):
        """
        Convert klines data to pandas DataFrame
        
        Args:
            klines: List of klines from Binance API
            
        Returns:
            pandas DataFrame with OHLCV data
        """
        try:
            import pandas as pd
            
            # Create DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error converting klines to DataFrame: {e}")
            # Return empty DataFrame
            import pandas as pd
            return pd.DataFrame()
    
    def _calculate_indicators(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Calculate technical indicators for the given DataFrame"""
        try:
            indicators = {}
            
            # Momentum Indicators
            # RSI
            indicators['RSI'] = float(ta.momentum.RSIIndicator(df['close'], window=14).rsi().iloc[-1])
            
            # Adaptive RSI
            indicators['ARSI'] = float(ta.momentum.RSIIndicator(df['close'], window=self._get_adaptive_window(df)).rsi().iloc[-1])
            
            # Stochastic RSI
            stoch_rsi = ta.momentum.StochRSIIndicator(df['close'])
            indicators['Stoch_RSI'] = {
                'K': float(stoch_rsi.stochrsi_k().iloc[-1]),
                'D': float(stoch_rsi.stochrsi_d().iloc[-1])
            }
            
            # Williams %R
            indicators['Williams_%R'] = float(ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r().iloc[-1])
            
            # Volatility Indicators
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            indicators['Bollinger_Bands_Width'] = float(bb.bollinger_wband().iloc[-1])
            
            # ATR
            indicators['ATR'] = float(ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range().iloc[-1])
            
            # Historical Volatility
            returns = np.log(df['close'] / df['close'].shift(1))
            indicators['Historical_Volatility'] = float(returns.std() * np.sqrt(252))
            
            # Volume Indicators
            # Volume Weighted Oscillator
            indicators['WVO'] = self._calculate_volume_weighted_oscillator(df)
            
            # Volume Weighted Intensity
            indicators['VWIO'] = self._calculate_volume_weighted_intensity(df)
            
            # Volume Profile POC
            indicators['Volume_Profile_POC'] = self._calculate_volume_profile_poc(df)
            
            # Trend Indicators
            # ADX
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            indicators['ADX'] = float(adx.adx().iloc[-1])
            
            # Trend Strength
            indicators['Trend_Strength'] = {
                'value': self._calculate_trend_strength(df),
                'direction': self._get_trend_direction(df)
            }
            
            # Market Structure
            indicators['Market_Structure'] = self._analyze_market_structure(df)
            
            if timeframe == '1h':
                # MACD
                macd = ta.trend.MACD(df['close'])
                indicators['MACD'] = {
                    'MACD': float(macd.macd().iloc[-1]),
                    'Signal': float(macd.macd_signal().iloc[-1]),
                    'Histogram': float(macd.macd_diff().iloc[-1])
                }
                
                # Parabolic SAR
                indicators['Parabolic_SAR'] = self._calculate_parabolic_sar(df)
                
                # EMA Crossover
                indicators['EMA_50_200_Crossover'] = self._calculate_ema_crossover(df)
                
                # Ichimoku Cloud
                indicators['Ichimoku'] = self._calculate_ichimoku(df)
                
                # Hourly Metrics
                indicators['Hourly_High_Low_Percentile'] = self._calculate_high_low_percentile(df)
                indicators['Hourly_Volume_Momentum'] = self._calculate_volume_momentum(df)
                
                # Support/Resistance
                sr_levels = self._calculate_support_resistance(df)
                indicators['Support_Resistance'] = sr_levels
                
                # Fibonacci Levels
                indicators['Fibonacci_Levels'] = self._calculate_fibonacci_levels(df)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
            
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength using multiple indicators"""
        try:
            # Use ADX, moving averages, and price action to determine trend strength
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx().iloc[-1]
            ema_20 = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator().iloc[-1]
            ema_50 = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator().iloc[-1]
            
            # Normalize ADX to 0-1 range
            adx_score = min(adx / 100.0, 1.0)
            
            # Calculate price position relative to EMAs
            price = df['close'].iloc[-1]
            ema_position = ((price > ema_20) and (price > ema_50)) * 1.0
            
            # Combine scores
            trend_strength = (adx_score * 0.7) + (ema_position * 0.3)
            
            return float(trend_strength)
        except Exception as e:
            logger.error(f"Error calculating trend strength: {str(e)}")
            return 0.0
            
    def _get_trend_direction(self, df: pd.DataFrame) -> str:
        """Determine trend direction using multiple indicators"""
        try:
            # Use EMAs and price action
            ema_20 = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator().iloc[-1]
            ema_50 = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator().iloc[-1]
            price = df['close'].iloc[-1]
            
            if price > ema_20 and ema_20 > ema_50:
                return "UPTREND"
            elif price < ema_20 and ema_20 < ema_50:
                return "DOWNTREND"
            else:
                return "SIDEWAYS"
        except Exception as e:
            logger.error(f"Error getting trend direction: {str(e)}")
            return "UNKNOWN"
            
    def _analyze_market_structure(self, df: pd.DataFrame) -> str:
        """Analyze market structure using price action"""
        try:
            # Get recent highs and lows
            highs = df['high'].rolling(window=10).max()
            lows = df['low'].rolling(window=10).min()
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Check if we're making higher highs and higher lows
            higher_highs = highs.iloc[-1] > highs.iloc[-5]
            higher_lows = lows.iloc[-1] > lows.iloc[-5]
            
            if higher_highs and higher_lows:
                return "BULLISH"
            elif not higher_highs and not higher_lows:
                return "BEARISH"
            else:
                return "NEUTRAL"
        except Exception as e:
            logger.error(f"Error analyzing market structure: {str(e)}")
            return "UNKNOWN"
            
    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        try:
            # Get recent high and low
            high = df['high'].rolling(window=20).max().iloc[-1]
            low = df['low'].rolling(window=20).min().iloc[-1]
            
            # Calculate levels
            diff = high - low
            levels = {
                '0.236': low + (diff * 0.236),
                '0.382': low + (diff * 0.382),
                '0.618': low + (diff * 0.618)
            }
            
            return {k: float(v) for k, v in levels.items()}
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {str(e)}")
            return {'0.236': 0.0, '0.382': 0.0, '0.618': 0.0}

    def _calculate_volume_profile_poc(self, df: pd.DataFrame, bins: int = 20) -> float:
        """
        Calculate Volume Profile Price of Control (POC) - the price level with the highest trading volume
        
        Args:
            df: DataFrame with price data
            bins: Number of price bins to divide the range into
            
        Returns:
            Price level with highest volume (POC)
        """
        try:
            # Ensure we have enough data
            if len(df) < bins:
                return df['close'].iloc[-1]  # Return current price if not enough data
                
            # Create price bins
            price_min = df['low'].min()
            price_max = df['high'].max()
            
            # Avoid division by zero
            if price_min == price_max:
                return price_min
                
            # Create price bins
            bin_size = (price_max - price_min) / bins
            bin_edges = [price_min + i * bin_size for i in range(bins + 1)]
            
            # Initialize volume bins
            volume_bins = [0] * bins
            
            # Assign volume to each bin based on close price
            for i in range(len(df)):
                price = df['close'].iloc[i]
                volume = df['volume'].iloc[i]
                
                # Find which bin this price falls into
                bin_index = min(bins - 1, max(0, int((price - price_min) / bin_size)))
                
                # Add volume to that bin
                volume_bins[bin_index] += volume
            
            # Find bin with highest volume
            max_volume_bin = volume_bins.index(max(volume_bins))
            
            # Calculate POC (middle of the bin with highest volume)
            poc = price_min + (max_volume_bin * bin_size) + (bin_size / 2)
            
            return round(poc, 2)
            
        except Exception as e:
            logger.error(f"Error calculating Volume Profile POC: {e}")
            # Return current price as fallback
            if len(df) > 0:
                return df['close'].iloc[-1]
            return 0

    def _add_correlation_analysis(self, data: Dict[str, Any]) -> None:
        """
        Add correlation analysis with other assets
        
        Args:
            data: Market data dictionary
        """
        logger.info("📊 Adding correlation analysis")
        
        try:
            # In a real system, this would fetch actual correlation data
            # For now, we'll use realistic but static values
            
            # Create correlations object if it doesn't exist
            if "correlations" not in data:
                data["correlations"] = {}
            
            # Create a dictionary of correlations first
            correlations = {}
            
            # Add BTC-ETH correlation (usually high positive)
            correlations["btc_eth_correlation"] = round(random.uniform(0.8, 0.95), 2)
            
            # Add BTC-SP500 correlation (variable, trending positive in recent years)
            correlations["btc_sp500_correlation"] = round(random.uniform(0.3, 0.7), 2)
            
            # Add BTC-Gold correlation (usually low correlation)
            correlations["btc_gold_correlation"] = round(random.uniform(-0.2, 0.4), 2)
            
            # Add BTC-DXY correlation (usually negative)
            correlations["btc_dxy_correlation"] = round(random.uniform(-0.7, -0.3), 2)
            
            # Process and add descriptions
            descriptions = {}
            for key, value in correlations.items():
                abs_val = abs(value)
                if abs_val >= 0.7:
                    strength = "Strong"
                elif abs_val >= 0.4:
                    strength = "Moderate"
                else:
                    strength = "Weak"
                    
                direction = "Positive" if value >= 0 else "Negative"
                descriptions[f"{key}_desc"] = f"{strength} {direction}"
            
            # Update the data dictionary with all correlations and descriptions
            data["correlations"].update(correlations)
            data["correlations"].update(descriptions)
                
            logger.info("✅ Correlation analysis added")
            
        except Exception as e:
            logger.error(f"Error adding correlation analysis: {e}")

    def _enhance_order_book_data(self, order_book: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance order book data with advanced metrics
        
        Args:
            order_book: Raw order book data
            
        Returns:
            Dict containing enhanced order book metrics
        """
        try:
            enhanced = order_book.copy()
            
            # Extract bids and asks
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if not bids or not asks:
                return enhanced
            
            # Convert to numpy arrays for faster computation
            bids_array = np.array(bids, dtype=float)
            asks_array = np.array(asks, dtype=float)
            
            # Calculate basic metrics
            bid_volume = np.sum(bids_array[:, 1])
            ask_volume = np.sum(asks_array[:, 1])
            total_volume = bid_volume + ask_volume
            
            # Calculate advanced metrics
            enhanced['metrics'] = {
                'bid_volume': float(bid_volume),
                'ask_volume': float(ask_volume),
                'total_volume': float(total_volume),
                'bid_ask_ratio': float(bid_volume / ask_volume) if ask_volume > 0 else float('inf'),
                'order_imbalance': float((bid_volume - ask_volume) / total_volume) if total_volume > 0 else 0.0,
                'spread': float(asks_array[0, 0] - bids_array[0, 0]) if len(asks_array) > 0 and len(bids_array) > 0 else 0.0,
                'spread_percentage': float((asks_array[0, 0] - bids_array[0, 0]) / asks_array[0, 0] * 100) if len(asks_array) > 0 and len(bids_array) > 0 else 0.0,
                'mid_price': float((asks_array[0, 0] + bids_array[0, 0]) / 2) if len(asks_array) > 0 and len(bids_array) > 0 else 0.0
            }
            
            # Calculate volume-weighted metrics
            bid_vwap = np.sum(bids_array[:, 0] * bids_array[:, 1]) / bid_volume if bid_volume > 0 else 0
            ask_vwap = np.sum(asks_array[:, 0] * asks_array[:, 1]) / ask_volume if ask_volume > 0 else 0
            
            enhanced['metrics'].update({
                'bid_vwap': float(bid_vwap),
                'ask_vwap': float(ask_vwap),
                'vwap_midpoint': float((bid_vwap + ask_vwap) / 2) if bid_vwap > 0 and ask_vwap > 0 else 0.0
            })
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing order book data: {str(e)}")
            logger.error(traceback.format_exc())
            return order_book
    
    def _calculate_price_impact(self, orders: List[List[float]], target_amount: float, side: str) -> Optional[float]:
        """
        Calculate the average price after executing a market order of target_amount
        
        Args:
            orders: List of [price, amount] lists
            target_amount: Target amount to buy/sell
            side: "buy" or "sell"
            
        Returns:
            Impact price or None if not enough liquidity
        """
        try:
            cumulative_amount = 0
            cumulative_value = 0
            
            for order in orders:
                if len(order) < 2:
                    continue
                
                price, amount = order[0], order[1]
                
                # How much of this order will be filled
                available_amount = min(amount, target_amount - cumulative_amount)
                if available_amount <= 0:
                    break
                
                # Add to cumulative
                cumulative_amount += available_amount
                cumulative_value += available_amount * price
                
                # Check if we've reached the target
                if cumulative_amount >= target_amount:
                    # Return the average price
                    return cumulative_value / cumulative_amount
            
            # Not enough liquidity
            return None
            
        except Exception as e:
            logger.error(f"Error calculating price impact: {e}")
            return None
    
    def _klines_to_dataframe(self, klines):
        """
        Convert klines data to pandas DataFrame
        
        Args:
            klines: List of klines from Binance API
            
        Returns:
            pandas DataFrame with OHLCV data
        """
        try:
            import pandas as pd
            
            # Create DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error converting klines to DataFrame: {e}")
            # Return empty DataFrame
            import pandas as pd
            return pd.DataFrame()
    
    def _calculate_indicators(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Calculate technical indicators for a given timeframe"""
        try:
            indicators = {}
            
            # RSI (14 periods)
            rsi = ta.momentum.RSIIndicator(df['close'], window=14)
            indicators['RSI'] = float(rsi.rsi().iloc[-1])
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            upper_band = bb.bollinger_hband()
            lower_band = bb.bollinger_lband()
            middle_band = bb.bollinger_mavg()
            
            # Calculate Bollinger Bands Width
            bb_width = (upper_band - lower_band) / middle_band
            indicators['Bollinger_Bands_Width'] = float(bb_width.iloc[-1])
            
            # MACD (if 1h timeframe)
            if timeframe == '1h':
                macd = ta.trend.MACD(df['close'])
                indicators['MACD'] = {
                    'MACD': float(macd.macd().iloc[-1]),
                    'Signal': float(macd.macd_signal().iloc[-1]),
                    'Histogram': float(macd.macd_diff().iloc[-1])
                }
                
                # Add other 1h specific indicators
                # Parabolic SAR
                psar = ta.trend.PSARIndicator(df['high'], df['low'], df['close'])
                psar_value = float(psar.psar().iloc[-1])
                current_price = float(df['close'].iloc[-1])
                indicators['Parabolic_SAR'] = "ABOVE" if psar_value > current_price else "BELOW"
                
                # EMA 50/200 Crossover
                ema_50 = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
                ema_200 = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
                
                if ema_50.iloc[-1] > ema_200.iloc[-1]:
                    if ema_50.iloc[-2] <= ema_200.iloc[-2]:
                        indicators['EMA_50_200_Crossover'] = "BULLISH_CROSS"
                    else:
                        indicators['EMA_50_200_Crossover'] = "BULLISH"
                elif ema_50.iloc[-1] < ema_200.iloc[-1]:
                    if ema_50.iloc[-2] >= ema_200.iloc[-2]:
                        indicators['EMA_50_200_Crossover'] = "BEARISH_CROSS"
                    else:
                        indicators['EMA_50_200_Crossover'] = "BEARISH"
                else:
                    indicators['EMA_50_200_Crossover'] = "NEUTRAL"
                
                # Williams %R
                williams_r = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'])
                indicators['Williams_%R'] = float(williams_r.williams_r().iloc[-1])
                
                # ADX
                adx = self._calculate_adx(df)
                indicators['ADX'] = float(adx)
                
            elif timeframe == '5m':
                # Add 5m specific indicators
                # Volume Weighted Oscillator (WVO)
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                volume_ma = df['volume'].rolling(window=20).mean()
                wvo = ((typical_price * df['volume']) - (typical_price.rolling(window=20).mean() * volume_ma)) / (typical_price.rolling(window=20).std() * volume_ma.std())
                indicators['WVO'] = float(wvo.iloc[-1])
                
                # Adaptive RSI (using EMA-based smoothing)
                close_diff = df['close'].diff()
                gains = close_diff.where(close_diff > 0, 0)
                losses = -close_diff.where(close_diff < 0, 0)
                
                avg_gains = gains.ewm(span=14, adjust=False).mean()
                avg_losses = losses.ewm(span=14, adjust=False).mean()
                
                rs = avg_gains / avg_losses
                arsi = 100 - (100 / (1 + rs))
                indicators['ARSI'] = float(arsi.iloc[-1])
                
                # Volume Weighted Intensity Oscillator (VWIO)
                price_change = df['close'].diff()
                volume_intensity = price_change * df['volume']
                vwio = volume_intensity.rolling(window=20).mean() / df['volume'].rolling(window=20).mean()
                indicators['VWIO'] = float(vwio.iloc[-1])
                
                # ADX for 5m
                adx = self._calculate_adx(df)
                indicators['ADX'] = float(adx)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {timeframe}: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def _calculate_correlations(self, klines_data: Dict[str, List[List[Any]]]) -> Dict[str, float]:
        """Calculate correlations between different timeframes and assets"""
        try:
            # Convert 1h klines data to DataFrame
            df_1h = pd.DataFrame(klines_data.get('1h', []), columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            if df_1h.empty:
                logger.warning("No 1h timeframe data available for correlation analysis")
                return {}
            
            # Convert numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df_1h[col] = pd.to_numeric(df_1h[col], errors='coerce')
            
            # Calculate correlations
            correlations = {
                'price_volume_correlation': df_1h['close'].corr(df_1h['volume']),
                'high_low_correlation': df_1h['high'].corr(df_1h['low']),
                'open_close_correlation': df_1h['open'].corr(df_1h['close'])
            }
            
            # Calculate volatility correlation
            df_1h['returns'] = df_1h['close'].pct_change()
            df_1h['volatility'] = df_1h['returns'].rolling(window=14).std()
            correlations['volatility_volume_correlation'] = df_1h['volatility'].corr(df_1h['volume'])
            
            # Log correlations
            logger.info(f"✅ Calculated correlations: {correlations}")
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error calculating correlations: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _get_default_sentiment(self) -> Dict[str, Any]:
        """Get default sentiment values when Santiment API is not available"""
        return {
            'social_volume': 0.0,
            'social_dominance': 0.0,
            'sentiment_score': 0.5,
            'news_sentiment': {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0
            },
            'github_activity': {
                'commits': 0,
                'contributors': 0,
                'stars': 0
            },
            'whale_transactions': {
                'count': 0,
                'volume': 0.0
            },
            'exchange_flow': {
                'inflow': 0.0,
                'outflow': 0.0,
                'net_flow': 0.0
            },
            'fear_and_greed': {
                'value': 50,
                'classification': 'Neutral'
            }
        }
        