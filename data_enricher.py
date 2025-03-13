"""
Module for enriching market data with technical indicators, sentiment and macro data
"""

import logging
import random
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from functools import lru_cache
import json
import copy

# Import SantimentAPI
from santiment_api import SantimentAPI

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
        Enrich market data with technical indicators, sentiment, and macro factors
        
        Args:
            market_data: Raw market data from Binance
            
        Returns:
            Enriched market data
        """
        # Create a copy to avoid modifying the original
        enriched_data = copy.deepcopy(market_data)
        
        # Add technical indicators
        self._add_technical_indicators(enriched_data)
        
        # Add sentiment data
        self._add_sentiment_data(enriched_data)
        
        # Add macro factors
        self._add_macro_factors(enriched_data)
        
        # Add futures-specific data if available
        if "futures_data" in enriched_data:
            self._add_futures_metrics(enriched_data)
        
        return enriched_data
    
    def _add_technical_indicators(self, data: Dict[str, Any]) -> None:
        """
        Calculate and add technical indicators to market data
        
        Args:
            data: Market data dictionary
        """
        logger.info("🔍 Calculating technical indicators for all timeframes")
        
        # Calculate market_structure before other indicators
        if "timeframes" in data and "1d" in data["timeframes"]:
            timeframe_1d = data["timeframes"]["1d"]
            if "price" in timeframe_1d and "klines" in timeframe_1d["price"]:
                # Create DataFrame
                df_1d = pd.DataFrame(timeframe_1d["price"]["klines"],
                                   columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Convert to numeric
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df_1d[col] = pd.to_numeric(df_1d[col], errors='coerce')
                
                # Initialize indicators
                if "indicators" not in timeframe_1d:
                    timeframe_1d["indicators"] = {}
                
                # Calculate market structure (trend)
                if len(df_1d) >= 14:
                    market_structure = self._calculate_market_structure(df_1d)
                    timeframe_1d["indicators"]["market_structure"] = market_structure
                    logger.info(f"📈 Market structure: {market_structure}")
        
        # 5-minute timeframe indicators
        if "timeframes" in data and "5m" in data["timeframes"]:
            timeframe_5m = data["timeframes"]["5m"]
            
            if "price" in timeframe_5m:
                price_data = timeframe_5m["price"]
                if "klines" in price_data:
                    # Create DataFrame with proper column names
                    df_5m = pd.DataFrame(price_data["klines"], 
                                       columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # Convert string values to float
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df_5m[col] = pd.to_numeric(df_5m[col], errors='coerce')
                    
                    # Initialize indicators dictionary if not exists
                    if "indicators" not in timeframe_5m:
                        timeframe_5m["indicators"] = {}
                    
                    # Only calculate if we have enough data points
                    if len(df_5m) >= 14:  # Minimum for RSI
                        timeframe_5m["indicators"]["RSI"] = self._calculate_rsi(df_5m['close'])
                        timeframe_5m["indicators"]["ARSI"] = self._calculate_arsi(df_5m['close'])
                        timeframe_5m["indicators"]["MACD"] = self._calculate_macd(df_5m['close'])
                        timeframe_5m["indicators"]["ADX"] = self._calculate_adx(df_5m)
                        
                        # Calculate Stochastic RSI for better short term signals
                        timeframe_5m["indicators"]["Stoch_RSI"] = self._calculate_stoch_rsi(df_5m['close'])
                    
                    if len(df_5m) >= 20:  # Minimum for Bollinger Bands
                        timeframe_5m["indicators"]["Bollinger_Width"] = self._calculate_bollinger_width(df_5m['close'])
                        timeframe_5m["indicators"]["Williams_R"] = self._calculate_williams_r(df_5m)
                        timeframe_5m["indicators"]["VWAP"] = self._calculate_vwap(df_5m)
                    
                    if len(df_5m) >= 10:  # Minimum for volume indicators
                        timeframe_5m["indicators"]["WVO"] = self._calculate_vwo(df_5m['volume'])
                        timeframe_5m["indicators"]["VWIO"] = self._calculate_vwio(df_5m['volume'])
                        timeframe_5m["indicators"]["CVD"] = self._calculate_cvd(df_5m)
                        
                    # Calculate momentum score (combined indicator)
                    if len(df_5m) >= 20:
                        timeframe_5m["indicators"]["momentum_score"] = self._calculate_momentum_score(df_5m)
        
        # 1-hour timeframe indicators
        if "timeframes" in data and "1h" in data["timeframes"]:
            timeframe_1h = data["timeframes"]["1h"]
            
            if "price" in timeframe_1h:
                price_data = timeframe_1h["price"]
                if "klines" in price_data:
                    df_1h = pd.DataFrame(price_data["klines"], 
                                       columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df_1h[col] = pd.to_numeric(df_1h[col], errors='coerce')
                    
                    if "indicators" not in timeframe_1h:
                        timeframe_1h["indicators"] = {}
                    
                    if len(df_1h) >= 14:
                        timeframe_1h["indicators"]["RSI"] = self._calculate_rsi(df_1h['close'])
                        timeframe_1h["indicators"]["EMA_Crossover"] = self._calculate_ema_crossover(df_1h['close'])
                        timeframe_1h["indicators"]["Ichimoku"] = self._calculate_ichimoku(df_1h['high'], df_1h['low'], df_1h['close'])
                        timeframe_1h["indicators"]["historical_volatility"] = self._calculate_historical_volatility(df_1h)
                        timeframe_1h["indicators"]["ATR"] = self._calculate_atr(df_1h)
                        
                        # Add pivot points for futures trading
                        timeframe_1h["indicators"]["pivot_points"] = self._calculate_pivot_points(df_1h)
                    
                    if len(df_1h) >= 20:
                        timeframe_1h["indicators"]["Bollinger_Width"] = self._calculate_bollinger_width(df_1h['close'])
                        
                    if len(df_1h) >= 10:
                        timeframe_1h["indicators"]["volume_momentum"] = self._calculate_volume_momentum(df_1h['volume'])
                    
                    # Calculate key levels for support/resistance
                    if len(df_1h) >= 30:
                        timeframe_1h["indicators"]["key_levels"] = self._identify_key_levels(df_1h)
                    
                    # Calculate futures-specific indicators
                    timeframe_1h["indicators"]["sar_position"] = self._calculate_parabolic_sar(df_1h['high'], df_1h['low'], df_1h['close'])
                    
                    if "funding_rate" in data:
                        timeframe_1h["indicators"]["funding_impact"] = self._calculate_funding_impact(data["funding_rate"])
                    
                    # Calculate liquidity zones
                    timeframe_1h["indicators"]["liquidity_zones"] = self._calculate_liquidity_zones(df_1h)
                    
                    # Order flow imbalance score
                    if len(df_1h) >= 20:
                        timeframe_1h["indicators"]["order_flow_imbalance"] = self._calculate_order_flow_imbalance(df_1h)
        
        # 4-hour timeframe indicators
        if "timeframes" in data and "4h" in data["timeframes"]:
            timeframe_4h = data["timeframes"]["4h"]
            
            if "price" in timeframe_4h:
                price_data = timeframe_4h["price"]
                if "klines" in price_data:
                    df_4h = pd.DataFrame(price_data["klines"], 
                                       columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df_4h[col] = pd.to_numeric(df_4h[col], errors='coerce')
                    
                    if "indicators" not in timeframe_4h:
                        timeframe_4h["indicators"] = {}
                    
                    # Calculate trend strength indicators
                    if len(df_4h) >= 14:
                        timeframe_4h["indicators"]["ADX"] = self._calculate_adx(df_4h)
                        timeframe_4h["indicators"]["RSI"] = self._calculate_rsi(df_4h['close'])
                        timeframe_4h["indicators"]["trend_strength"] = self._calculate_trend_strength(df_4h)
                        
                        # Futures position sizing indicator
                        timeframe_4h["indicators"]["optimal_leverage"] = self._calculate_optimal_leverage(df_4h)
                    
                    # Calculate support/resistance levels
                    if len(df_4h) >= 30:
                        timeframe_4h["indicators"]["support_resistance"] = self._calculate_support_resistance(df_4h)
                    
                    # Calculate potential liquidation levels based on volatility
                    timeframe_4h["indicators"]["liquidation_risk"] = self._calculate_liquidation_risk(df_4h)
        
        logger.info("✅ Technical indicators calculated successfully")
    
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
            Default sentiment data
        """
        # Use basic price action to estimate sentiment
        try:
            # Get price data
            price_5m = data.get("timeframes", {}).get("5m", {}).get("price", {})
            price_1h = data.get("timeframes", {}).get("1h", {}).get("price", {})
            
            # Calculate price changes
            price_5m_change = 0
            price_1h_change = 0
            
            if price_5m and "open" in price_5m and "close" in price_5m:
                price_5m_change = (price_5m["close"] - price_5m["open"]) / price_5m["open"] * 100
                
            if price_1h and "open" in price_1h and "close" in price_1h:
                price_1h_change = (price_1h["close"] - price_1h["open"]) / price_1h["open"] * 100
                
            # Map price changes to sentiment values
            sentiment_value = 0
            if price_5m_change > 1 or price_1h_change > 3:
                sentiment_value = 0.7  # Positive
            elif price_5m_change < -1 or price_1h_change < -3:
                sentiment_value = 0.3  # Negative
            else:
                sentiment_value = 0.5  # Neutral
                
            # Create mock sentiment data
            return {
                "social_sentiment": {
                    "twitter": sentiment_value,
                    "reddit": sentiment_value,
                    "telegram": sentiment_value,
                    "overall": sentiment_value
                },
                "news_sentiment": sentiment_value,
                "fear_greed_index": 50 + int(sentiment_value * 50),
                "source": "default"
            }
            
        except Exception as e:
            logger.error(f"Error generating default sentiment: {str(e)}")
            # Return empty sentiment data
            return {
                "social_sentiment": {},
                "source": "default"
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
            return 25.0  # Default value if not enough data
            
        try:
            # Get high, low, close prices
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Calculate price movements
            plus_dm = high.diff()
            minus_dm = low.diff(-1).abs()
            
            # Calculate directional movement
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            # Calculate true range
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            
            # Calculate ATR
            atr = tr.rolling(window=period).mean()
            
            # Calculate plus and minus directional indicators
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
            
            # Calculate directional index
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
            
            # Calculate ADX
            adx = dx.rolling(window=period).mean().iloc[-1]
            
            return round(adx, 2)
        except Exception as e:
            # If there's an error, return a default value
            return 25.0 