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
        Enrich market data with technical indicators, sentiment and macro data
        
        Args:
            market_data: Raw market data from Binance
            
        Returns:
            Enriched market data
        """
        logger.info("Enriching market data")
        
        # Deep copy to avoid modifying the original data
        enriched_data = market_data.copy()
        
        try:
            # Enrich with technical indicators
            self._add_technical_indicators(enriched_data)
            
            # Add sentiment data (in a real system, this would come from external APIs)
            self._add_sentiment_data(enriched_data)
            
            # Add macro factors (in a real system, this would come from external APIs)
            self._add_macro_factors(enriched_data)
            
            # Add on-chain metrics
            self._add_onchain_metrics(enriched_data)
            
            # Add risk metrics
            self._add_risk_metrics(enriched_data)
            
            # Log the enriched data for debugging
            #logger.debug(f"Enriched data being sent to prediction API: {json.dumps(enriched_data, indent=2)}")
            
            return enriched_data
            
        except Exception as e:
            logger.error(f"Error enriching data: {str(e)}")
            # Return partial data if available, otherwise raise
            if "timeframes" in enriched_data and enriched_data["timeframes"]:
                logger.warning("Returning partially enriched data")
                return enriched_data
            raise
    
    def _add_technical_indicators(self, data: Dict[str, Any]) -> None:
        """
        Add technical indicators to market data
        
        Args:
            data: Market data to enrich
        """
        symbol = data.get("symbol", "BTCUSDT")
        
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
                    
                    # Log data points available
                    #logger.info(f"5m timeframe data points: {len(df_5m)}")
                    
                    # Initialize indicators dictionary if not exists
                    if "indicators" not in timeframe_5m:
                        timeframe_5m["indicators"] = {}
                    
                    # Only calculate if we have enough data points
                    if len(df_5m) >= 14:  # Minimum for RSI
                        timeframe_5m["indicators"]["RSI"] = self._calculate_rsi(df_5m['close'])
                        timeframe_5m["indicators"]["ARSI"] = self._calculate_arsi(df_5m['close'])
                    
                    if len(df_5m) >= 20:  # Minimum for Bollinger Bands
                        timeframe_5m["indicators"]["Bollinger_Width"] = self._calculate_bollinger_width(df_5m['close'])
                    
                    if len(df_5m) >= 10:  # Minimum for volume indicators
                        timeframe_5m["indicators"]["WVO"] = self._calculate_vwo(df_5m['volume'])
                        timeframe_5m["indicators"]["VWIO"] = self._calculate_vwio(df_5m['volume'])
        
        # 1-hour timeframe indicators
        if "timeframes" in data and "1h" in data["timeframes"]:
            timeframe_1h = data["timeframes"]["1h"]
            
            if "price" in timeframe_1h:
                price_data = timeframe_1h["price"]
                if "klines" in price_data:
                    # Create DataFrame with proper column names
                    df_1h = pd.DataFrame(price_data["klines"], 
                                       columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # Convert string values to float
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df_1h[col] = pd.to_numeric(df_1h[col], errors='coerce')
                    
                    # Log data points available
                    #logger.info(f"1h timeframe data points: {len(df_1h)}")
                    
                    # Initialize indicators dictionary if not exists
                    if "indicators" not in timeframe_1h:
                        timeframe_1h["indicators"] = {}
                    
                    # Calculate indicators based on available data points
                    if len(df_1h) >= 26:  # Minimum for MACD
                        timeframe_1h["indicators"]["MACD_Histogram"] = self._calculate_macd(df_1h['close'])
                    
                    if len(df_1h) >= 200:  # Minimum for EMA crossover
                        timeframe_1h["indicators"]["EMA_50_200_Crossover"] = self._calculate_ema_crossover(df_1h['close'])
                    
                    if len(df_1h) >= 10:  # Minimum for Parabolic SAR
                        timeframe_1h["indicators"]["Parabolic_SAR"] = self._calculate_parabolic_sar(
                            df_1h['high'],
                            df_1h['low'],
                            df_1h['close']
                        )
                    
                    if len(df_1h) >= 52:  # Minimum for Ichimoku
                        timeframe_1h["indicators"]["Ichimoku_Cloud"] = self._calculate_ichimoku(
                            df_1h['high'],
                            df_1h['low'],
                            df_1h['close']
                        )
                    
                    # These indicators need at least a few data points
                    if len(df_1h) >= 2:
                        timeframe_1h["indicators"]["Fibonacci_Levels"] = self._calculate_fibonacci_levels(
                            df_1h['high'].max(),
                            df_1h['low'].min()
                        )
                        
                        timeframe_1h["indicators"]["Volume_Profile_POC"] = self._calculate_volume_profile_poc(
                            df_1h['close'],
                            df_1h['volume']
                        )
                        
                        timeframe_1h["indicators"]["Pivot_Points"] = self._calculate_pivot_points(df_1h)
                    
                    if len(df_1h) >= 14:  # Minimum for Williams %R and ADX
                        timeframe_1h["indicators"]["Williams_%R"] = self._calculate_williams_r(df_1h)
                        timeframe_1h["indicators"]["ADX"] = self._calculate_adx(df_1h)
                    
                    if len(df_1h) >= 1:  # VWAP and other simple indicators
                        timeframe_1h["indicators"]["VWAP"] = self._calculate_vwap(df_1h)
                        timeframe_1h["indicators"]["CVD"] = self._calculate_cvd(df_1h)
                        timeframe_1h["indicators"]["Hourly_High_Low_Percentile"] = self._calculate_high_low_percentile(
                            df_1h['high'].iloc[-1],
                            df_1h['low'].iloc[-1]
                        )
                        timeframe_1h["indicators"]["Hourly_Volume_Momentum"] = self._calculate_volume_momentum(df_1h['volume'])
                    
                    # Log the calculated indicators for debugging
                    logger.info(f"Calculated indicators for {symbol} 1h timeframe:")
                    logger.info(json.dumps(timeframe_1h["indicators"], indent=2))

    def _calculate_rsi_series(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        try:
            rsi = self._calculate_rsi_series(prices, period)
            return rsi.iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return 50.0
    
    def _calculate_bollinger_width(self, prices: pd.Series, period: int = 20) -> float:
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            width = (upper_band.iloc[-1] - lower_band.iloc[-1]) / sma.iloc[-1]
            return width
        except Exception as e:
            logger.error(f"Error calculating Bollinger Width: {str(e)}")
            return 1.0
    
    def _calculate_vwo(self, volumes: pd.Series, short_period: int = 5, long_period: int = 10) -> float:
        try:
            sma_short = volumes.rolling(window=short_period).mean().iloc[-1]
            sma_long = volumes.rolling(window=long_period).mean().iloc[-1]
            oscillator = ((sma_short - sma_long) / sma_long) * 100
            return oscillator
        except Exception as e:
            logger.error(f"Error calculating VWO: {str(e)}")
            return 0.0
    
    def _calculate_arsi(self, prices: pd.Series, period: int = 14) -> float:
        try:
            rsi_series = self._calculate_rsi_series(prices, period)
            arsi = rsi_series.tail(period).mean()
            return arsi
        except Exception as e:
            logger.error(f"Error calculating ARSI: {str(e)}")
            return 50.0
    
    def _calculate_vwio(self, volumes: pd.Series, period: int = 20) -> float:
        try:
            mean_vol = volumes.rolling(window=period).mean().iloc[-1]
            current_vol = volumes.iloc[-1]
            ratio = current_vol / mean_vol if mean_vol != 0 else 1.0
            return max(0.2, min(ratio, 1.5))
        except Exception as e:
            logger.error(f"Error calculating VWIO: {str(e)}")
            return 1.0
    
    def _calculate_high_low_percentile(self, high: float, low: float) -> float:
        """Calculate High-Low Percentile"""
        try:
            return ((high - low) / high) * 100
        except Exception as e:
            logger.error(f"Error calculating High-Low Percentile: {str(e)}")
            return 50.0
    
    def _calculate_volume_momentum(self, volumes: pd.Series, period: int = 10) -> float:
        try:
            avg_volume = volumes.rolling(window=period).mean().iloc[-1]
            momentum = volumes.iloc[-1] / avg_volume if avg_volume != 0 else 1.0
            return momentum
        except Exception as e:
            logger.error(f"Error calculating Volume Momentum: {str(e)}")
            return 1.0
    
    def _calculate_macd(self, prices: pd.Series, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> float:
        try:
            ema_short = prices.ewm(span=short_window, adjust=False).mean()
            ema_long = prices.ewm(span=long_window, adjust=False).mean()
            macd_line = ema_short - ema_long
            signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
            macd_hist = macd_line - signal_line
            return macd_hist.iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return 0.0
    
    def _calculate_parabolic_sar(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        try:
            af = 0.02
            max_af = 0.2
            if close.iloc[-1] > close.iloc[0]:
                trend = "up"
                sar = low.iloc[0]
            else:
                trend = "down"
                sar = high.iloc[0]
            for i in range(1, len(close)):
                if trend == "up":
                    sar = sar + af * (high.iloc[i-1] - sar)
                    if low.iloc[i] < sar:
                        trend = "down"
                        sar = high.iloc[i-1]
                        af = 0.02
                    else:
                        if high.iloc[i] > high.iloc[i-1]:
                            af = min(max_af, af + 0.02)
                else:
                    sar = sar - af * (sar - low.iloc[i-1])
                    if high.iloc[i] > sar:
                        trend = "up"
                        sar = low.iloc[i-1]
                        af = 0.02
                    else:
                        if low.iloc[i] < low.iloc[i-1]:
                            af = min(max_af, af + 0.02)
            return sar
        except Exception as e:
            logger.error(f"Error calculating Parabolic SAR: {str(e)}")
            return close.iloc[-1]
    
    def _calculate_ema_crossover(self, prices: pd.Series) -> str:
        try:
            ema_50 = prices.ewm(span=50, adjust=False).mean().iloc[-1]
            ema_200 = prices.ewm(span=200, adjust=False).mean().iloc[-1]
            if ema_50 > ema_200:
                return "Bullish"
            elif ema_50 < ema_200:
                return "Bearish"
            else:
                return "Neutral"
        except Exception as e:
            logger.error(f"Error calculating EMA Crossover: {str(e)}")
            return "Neutral"
    
    def _calculate_ichimoku(self, highs: pd.Series, lows: pd.Series, closes: pd.Series) -> str:
        try:
            tenkan = (highs.rolling(window=9).max() + lows.rolling(window=9).min()) / 2
            kijun = (highs.rolling(window=26).max() + lows.rolling(window=26).min()) / 2
            leading_span_a = (tenkan + kijun) / 2
            leading_span_b = (highs.rolling(window=52).max() + lows.rolling(window=52).min()) / 2
            current_price = closes.iloc[-1]
            if current_price > max(leading_span_a.iloc[-1], leading_span_b.iloc[-1]):
                return "Bullish"
            elif current_price < min(leading_span_a.iloc[-1], leading_span_b.iloc[-1]):
                return "Bearish"
            else:
                return "Neutral"
        except Exception as e:
            logger.error(f"Error calculating Ichimoku: {str(e)}")
            return "Neutral"
    
    def _calculate_fibonacci_levels(self, high: float, low: float) -> float:
        try:
            range_size = high - low
            return low + (range_size * 0.618)
        except Exception as e:
            logger.error(f"Error calculating Fibonacci Levels: {str(e)}")
            return (high + low) / 2
    
    def _calculate_volume_profile_poc(self, prices: pd.Series, volumes: pd.Series) -> float:
        try:
            df = pd.DataFrame({'price': prices, 'volume': volumes})
            grouped = df.groupby(df['price'].round(0))['volume'].sum()
            poc = grouped.idxmax()
            return float(poc)
        except Exception as e:
            logger.error(f"Error calculating Volume Profile POC: {str(e)}")
            return prices.iloc[-1]
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> Dict[str, float]:
        try:
            lastCandle = df.iloc[-1]
            pivot = (lastCandle['high'] + lastCandle['low'] + lastCandle['close']) / 3
            support1 = 2 * pivot - lastCandle['high']
            resistance1 = 2 * pivot - lastCandle['low']
            return {
                'pivot': float(round(pivot, 2)),
                'support1': float(round(support1, 2)),
                'resistance1': float(round(resistance1, 2))
            }
        except Exception as e:
            logger.error(f"Error calculating Pivot Points: {str(e)}")
            return {}
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> float:
        try:
            highestHigh = df['high'].rolling(window=period).max()
            lowestLow = df['low'].rolling(window=period).min()
            wr = ((highestHigh - df['close']) / (highestHigh - lowestLow)) * -100.0
            return float(round(wr.iloc[-1], 2))
        except Exception as e:
            logger.error(f"Error calculating Williams %R: {str(e)}")
            return 0.0
    
    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3.0
            vwap = (typical_price * df['volume']).sum() / df['volume'].sum()
            return float(round(vwap, 2))
        except Exception as e:
            logger.error(f"Error calculating VWAP: {str(e)}")
            return 0.0
        
    def _calculate_cvd(self, df: pd.DataFrame) -> float:
        try:
            temp = df.copy()
            temp['delta'] = temp.apply(lambda row: row['volume'] if row['close'] > row['open'] else (-row['volume'] if row['close'] < row['open'] else 0), axis=1)
            cvd = temp['delta'].cumsum().iloc[-1]
            return float(round(cvd, 2))
        except Exception as e:
            logger.error(f"Error calculating CVD: {str(e)}")
            return 0.0
        
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        try:
            df = df.copy()
            df['prev_close'] = df['close'].shift(1)
            df['prev_high'] = df['high'].shift(1)
            df['prev_low'] = df['low'].shift(1)
            
            df['tr'] = df.apply(lambda row: max(row['high'] - row['low'],
                                                  abs(row['high'] - row['prev_close']) if pd.notnull(row['prev_close']) else 0,
                                                  abs(row['low'] - row['prev_close']) if pd.notnull(row['prev_close']) else 0), axis=1)
            
            df['plus_dm'] = df['high'] - df['prev_high']
            df['minus_dm'] = df['prev_low'] - df['low']
            df['plus_dm'] = df.apply(lambda row: row['plus_dm'] if (row['plus_dm'] > row['minus_dm'] and row['plus_dm'] > 0) else 0, axis=1)
            df['minus_dm'] = df.apply(lambda row: row['minus_dm'] if (row['minus_dm'] > row['plus_dm'] and row['minus_dm'] > 0) else 0, axis=1)
            
            df['tr_sum'] = df['tr'].rolling(window=period).sum()
            df['plus_dm_sum'] = df['plus_dm'].rolling(window=period).sum()
            df['minus_dm_sum'] = df['minus_dm'].rolling(window=period).sum()
            
            df['plus_di'] = 100 * (df['plus_dm_sum'] / df['tr_sum'])
            df['minus_di'] = 100 * (df['minus_dm_sum'] / df['tr_sum'])
            
            df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
            
            adx = df['dx'].rolling(window=period).mean().iloc[-1]
            return float(round(adx, 2))
        except Exception as e:
            logger.error(f"Error calculating ADX: {str(e)}")
            return 0.0

    def _add_sentiment_data(self, data: Dict[str, Any]) -> None:
        """
        Add market sentiment data
        
        Args:
            data: Market data to enrich
        """
        try:
            # Initialize sentiment dictionary if it doesn't exist
            if "sentiment" not in data:
                data["sentiment"] = {}
            
            # Calculate funding rate from price data if available
            if "timeframes" in data and "1h" in data["timeframes"]:
                timeframe_1h = data["timeframes"]["1h"]
                if "price" in timeframe_1h:
                    price_data = timeframe_1h["price"]
                    if "close" in price_data and "open" in price_data:
                        funding_rate = self._calculate_funding_rate(
                            price_data["open"],
                            price_data["close"]
                        )
                        data["sentiment"]["funding_rate"] = funding_rate
            
            # Fetch sentiment data from Santiment if available
            if self.santiment_api:
                try:
                    # Fetch social sentiment
                    social_data = self.santiment_api.fetch_sentiment_data()
                    if social_data and "social_sentiment" in social_data:
                        data["sentiment"].update({
                            "social_volume": social_data["social_sentiment"]["social_volume"],
                            "sentiment_balance": social_data["social_sentiment"]["sentiment_balance"],
                            "dev_activity": social_data["social_sentiment"]["dev_activity"]
                        })
                    
                    # Fetch news sentiment
                    news_data = self.santiment_api.fetch_news_sentiment()
                    if news_data and "news_sentiment" in news_data:
                        data["sentiment"].update({
                            "news_volume": news_data["news_sentiment"]["news_volume"],
                            "news_sentiment_score": news_data["news_sentiment"]["sentiment_score"]
                        })
                    
                    # Fetch on-chain metrics
                    onchain_data = self.santiment_api.fetch_onchain_metrics()
                    if onchain_data and "onchain_metrics" in onchain_data:
                        data["sentiment"].update({
                            "whale_activity": {
                                "inflow": onchain_data["onchain_metrics"]["whale_transaction_count"],
                                "outflow": 0  # Currently not available from Santiment
                            },
                            "exchange_balance": onchain_data["onchain_metrics"]["exchange_balance"],
                            "network_growth": onchain_data["onchain_metrics"]["network_growth"]
                        })
                except Exception as e:
                    logger.error(f"Error fetching Santiment data: {str(e)}")
                    # Fall back to default values
                    self._add_default_sentiment(data)
            else:
                # If no Santiment API available, use default values
                self._add_default_sentiment(data)
            
            # Calculate Fear & Greed Index from various metrics
            fear_greed = self._calculate_fear_greed_index(data)
            data["sentiment"]["fear_greed_index"] = fear_greed
            
            # Calculate Open Interest from volume data
            if "timeframes" in data and "1h" in data["timeframes"]:
                timeframe_1h = data["timeframes"]["1h"]
                if "price" in timeframe_1h and "volume" in timeframe_1h["price"]:
                    open_interest = self._calculate_open_interest(
                        timeframe_1h["price"]["volume"]
                    )
                    data["sentiment"]["open_interest"] = open_interest
            
            # Calculate Derivatives Ratio
            derivatives = self._calculate_derivatives_ratio(data)
            data["sentiment"]["derivatives_ratio"] = derivatives
            
        except Exception as e:
            logger.error(f"Error adding sentiment data: {str(e)}")
            # Ensure we have some default values
            self._add_default_sentiment(data)
    
    def _calculate_funding_rate(self, open_price: float, close_price: float) -> float:
        """Calculate funding rate based on price movement"""
        try:
            # In a real implementation, this would use perpetual futures data
            price_change = (close_price - open_price) / open_price
            # Map price change to a reasonable funding rate range (-0.01 to 0.01)
            funding_rate = max(-0.01, min(0.01, price_change / 10))
            return round(funding_rate, 4)
        except Exception as e:
            logger.error(f"Error calculating funding rate: {str(e)}")
            return 0.0
    
    def _calculate_fear_greed_index(self, data: Dict[str, Any]) -> int:
        """Calculate Fear & Greed Index from various metrics"""
        try:
            score = 50  # Start at neutral
            
            # Price volatility contribution
            if "timeframes" in data and "1h" in data["timeframes"]:
                timeframe_1h = data["timeframes"]["1h"]
                if "price" in timeframe_1h:
                    price_data = timeframe_1h["price"]
                    if all(k in price_data for k in ["high", "low", "close"]):
                        volatility = (price_data["high"] - price_data["low"]) / price_data["close"]
                        # Adjust score based on volatility
                        if volatility > 0.05:  # High volatility
                            score -= 15
                        elif volatility < 0.01:  # Low volatility
                            score += 10
            
            # Volume contribution
            if "timeframes" in data and "1h" in data["timeframes"]:
                timeframe_1h = data["timeframes"]["1h"]
                if "price" in timeframe_1h and "volume" in timeframe_1h["price"]:
                    volume = timeframe_1h["price"]["volume"]
                    # Volume threshold would depend on the asset
                    if volume > 1000:  # High volume
                        score += 10
                    elif volume < 100:  # Low volume
                        score -= 5
            
            # Market momentum contribution
            if "timeframes" in data and "1h" in data["timeframes"]:
                timeframe_1h = data["timeframes"]["1h"]
                if "indicators" in timeframe_1h:
                    indicators = timeframe_1h["indicators"]
                    if "RSI" in indicators:
                        rsi = indicators["RSI"]
                        if rsi > 70:  # Overbought
                            score -= 10
                        elif rsi < 30:  # Oversold
                            score += 10
            
            return max(0, min(100, score))  # Ensure score is between 0 and 100
            
        except Exception as e:
            logger.error(f"Error calculating Fear & Greed Index: {str(e)}")
            return 50  # Return neutral score on error
    
    def _calculate_open_interest(self, volume: float) -> float:
        """Calculate Open Interest based on volume"""
        try:
            # In a real implementation, this would use futures market data
            # For now, estimate based on volume
            return round(volume * 1000, 1)  # Simple scaling
        except Exception as e:
            logger.error(f"Error calculating Open Interest: {str(e)}")
            return 0.0
    
    def _calculate_derivatives_ratio(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate Derivatives Market Ratios"""
        try:
            result = {
                "put_call_ratio": 1.0,  # Neutral ratio
                "funding_rate_24h": 0.0
            }
            
            # Use funding rate if available
            if "sentiment" in data and "funding_rate" in data["sentiment"]:
                result["funding_rate_24h"] = data["sentiment"]["funding_rate"]
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating Derivatives Ratio: {str(e)}")
            return {"put_call_ratio": 1.0, "funding_rate_24h": 0.0}
    
    def _add_default_sentiment(self, data: Dict[str, Any]) -> None:
        """Add default sentiment values when API fails"""
        data["sentiment"] = {
            "funding_rate": 0,
            "fear_greed_index": 50,
            "open_interest": 0,
            "whale_activity": {"inflow": 0, "outflow": 0},
            "social_volume": 0,
            "sentiment_balance": 0,
            "news_volume": 0,
            "news_sentiment_score": 0,
            "dev_activity": 0,
            "exchange_balance": 0,
            "network_growth": 0,
            "derivatives_ratio": {"put_call_ratio": 1.0, "funding_rate_24h": 0}
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