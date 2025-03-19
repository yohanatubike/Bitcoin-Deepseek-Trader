"""
Binance Futures API client using the official binance-futures-connector
"""

import time
import logging
import json
from typing import Dict, Any, List, Optional
from binance.um_futures import UMFutures
from binance.error import ClientError, ServerError
import traceback
import math

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BinanceClient:
    """
    Binance Futures client using the official connector
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True, leverage: int = 10, symbol: str = "BTCUSDT"):
        """
        Initialize Binance client for futures trading
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Whether to use testnet (default: True)
            leverage: Default leverage for futures trading (1-125)
            symbol: Default trading symbol (only BTCUSDT supported)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.symbol = symbol
        self.leverage = min(125, max(1, leverage))  # Ensure leverage is between 1-125
        
        # Futures specific attributes
        self.margin_type = "ISOLATED"  # Default margin type
        self.position_mode = "ONE_WAY"  # Default position mode
        
        # Initialize the official futures client
        self.client = UMFutures(
            key=api_key,
            secret=api_secret,
            base_url="https://testnet.binancefuture.com" if testnet else "https://fapi.binance.com"
        )
        
        # Log connection info
        if self.testnet:
            logger.info("🌐 Using Binance Futures TESTNET")
        else:
            logger.info("🌐 Using Binance Futures PRODUCTION")
        
        # Initialize exchange info cache
        self.symbol_info_cache = {}
        self._exchange_info = {}
        self._exchange_info_timestamp = 0
        
        # Check connection
        self._test_connection()
        
        # Initialize futures settings
        self._initialize_futures_settings()
        
        logger.info(f"✅ Binance Futures client initialized for {symbol} with {leverage}x leverage")
    
    def _mask_sensitive_data(self, data: str) -> str:
        """
        Mask sensitive data for logging
        
        Args:
            data: Data to mask
            
        Returns:
            Masked data
        """
        masked_data = data
        
        if self.api_key and len(self.api_key) > 8:
            masked_data = masked_data.replace(self.api_key, f"{self.api_key[:4]}...{self.api_key[-4:]}")
            
        if self.api_secret and len(self.api_secret) > 8:
            masked_data = masked_data.replace(self.api_secret, f"{self.api_secret[:4]}...{self.api_secret[-4:]}")
            
        return masked_data
    
    def _test_connection(self):
        """
        Test connection to Binance API
        """
        try:
            # Test ping endpoint
            result = self.client.ping()
            logger.info("✅ Successfully connected to Binance Futures API")
            
            # Test API key permissions
            try:
                self.test_api_permissions()
            except Exception as e:
                logger.warning(f"⚠️ API permissions test failed: {str(e)}")
                
        except Exception as e:
            logger.error(f"❌ Basic connectivity test failed: {str(e)}")
            logger.warning("⚠️ API key is missing some permissions. See logs for details.")
            logger.error("❌ The API key is invalid or has been revoked. Generate a new API key.")
    
    def _initialize_futures_settings(self):
        """
        Initialize futures settings like leverage and margin type
        """
        try:
            logger.info(f"Initializing futures settings for {self.symbol}")
            
            # For futures mode, set leverage
            try:
                self.set_leverage(self.symbol, self.leverage)
                logger.info(f"Set default leverage {self.leverage}x for {self.symbol}")
            except Exception as e:
                logger.warning(f"Could not set leverage for {self.symbol}: {str(e)}")
            
            # Set margin type to ISOLATED by default
            try:
                self.set_margin_type(self.symbol, self.margin_type)
                logger.info(f"Set margin type to {self.margin_type} for {self.symbol}")
            except Exception as e:
                # Ignore if already set
                if "already" not in str(e).lower():
                    logger.warning(f"Could not set margin type for {self.symbol}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error initializing futures settings: {str(e)}")
    
    def fetch_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch market data for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Market data dictionary
        """
        if symbol != self.symbol:
            logger.warning(f"This client is configured for {self.symbol} only. Ignoring request for {symbol}.")
            symbol = self.symbol
            
        try:
            market_data = {
                "symbol": symbol,
                "timestamp": int(time.time() * 1000),
                "timeframes": {
                    "5m": {
                        "price": {},
                        "indicators": {}
                    },
                    "1h": {
                        "price": {},
                        "indicators": {}
                    }
                },
                "order_book": {},
                "futures_data": {}
            }
            
            # Fetch 5-minute klines
            try:
                klines_5m = self.client.klines(
                    symbol=symbol,
                    interval="5m",
                    limit=100
                )
                
                # Format klines data
                market_data["timeframes"]["5m"]["price"] = {
                    "open": float(klines_5m[-1][1]),
                    "high": float(klines_5m[-1][2]),
                    "low": float(klines_5m[-1][3]),
                    "close": float(klines_5m[-1][4]),
                    "volume": float(klines_5m[-1][5]),
                    "klines": [
                        {
                            "timestamp": int(k[0]),
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[5])
                        } for k in klines_5m
                    ]
                }
            except Exception as e:
                logger.error(f"Error fetching 5m klines: {str(e)}")
            
            # Fetch 1-hour klines
            try:
                klines_1h = self.client.klines(
                    symbol=symbol,
                    interval="1h",
                    limit=100
                )
                
                # Format klines data
                market_data["timeframes"]["1h"]["price"] = {
                    "open": float(klines_1h[-1][1]),
                    "high": float(klines_1h[-1][2]),
                    "low": float(klines_1h[-1][3]),
                    "close": float(klines_1h[-1][4]),
                    "volume": float(klines_1h[-1][5]),
                    "klines": [
                        {
                            "timestamp": int(k[0]),
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[5])
                        } for k in klines_1h
                    ]
                }
            except Exception as e:
                logger.error(f"Error fetching 1h klines: {str(e)}")
            
            # Initialize klines_4h
            klines_4h = []
            
            # Fetch 4-hour klines
            try:
                klines_4h = self.client.klines(
                    symbol=symbol,
                    interval="4h",
                    limit=100
                )
            except Exception as e:
                logger.error(f"Error fetching 4h klines: {str(e)}")
            
            # Fetch order book
            try:
                order_book = self.client.depth(symbol=symbol, limit=20)
                
                # Calculate bid-ask spread
                best_bid = float(order_book["bids"][0][0])
                best_ask = float(order_book["asks"][0][0])
                spread = best_ask - best_bid
                spread_pct = (spread / best_bid) * 100
                
                # Calculate order imbalance
                total_bid_qty = sum(float(bid[1]) for bid in order_book["bids"][:5])
                total_ask_qty = sum(float(ask[1]) for ask in order_book["asks"][:5])
                imbalance = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty)
                
                market_data["order_book"] = {
                    "bids": [[float(bid[0]), float(bid[1])] for bid in order_book["bids"]],
                    "asks": [[float(ask[0]), float(ask[1])] for ask in order_book["asks"]],
                    "bid_ask_spread": spread,
                    "bid_ask_spread_pct": spread_pct,
                    "order_imbalance": imbalance,
                    "bid_ask_ratio": total_bid_qty / total_ask_qty if total_ask_qty > 0 else 0
                }
            except Exception as e:
                logger.error(f"Error fetching order book: {str(e)}")
            
            # Fetch futures-specific data
            try:
                # Get funding rate
                funding_rate_data = self.client.funding_rate(symbol=symbol, limit=1)
                if funding_rate_data and len(funding_rate_data) > 0:
                    # Check if the expected keys exist
                    funding_data = funding_rate_data[0]
                    funding_rate = float(funding_data.get('fundingRate', 0))
                    
                    # Safely get nextFundingTime or use current time + 8 hours as fallback
                    if 'nextFundingTime' in funding_data:
                        next_funding_time = int(funding_data['nextFundingTime'])
                    else:
                        # Funding typically occurs every 8 hours, so add 8 hours (in milliseconds)
                        next_funding_time = int(time.time() * 1000) + (8 * 60 * 60 * 1000)
                        logger.warning(f"nextFundingTime not found in API response, using estimated time")
                else:
                    funding_rate = 0
                    # Use current time + 8 hours as fallback
                    next_funding_time = int(time.time() * 1000) + (8 * 60 * 60 * 1000)
                    logger.warning(f"No funding rate data available, using defaults")
                
                # Get open interest
                open_interest_data = self.client.open_interest(symbol=symbol)
                open_interest = float(open_interest_data['openInterest'])
                
                # Get symbol info for max leverage
                symbol_info = self.get_exchange_info(symbol)
                max_leverage = 20  # Default value
                
                for s in symbol_info.get("symbols", []):
                    if s.get("symbol") == symbol:
                        for info in s.get("info", {}).get("leverageBracket", []):
                            if info["bracket"] == 0:
                                max_leverage = info["initialLeverage"]
                                break
                
                market_data["futures_data"] = {
                    "funding_rate": funding_rate,
                    "next_funding_time": next_funding_time,
                    "open_interest": open_interest,
                    "max_leverage": max_leverage,
                    "funding_rate_analysis": {
                        "sentiment": "Bullish" if funding_rate < 0 else "Bearish" if funding_rate > 0 else "Neutral",
                        "magnitude": "High" if abs(funding_rate) > 0.01 else "Medium" if abs(funding_rate) > 0.005 else "Low",
                        "annualized_rate": funding_rate * 100 * 3 * 365  # 3 fundings per day, 365 days
                    },
                    "open_interest_analysis": {
                        "formatted": f"{open_interest/1000:.2f}k",
                        "value": open_interest
                    },
                    "leverage_analysis": {
                        "max_leverage": max_leverage,
                        "category": "High" if max_leverage >= 50 else "Medium" if max_leverage >= 20 else "Low"
                    }
                }
            except Exception as e:
                logger.error(f"Error fetching futures data: {str(e)}")
            
            # Add klines to market data
            market_data["klines"] = {
                "5m": klines_5m,
                "1h": klines_1h,
                "4h": klines_4h
            }
            
            # Log the structure of the market data
            logger.info(f"Market data structure: {json.dumps({k: type(v).__name__ for k, v in market_data.items()})}")
            logger.info(f"Klines structure: {json.dumps({k: len(v) for k, v in market_data.get('klines', {}).items()})}")
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def get_open_algo_orders(self, symbol: str = None) -> List[Dict]:
        """
        Get all open algorithmic orders (stop loss, take profit)
        
        Args:
            symbol: Optional symbol to filter orders
            
        Returns:
            List of open algorithmic orders
        """
        try:
            if symbol is None:
                symbol = self.symbol
                
            orders = self.client.get_open_orders(symbol=symbol)
            
            # Filter for algorithmic orders (STOP_LOSS, TAKE_PROFIT, etc.)
            algo_orders = [
                order for order in orders 
                if order.get("type") in ["STOP", "STOP_MARKET", "TAKE_PROFIT", "TAKE_PROFIT_MARKET"]
            ]
            
            return algo_orders
            
        except Exception as e:
            logger.error(f"Error getting open algo orders: {str(e)}")
            return []
            
    def cancel_algo_order(self, symbol: str, order_id: int) -> bool:
        """
        Cancel a specific algorithmic order
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID to cancel
            
        Returns:
            Whether the cancellation was successful
        """
        try:
            self.client.cancel_order(symbol=symbol, orderId=order_id)
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id} for {symbol}: {str(e)}")
            return False
    
    def manage_algo_orders(self, symbol: str) -> bool:
        """
        Manage algorithmic orders for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Whether the management operation was successful
        """
        try:
            # Get current positions
            positions = self.get_positions(symbol)
            
            # If no positions, cancel all open algo orders
            if not positions:
                open_orders = self.get_open_algo_orders(symbol)
                if open_orders:
                    for order in open_orders:
                        order_id = order.get("orderId")
                        logger.info(f"Cancelling orphaned algo order {order_id} for {symbol}")
                        self.cancel_algo_order(symbol, order_id)
                    logger.info(f"Cancelled {len(open_orders)} orphaned algo orders for {symbol}")
                return True
                
            return True
            
        except Exception as e:
            logger.error(f"Error managing algo orders for {symbol}: {str(e)}")
            return False
    
    def execute_order(self, symbol: str, side: str, quantity: float, 
                      order_type: str = "MARKET", price: float = None,
                      stop_loss: float = None, take_profit: float = None,
                      reduce_only: bool = False, raw_quantity: bool = False) -> Dict:
        """
        Execute a futures order with optional stop loss and take profit
        
        Args:
            symbol: Trading pair symbol
            side: Order side (BUY or SELL)
            quantity: Order quantity
            order_type: Order type (MARKET, LIMIT, etc.)
            price: Order price (required for LIMIT orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
            reduce_only: Whether the order should only reduce a position
            raw_quantity: If True, use the quantity as provided without formatting (for exact quantity specification)
            
        Returns:
            Order response
        """
        if symbol != self.symbol:
            logger.warning(f"This client is configured for {self.symbol} only. Ignoring request for {symbol}.")
            symbol = self.symbol
        
        try:
            # Format the quantity if needed
            original_quantity = quantity
            if not raw_quantity:
                quantity = self.format_quantity(symbol, quantity)
                
                # Double-check the notional value meets minimum requirements
                current_price = self.get_current_price(symbol)
                if current_price > 0:
                    notional_value = quantity * current_price
                    if notional_value < 100.0:  # Binance minimum is 100 USDT
                        logger.warning(f"Order would fail: Notional value {notional_value:.2f} USDT is less than 100 USDT minimum")
                        # Force a quantity that meets the minimum
                        min_qty = (100.0 * 1.01) / current_price  # Add 1% buffer
                        quantity = self.format_quantity(symbol, min_qty)
                        logger.info(f"Adjusted quantity from {original_quantity} to {quantity} to ensure minimum notional value")
            
            # Format prices
            if price is not None:
                price = self.format_price(symbol, price)
                
            if stop_loss is not None:
                stop_loss = self.format_price(symbol, stop_loss)
                
            if take_profit is not None:
                take_profit = self.format_price(symbol, take_profit)
            
            # Prepare order parameters
            params = {
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "quantity": quantity
            }
            
            # Add price for LIMIT orders
            if order_type == "LIMIT" and price is not None:
                params["price"] = price
                params["timeInForce"] = "GTC"
                
            # Add reduceOnly parameter for futures
            if reduce_only:
                params["reduceOnly"] = "true"
            
            # Execute the main order
            logger.info(f"Executing {side} {order_type} order: {quantity} {symbol} (estimated value: {quantity * self.get_current_price(symbol):.2f} USDT)")
            response = self.client.new_order(**params)
            
            logger.info(f"Order executed: {response}")
            
            # Set up stop loss if provided
            if stop_loss is not None and order_type == "MARKET":
                try:
                    sl_side = "SELL" if side == "BUY" else "BUY"
                    
                    sl_params = {
                        "symbol": symbol,
                        "side": sl_side,
                        "type": "STOP_MARKET",
                        "quantity": quantity,
                        "stopPrice": stop_loss,
                        "reduceOnly": "true"
                    }
                    
                    logger.info(f"Placing stop loss at {stop_loss} for {quantity} {symbol}")
                    sl_response = self.client.new_order(**sl_params)
                    logger.info(f"Stop loss placed at {stop_loss}: {sl_response}")
                    
                except Exception as e:
                    logger.error(f"Error placing stop loss: {str(e)}")
            
            # Set up take profit if provided
            if take_profit is not None and order_type == "MARKET":
                try:
                    tp_side = "SELL" if side == "BUY" else "BUY"
                    
                    tp_params = {
                        "symbol": symbol,
                        "side": tp_side,
                        "type": "TAKE_PROFIT_MARKET",
                        "quantity": quantity,
                        "stopPrice": take_profit,
                        "reduceOnly": "true"
                    }
                    
                    logger.info(f"Placing take profit at {take_profit} for {quantity} {symbol}")
                    tp_response = self.client.new_order(**tp_params)
                    logger.info(f"Take profit placed at {take_profit}: {tp_response}")
                    
                except ClientError as e:
                    error_code = str(e).split(",")[1].strip() if "," in str(e) else ""
                    error_msg = str(e).split(",")[2].strip() if len(str(e).split(",")) > 2 else ""
                    
                    if "-1111" in error_code and "Precision" in error_msg:
                        # Try to fix precision issue and retry
                        logger.warning(f"Precision error for take profit: {error_msg}")
                        
                        # Get the price filter to determine correct precision
                        symbol_info = self.get_exchange_info(symbol)
                        price_filter = None
                        for f in symbol_info.get("filters", []):
                            if f.get("filterType") == "PRICE_FILTER":
                                price_filter = f
                                break
                        
                        if price_filter:
                            # Get the correct precision for this symbol
                            tick_size = float(price_filter.get("tickSize", 0.1))
                            if symbol == "BTCUSDT":
                                # For BTCUSDT, hardcode to 1 decimal place as this is standard
                                corrected_price = round(float(take_profit), 1)
                            else:
                                # For other symbols, try to reduce precision
                                corrected_price = round(float(take_profit) / tick_size) * tick_size
                            
                            # Retry with corrected price
                            logger.info(f"Retrying take profit with corrected price: {corrected_price}")
                            tp_params["stopPrice"] = corrected_price
                            
                            try:
                                tp_retry_response = self.client.new_order(**tp_params)
                                logger.info(f"Take profit placed after precision correction at {corrected_price}: {tp_retry_response}")
                            except Exception as retry_e:
                                logger.error(f"Error placing take profit after correction: {str(retry_e)}")
                    else:
                        logger.error(f"Error placing take profit: {str(e)}")
                except Exception as e:
                    logger.error(f"Error placing take profit: {str(e)}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error executing order: {str(e)}")
            return {"error": str(e)}
    
    def get_account_info(self) -> Dict:
        """
        Get futures account information
        
        Returns:
            Account information
        """
        try:
            return self.client.account()
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return {}
    
    def get_asset_balance(self, asset: str) -> Dict[str, float]:
        """
        Get futures balance for an asset
        
        Args:
            asset: Asset symbol (e.g., USDT)
            
        Returns:
            Balance information
        """
        try:
            account_info = self.get_account_info()
            
            # Find the asset in the account
            for balance in account_info.get("assets", []):
                if balance.get("asset") == asset:
                    return {
                        "asset": asset,
                        "free": float(balance.get("availableBalance", 0)),
                        "locked": float(balance.get("initialMargin", 0)),
                        "total": float(balance.get("walletBalance", 0))
                    }
            
            # Asset not found
            return {
                "asset": asset,
                "free": 0.0,
                "locked": 0.0,
                "total": 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting asset balance: {str(e)}")
            return {
                "asset": asset,
                "free": 0.0,
                "locked": 0.0,
                "total": 0.0
            }
    
    def get_account_metrics(self) -> Dict[str, float]:
        """
        Get account metrics for futures account
        
        Returns:
            Account metrics (balance, equity, margin, etc.)
        """
        try:
            account_info = self.get_account_info()
            
            # Extract account metrics
            total_wallet_balance = float(account_info.get("totalWalletBalance", 0))
            total_unrealized_profit = float(account_info.get("totalUnrealizedProfit", 0))
            total_margin_balance = float(account_info.get("totalMarginBalance", 0))
            total_initial_margin = float(account_info.get("totalInitialMargin", 0))
            total_maint_margin = float(account_info.get("totalMaintMargin", 0))
            available_balance = float(account_info.get("availableBalance", 0))
            
            # Calculate metrics
            equity = total_wallet_balance + total_unrealized_profit
            used_margin = total_initial_margin
            free_margin = available_balance
            margin_level = (equity / used_margin) * 100 if used_margin > 0 else 0
            
            return {
                "balance_usd": total_wallet_balance,
                "equity_usd": equity,
                "margin_used_usd": used_margin,
                "margin_free_usd": free_margin,
                "margin_level": margin_level,
                "unrealized_pnl_usd": total_unrealized_profit
            }
            
        except Exception as e:
            logger.error(f"Error getting account metrics: {str(e)}")
            return {
                "balance_usd": 0.0,
                "equity_usd": 0.0,
                "margin_used_usd": 0.0,
                "margin_free_usd": 0.0,
                "margin_level": 0.0,
                "unrealized_pnl_usd": 0.0
            }
    
    def test_api_permissions(self) -> Dict:
        """
        Test API key permissions
        
        Returns:
            Dictionary with permission status
        """
        results = {
            "futures_read": False,
            "futures_write": False,
            "permissions_valid": False
        }
        
        try:
            # Check if ping works
            ping_result = self.client.ping()
            
            # Check futures account access (read)
            try:
                account_result = self.client.account()
                # If we got this far, we have futures read access
                results["futures_read"] = True
                logger.info("Futures account access: OK")
            except Exception as e:
                logger.warning(f"No futures account read access: {str(e)}")
            
            # Try to get open orders (read)
            try:
                orders_result = self.client.get_open_orders(symbol=self.symbol)
                # If we reached here without error, then we can read orders
                results["futures_read"] = True
                logger.info("Futures open orders access: OK")
            except Exception as e:
                if "API-key format invalid" in str(e):
                    logger.error("The API key is invalid.")
                    results["permissions_valid"] = False
                    return results
                logger.warning(f"No open orders read access: {str(e)}")
                
            # Test writing permissions by setting leverage
            try:
                current_leverage = self.client.change_leverage(symbol=self.symbol, leverage=self.leverage)
                results["futures_write"] = True
                logger.info("Futures write access: OK")
            except Exception as e:
                logger.warning(f"No futures write access: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error testing API permissions: {str(e)}")
        
        # Summarize the results
        if results["futures_read"] and results["futures_write"]:
            logger.info("API key has all required permissions for futures trading.")
            results["permissions_valid"] = True
        else:
            logger.warning("API key is missing some permissions. See logs for details.")
            
            # Provide detailed guidance based on the failures
            if not results["futures_read"]:
                logger.error("The API key does not have 'Read Info' permission for futures trading. Enable this permission for your API key.")
            elif not results["futures_write"]:
                logger.error("The API key does not have 'Trading' permission for futures trading. Enable this permission for your API key.")
        
        return results
    
    def get_exchange_info(self, symbol: str = None) -> Dict:
        """
        Get exchange information
        
        Args:
            symbol: Optional symbol to filter exchange info
            
        Returns:
            Exchange information
        """
        # Check if we have cached exchange info that's less than 1 hour old
        current_time = time.time()
        if self._exchange_info and (current_time - self._exchange_info_timestamp) < 3600:
            # Return cached info
            if symbol:
                # Filter for specific symbol
                for sym_info in self._exchange_info.get("symbols", []):
                    if sym_info.get("symbol") == symbol:
                        return sym_info
                return {}
            return self._exchange_info
        
        # Fetch fresh exchange info
        try:
            # Get all exchange info first (without symbol parameter)
            exchange_info = self.client.exchange_info()
            
            # Cache the result
            self._exchange_info = exchange_info
            self._exchange_info_timestamp = current_time
            
            # If symbol was specified, filter the response
            if symbol:
                for sym_info in exchange_info.get("symbols", []):
                    if sym_info.get("symbol") == symbol:
                        self.symbol_info_cache[symbol] = sym_info
                        return sym_info
                return {}
                
            # Cache individual symbols
            for sym_info in exchange_info.get("symbols", []):
                sym = sym_info.get("symbol")
                if sym:
                    self.symbol_info_cache[sym] = sym_info
                    
            return exchange_info
            
        except Exception as e:
            logger.error(f"Error getting exchange info: {str(e)}")
            return {}
    
    def format_price(self, symbol: str, price: float) -> float:
        """
        Format price according to symbol's price filter
        
        Args:
            symbol: Trading pair symbol
            price: Price to format
            
        Returns:
            Formatted price
        """
        # Get symbol info
        symbol_info = self.symbol_info_cache.get(symbol)
        if not symbol_info:
            symbol_info = self.get_exchange_info(symbol)
            
        # Get price filter
        price_filter = None
        for f in symbol_info.get("filters", []):
            if f.get("filterType") == "PRICE_FILTER":
                price_filter = f
                break
                
        if price_filter:
            tick_size = float(price_filter.get("tickSize", 0.1))
            
            # Calculate precision from tick size
            precision = 0
            if "." in str(tick_size):
                precision = len(str(tick_size).split(".")[1])
                # Remove trailing zeros from precision calculation
                tick_size_str = str(tick_size).rstrip('0').rstrip('.')
                if '.' in tick_size_str:
                    precision = len(tick_size_str.split('.')[1])
            
            # Round to the nearest tick size
            rounded_price = round(price / tick_size) * tick_size
            
            # Apply the precision to the final result
            result = round(rounded_price, precision)
            
            # Log the formatting for debugging
            logger.debug(f"Price formatting: Original={price}, TickSize={tick_size}, Precision={precision}, Result={result}")
            
            return result
        
        # Hard-coded precision for BTCUSDT if filter not found
        if symbol == "BTCUSDT":
            return round(price, 1)  # BTCUSDT typically has 1 decimal place precision
            
        # Default: round to 2 decimal places
        return round(price, 2)
    
    def format_quantity(self, symbol: str, quantity: float) -> float:
        """
        Format quantity according to symbol's lot size filter
        
        Args:
            symbol: Trading pair symbol
            quantity: Quantity to format
            
        Returns:
            Formatted quantity
        """
        # Get symbol info
        symbol_info = self.symbol_info_cache.get(symbol)
        if not symbol_info:
            symbol_info = self.get_exchange_info(symbol)
            
        # Get lot size filter
        lot_size_filter = None
        min_notional = 100.0  # Default minimum notional value in USDT
        
        for f in symbol_info.get("filters", []):
            if f.get("filterType") == "LOT_SIZE":
                lot_size_filter = f
            elif f.get("filterType") == "MIN_NOTIONAL":
                min_notional = float(f.get("minNotional", 100.0))
                
        if lot_size_filter:
            step_size = float(lot_size_filter.get("stepSize", 0.001))
            min_qty = float(lot_size_filter.get("minQty", 0.001))
            
            # Round down to the nearest step size using math.floor instead of int
            quantity = math.floor(quantity / step_size) * step_size
            
            # Ensure minimum quantity
            if quantity < min_qty:
                quantity = min_qty
                
            # Apply precision based on step size
            precision = 0
            if "." in str(step_size):
                precision = len(str(step_size).split(".")[1])
            
            result = round(quantity, precision)
            
            # Check if the formatted quantity meets minimum notional value
            try:
                current_price = self.get_current_price(symbol)
                if current_price > 0:
                    notional_value = result * current_price
                    if notional_value < min_notional:
                        # Calculate the minimum quantity needed to meet min_notional
                        min_required_qty = min_notional / current_price
                        # Round up to the nearest step size to ensure we meet the minimum
                        min_required_qty = math.ceil(min_required_qty / step_size) * step_size
                        # Apply precision
                        result = round(min_required_qty, precision)
                        logger.info(f"Adjusted quantity from {quantity} to {result} to meet minimum notional value of {min_notional} USDT")
            except Exception as e:
                logger.warning(f"Error checking minimum notional value: {str(e)}")
            
            return result
        
        # Default: round to 8 decimal places
        return round(quantity, 8)
    
    def get_leverage(self, symbol: str = None) -> int:
        """
        Get current leverage for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Current leverage
        """
        if symbol is None:
            symbol = self.symbol
            
        try:
            position_info = self.client.get_position_risk(symbol=symbol)
            if position_info:
                return int(position_info[0].get("leverage", self.leverage))
            return self.leverage
        except Exception as e:
            logger.error(f"Error getting leverage: {str(e)}")
            return self.leverage
    
    def set_leverage(self, symbol, leverage):
        """
        Set leverage for a futures trading pair.
        
        Args:
            symbol: Trading pair symbol
            leverage: Leverage value (1-125)
            
        Returns:
            API response
        """
        # Ensure leverage is within valid range
        leverage = max(1, min(leverage, 125))
        
        try:
            return self.client.change_leverage(symbol=symbol, leverage=leverage)
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
            return None
    
    def set_margin_type(self, symbol, margin_type):
        """
        Set margin type for a futures trading pair.
        
        Args:
            symbol: Trading pair symbol
            margin_type: Margin type (ISOLATED or CROSSED)
            
        Returns:
            API response
        """
        if margin_type not in ["ISOLATED", "CROSSED"]:
            raise ValueError("Margin type must be either ISOLATED or CROSSED")
        
        try:
            # First check current margin type
            current_position_info = self.get_position_risk(symbol)
            
            # If we have position info, check if margin type already matches
            if current_position_info and len(current_position_info) > 0:
                current_margin_type = current_position_info[0].get("marginType", "")
                
                # If margin type already matches, no need to change
                if current_margin_type.upper() == margin_type:
                    logger.info(f"Margin type for {symbol} is already set to {margin_type}")
                    return {"msg": f"Margin type already set to {margin_type}"}
            
            # If we get here, we need to change the margin type
            return self.client.change_margin_type(symbol=symbol, marginType=margin_type)
        except Exception as e:
            error_str = str(e).lower()
            # Handle cases where margin type is already set or doesn't need to be changed
            if "already" in error_str or "no need to change margin type" in error_str or "-4046" in error_str:
                logger.info(f"Margin type for {symbol} is already set to {margin_type}")
                return {"msg": f"Margin type already set to {margin_type}"}
            logger.error(f"Error setting margin type: {e}")
            return None

    def get_positions(self, symbol: str = None) -> List[Dict]:
        """
        Get current futures positions
        
        Args:
            symbol: Optional symbol to filter positions
            
        Returns:
            List of positions
        """
        try:
            # Get position risk information
            position_risk = self.get_position_risk(symbol)
            
            # Filter positions with non-zero amounts
            positions = [p for p in position_risk if abs(float(p.get("positionAmt", 0))) > 0]
            
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return []

    def get_position_risk(self, symbol=None):
        """
        Get position risk information for futures positions.
        
        Args:
            symbol: Optional symbol to filter positions
            
        Returns:
            List of position risk information
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
            
        try:
            return self.client.get_position_risk(**params)
        except Exception as e:
            logger.error(f"Error getting position risk: {e}")
            return []

    def get_current_price(self, symbol: str = None) -> float:
        """
        Get current price for a symbol
        
        Args:
            symbol: Trading pair symbol (defaults to self.symbol if None)
            
        Returns:
            Current price as float
        """
        try:
            if symbol is None:
                symbol = self.symbol
                
            # Use the ticker endpoint to get the current price
            ticker = self.client.ticker_price(symbol=symbol)
            
            if isinstance(ticker, dict) and "price" in ticker:
                return float(ticker["price"])
            elif isinstance(ticker, list) and len(ticker) > 0 and "price" in ticker[0]:
                # Sometimes returns a list for a single symbol
                for t in ticker:
                    if t.get("symbol") == symbol:
                        return float(t["price"])
                return float(ticker[0]["price"])
            else:
                logger.error(f"Unexpected ticker response format: {ticker}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return 0.0

    def create_stop_loss_order(self, symbol: str, side: str, stop_price: float, quantity: float, close_position: bool = True) -> dict:
        """
        Create a stop loss order
        
        Args:
            symbol: Trading pair symbol
            side: Order side (BUY/SELL)
            stop_price: Stop price to trigger the order
            quantity: Order quantity
            close_position: Whether to close the entire position
        
        Returns:
            dict: Order response from Binance
        """
        try:
            # Format the quantity according to symbol info
            formatted_quantity = self.format_quantity(symbol, quantity)
            formatted_price = self.format_price(symbol, stop_price)
            
            params = {
                "symbol": symbol,
                "side": side,
                "type": "STOP_MARKET",
                "quantity": formatted_quantity,
                "stopPrice": formatted_price,
                "closePosition": str(close_position).lower(),
                "workingType": "MARK_PRICE",  # Use mark price to avoid premature triggers
                "timeInForce": "GTC"
            }
            
            # Using new_order instead of futures_create_order
            response = self.client.new_order(**params)
            logger.info(f"Created stop loss order: {side} {formatted_quantity} {symbol} @ {formatted_price}")
            return response
            
        except Exception as e:
            logger.error(f"Error creating stop loss order: {str(e)}")
            return None

    def create_take_profit_order(self, symbol: str, side: str, stop_price: float, quantity: float, close_position: bool = True) -> dict:
        """
        Create a take profit order
        
        Args:
            symbol: Trading pair symbol
            side: Order side (BUY/SELL)
            stop_price: Stop price to trigger the order
            quantity: Order quantity
            close_position: Whether to close the entire position
        
        Returns:
            dict: Order response from Binance
        """
        try:
            # Format the quantity according to symbol info
            formatted_quantity = self.format_quantity(symbol, quantity)
            formatted_price = self.format_price(symbol, stop_price)
            
            params = {
                "symbol": symbol,
                "side": side,
                "type": "TAKE_PROFIT_MARKET",
                "quantity": formatted_quantity,
                "stopPrice": formatted_price,
                "closePosition": str(close_position).lower(),
                "workingType": "MARK_PRICE",  # Use mark price to avoid premature triggers
                "timeInForce": "GTC"
            }
            
            # Using new_order instead of futures_create_order
            response = self.client.new_order(**params)
            logger.info(f"Created take profit order: {side} {formatted_quantity} {symbol} @ {formatted_price}")
            return response
            
        except Exception as e:
            logger.error(f"Error creating take profit order: {str(e)}")
            return None

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        """
        Cancel an order by its ID
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID to cancel
        
        Returns:
            dict: Cancellation response from Binance
        """
        try:
            response = self.client.cancel_order(
                symbol=symbol,
                orderId=order_id
            )
            logger.info(f"Cancelled order {order_id} for {symbol}")
            return response
            
        except Exception as e:
            # For order not found or already filled/canceled errors, don't treat as real error
            error_msg = str(e)
            if "Unknown order sent" in error_msg or "Order does not exist" in error_msg or "Order not found" in error_msg:
                logger.warning(f"Order {order_id} for {symbol} already filled or cancelled")
                return {"status": "CANCELED", "orderId": order_id, "clientOrderId": None, "info": "Order already cancelled/filled"}
            elif "-2011" in error_msg:  # Binance error code for canceled or not found orders
                logger.warning(f"Order {order_id} for {symbol} not found or already canceled (Code: -2011)")
                return {"status": "CANCELED", "orderId": order_id, "clientOrderId": None, "info": "Order not found"}
            else:
                logger.error(f"Error cancelling order {order_id} for {symbol}: {error_msg}")
                return None

    def get_open_orders(self, symbol: str) -> List[dict]:
        """
        Get all open orders for a symbol
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            List[dict]: List of open orders
        """
        try:
            orders = self.client.get_open_orders(symbol=symbol)
            return orders
            
        except Exception as e:
            error_msg = str(e)
            if "orderId is mandatory" in error_msg:
                logger.warning(f"Error getting open orders: {error_msg}. This appears to be a parameterization issue. Returning empty list.")
                return []
            else:
                logger.error(f"Error getting open orders: {error_msg}")
                return []

    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[dict]:
        """
        Get recent trades for a symbol
        
        Args:
            symbol: Trading pair symbol
            limit: Number of trades to fetch (max 1000)
        
        Returns:
            List[dict]: List of recent trades
        """
        try:
            trades = self.client.futures_account_trades(
                symbol=symbol,
                limit=limit
            )
            return trades
            
        except Exception as e:
            logger.error(f"Error getting recent trades: {str(e)}")
            return [] 