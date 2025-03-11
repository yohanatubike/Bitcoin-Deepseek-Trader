"""
Module for interacting with Binance API to fetch market data and execute trades
"""

import time
import logging
import hmac
import hashlib
import requests
import math
from typing import Dict, Any, List, Optional
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

class BinanceClient:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        Initialize the Binance client
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Whether to use testnet (default: True)
        """
        # Validate API credentials
        if not api_key or api_key == 'YOUR_BINANCE_API_KEY':
            logger.error("Invalid Binance API key. Please update your config.ini file.")
            raise ValueError("Invalid Binance API key")
            
        if not api_secret or api_secret == 'YOUR_BINANCE_API_SECRET':
            logger.error("Invalid Binance API secret. Please update your config.ini file.")
            raise ValueError("Invalid Binance API secret")
        
        # Log masked API key for debugging
        masked_key = self._mask_sensitive_data(api_key)
        masked_secret = self._mask_sensitive_data(api_secret)
        logger.info(f"Using API key: {masked_key}")
        logger.debug(f"Using API secret: {masked_secret}")
        
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        
        # Base URLs
        if testnet:
            self.base_url = "https://testnet.binance.vision/api"
            logger.info("Using Binance Testnet API")
        else:
            self.base_url = "https://api.binance.com/api"
            logger.info("Using Binance Production API")
            
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': self.api_key
        })
        
        # Test connection
        try:
            # Simple ping test to validate connectivity
            response = self._send_request(
                method="GET",
                endpoint="/v3/ping"
            )
            logger.info("Successfully connected to Binance API")
        except Exception as e:
            logger.error(f"Failed to connect to Binance API: {str(e)}")
            # Continue anyway as this is just a test
        
        # Cache for exchange info
        self._exchange_info = None
        self._exchange_info_timestamp = 0
        self._symbol_info_cache = {}
        
    def _mask_sensitive_data(self, data: str) -> str:
        """
        Mask sensitive data for logging
        
        Args:
            data: Sensitive data to mask
            
        Returns:
            Masked data
        """
        if not data or len(data) < 8:
            return "***Invalid***"
            
        # Show first 4 and last 4 characters
        return f"{data[:4]}...{data[-4:]}"
    
    def _get_signature(self, params: Dict) -> str:
        """
        Generate signature for authenticated requests
        
        Args:
            params: Request parameters
            
        Returns:
            HMAC SHA256 signature
        """
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret,
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _send_request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        """
        Send request to Binance API
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            params: Request parameters
            signed: Whether the request requires signature
            
        Returns:
            API response as dictionary
        """
        url = f"{self.base_url}{endpoint}"
        
        if params is None:
            params = {}
            
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            query_string = urlencode(params)
            signature = hmac.new(
                self.api_secret,
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            params['signature'] = signature
            
        try:
            if method == "GET":
                response = self.session.get(url, params=params)
            elif method == "POST":
                response = self.session.post(url, params=params)
            elif method == "DELETE":
                response = self.session.delete(url, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            if response.status_code != 200:
                logger.error(f"API error: {response.status_code} - {response.text}")
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Binance API request error: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response text: {e.response.text}")
            raise
    
    def fetch_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current market data for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Market data dictionary
        """
        result = {
            "symbol": symbol,
            "timestamp": int(time.time() * 1000),
            "timeframes": {
                "5m": {
                    "price": {
                        "klines": []  # Initialize empty klines array
                    },
                    "indicators": {}
                },
                "1h": {
                    "price": {
                        "klines": []  # Initialize empty klines array
                    },
                    "indicators": {}
                }
            },
            "order_book": {},
            "sentiment": {},
            "macro_factors": {}
        }
        
        try:
            # Fetch 5m klines
            klines_5m = self._send_request(
                method="GET",
                endpoint="/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": "5m",
                    "limit": 100  # Need enough data for indicators
                }
            )
            
            if klines_5m and len(klines_5m) > 0:
                # Store the full klines array
                result["timeframes"]["5m"]["price"]["klines"] = [
                    {
                        "timestamp": float(k[0]),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5])
                    } for k in klines_5m
                ]
                
                # Extract the latest candle for easy access
                latest_5m = klines_5m[-1]
                result["timeframes"]["5m"]["price"].update({
                    "open": float(latest_5m[1]),
                    "high": float(latest_5m[2]),
                    "low": float(latest_5m[3]),
                    "close": float(latest_5m[4]),
                    "volume": float(latest_5m[5])
                })
            
            # Fetch 1h klines
            klines_1h = self._send_request(
                method="GET",
                endpoint="/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": "1h",
                    "limit": 200  # Need more data for long-term indicators
                }
            )
            
            if klines_1h and len(klines_1h) > 0:
                # Store the full klines array
                result["timeframes"]["1h"]["price"]["klines"] = [
                    {
                        "timestamp": float(k[0]),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5])
                    } for k in klines_1h
                ]
                
                # Extract the latest candle for easy access
                latest_1h = klines_1h[-1]
                result["timeframes"]["1h"]["price"].update({
                    "open": float(latest_1h[1]),
                    "high": float(latest_1h[2]),
                    "low": float(latest_1h[3]),
                    "close": float(latest_1h[4]),
                    "volume": float(latest_1h[5])
                })
            
            # Fetch order book
            order_book = self._send_request(
                method="GET",
                endpoint="/v3/depth",
                params={
                    "symbol": symbol,
                    "limit": 100
                }
            )
            
            if order_book:
                # Calculate bid-ask spread
                best_bid = float(order_book["bids"][0][0]) if order_book["bids"] else 0
                best_ask = float(order_book["asks"][0][0]) if order_book["asks"] else 0
                
                if best_bid > 0 and best_ask > 0:
                    bid_ask_spread = best_ask - best_bid
                    
                    # Calculate order imbalance (simplified)
                    bid_volume = sum(float(item[1]) for item in order_book["bids"][:20])
                    ask_volume = sum(float(item[1]) for item in order_book["asks"][:20])
                    
                    if bid_volume + ask_volume > 0:
                        order_imbalance = bid_volume / (bid_volume + ask_volume)
                    else:
                        order_imbalance = 0.5
                    
                    result["order_book"] = {
                        "bid_ask_spread": bid_ask_spread,
                        "order_imbalance": order_imbalance,
                        "bids": order_book["bids"][:20],  # Store top 20 levels
                        "asks": order_book["asks"][:20]   # Store top 20 levels
                    }
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            raise
    
    def get_open_algo_orders(self, symbol: str = None) -> List[Dict]:
        """
        Get all open algorithmic orders (stop loss, take profit)
        
        Args:
            symbol: Optional symbol to filter orders
            
        Returns:
            List of open algorithmic orders
        """
        try:
            params = {}
            if symbol:
                params["symbol"] = symbol
                
            orders = self._send_request(
                method="GET",
                endpoint="/v3/openOrders",
                params=params,
                signed=True
            )
            
            # Filter for algorithmic orders (STOP_LOSS, TAKE_PROFIT, etc.)
            algo_orders = [
                order for order in orders
                if order.get("type") in ["STOP_LOSS", "STOP_LOSS_LIMIT", 
                                       "TAKE_PROFIT", "TAKE_PROFIT_LIMIT"]
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
            self._send_request(
                method="DELETE",
                endpoint="/v3/order",
                params={
                    "symbol": symbol,
                    "orderId": order_id
                },
                signed=True
            )
            return True
        except Exception as e:
            logger.error(f"Error canceling algo order: {str(e)}")
            return False
            
    def manage_algo_orders(self, symbol: str) -> bool:
        """
        Manage algorithmic orders to prevent MAX_NUM_ALGO_ORDERS error
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Whether we have room for new algo orders
        """
        try:
            # Get current algo orders
            algo_orders = self.get_open_algo_orders(symbol)
            
            # If we have too many algo orders, cancel the oldest ones
            MAX_ALGO_ORDERS = 5  # Binance's limit is typically 10, we'll stay under it
            if len(algo_orders) >= MAX_ALGO_ORDERS:
                logger.warning(f"Found {len(algo_orders)} algo orders, cleaning up old ones...")
                
                # Sort orders by time, oldest first
                sorted_orders = sorted(algo_orders, key=lambda x: x.get("time", 0))
                
                # Cancel oldest orders until we're under the limit
                orders_to_cancel = len(algo_orders) - MAX_ALGO_ORDERS + 2  # +2 for new SL/TP
                for order in sorted_orders[:orders_to_cancel]:
                    if self.cancel_algo_order(order["symbol"], order["orderId"]):
                        logger.info(f"Cancelled old algo order: {order['orderId']}")
                
                # Verify we now have room
                current_algo_orders = self.get_open_algo_orders(symbol)
                if len(current_algo_orders) < MAX_ALGO_ORDERS:
                    return True
                else:
                    logger.error("Could not free up algo order slots")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error managing algo orders: {str(e)}")
            return False

    def execute_order(self, symbol: str, side: str, quantity: float, 
                      order_type: str = "MARKET", price: float = None,
                      stop_loss: float = None, take_profit: float = None) -> Dict:
        """
        Execute a trade order
        
        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)
            side: Order side (BUY or SELL)
            quantity: Order quantity
            order_type: Order type (MARKET, LIMIT, etc.)
            price: Order price (required for LIMIT orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Order response
        """
        logger.info(f"Executing {side} order for {symbol}, quantity: {quantity}")
        
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
        }
        
        if order_type == "LIMIT" and price:
            # Format price according to symbol's price filter
            formatted_price = self.format_price(symbol, price)
            params["price"] = formatted_price
            params["timeInForce"] = "GTC"
        
        # Execute the main order
        response = self._send_request(
            method="POST",
            endpoint="/v3/order",
            params=params,
            signed=True
        )
        
        # Try to place stop loss and take profit as algo orders first
        try:
            # Check if we can place algo orders
            if (stop_loss or take_profit) and not self.manage_algo_orders(symbol):
                logger.warning("Cannot place algo orders due to existing order limits")
                return response
            
            # If stop loss is provided, try to create a stop loss order
            if stop_loss and response.get("orderId"):
                # Format stop loss price according to symbol's price filter
                formatted_stop_loss = self.format_price(symbol, stop_loss)
                
                stop_loss_params = {
                    "symbol": symbol,
                    "side": "SELL" if side == "BUY" else "BUY",
                    "type": "STOP_LOSS_LIMIT",
                    "quantity": quantity,
                    "price": formatted_stop_loss,  # Execution price
                    "stopPrice": formatted_stop_loss,  # Trigger price
                    "timeInForce": "GTC"
                }
                
                try:
                    self._send_request(
                        method="POST",
                        endpoint="/v3/order",
                        params=stop_loss_params,
                        signed=True
                    )
                except Exception as e:
                    if "MAX_NUM_ALGO_ORDERS" in str(e):
                        logger.warning("Could not place stop loss as algo order due to MAX_NUM_ALGO_ORDERS limit.")
                        # Note: The trading bot will handle stop loss in the main loop
                    else:
                        logger.error(f"Error placing stop loss order: {str(e)}")
            
            # If take profit is provided, try to create a take profit order
            if take_profit and response.get("orderId"):
                # Format take profit price according to symbol's price filter
                formatted_take_profit = self.format_price(symbol, take_profit)
                
                take_profit_params = {
                    "symbol": symbol,
                    "side": "SELL" if side == "BUY" else "BUY",
                    "type": "TAKE_PROFIT_LIMIT",
                    "quantity": quantity,
                    "price": formatted_take_profit,  # Execution price
                    "stopPrice": formatted_take_profit,  # Trigger price
                    "timeInForce": "GTC"
                }
                
                try:
                    self._send_request(
                        method="POST",
                        endpoint="/v3/order",
                        params=take_profit_params,
                        signed=True
                    )
                except Exception as e:
                    if "MAX_NUM_ALGO_ORDERS" in str(e):
                        logger.warning("Could not place take profit as algo order due to MAX_NUM_ALGO_ORDERS limit.")
                        # Note: The trading bot will handle take profit in the main loop
                    else:
                        logger.error(f"Error placing take profit order: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error handling stop loss/take profit orders: {str(e)}")
            # Main order was still executed, so we continue
        
        return response
        
    def get_account_info(self) -> Dict:
        """
        Get account information
        
        Returns:
            Account information
        """
        return self._send_request(
            method="GET",
            endpoint="/v3/account",
            signed=True
        )
        
    def get_account_metrics(self) -> Dict[str, float]:
        """
        Get account financial metrics (balance, equity, margin)
        
        Returns:
            Dictionary with balance, equity, and margin in USD
        """
        logger.info("Fetching account metrics")
        
        result = {
            "balance_usd": 0.0,
            "equity_usd": 0.0,
            "margin_used_usd": 0.0,
            "free_margin_usd": 0.0,
            "margin_level": 0.0
        }
        
        try:
            # Get account information
            account_info = self.get_account_info()
            
            # Get current prices for all assets
            prices = {}
            ticker_prices = self._send_request(
                method="GET",
                endpoint="/v3/ticker/price"
            )
            
            for ticker in ticker_prices:
                if "symbol" in ticker and "price" in ticker:
                    prices[ticker["symbol"]] = float(ticker["price"])
            
            total_balance_usd = 0.0
            total_equity_usd = 0.0
            margin_used_usd = 0.0
            
            # Calculate total balance in USD
            for balance in account_info.get("balances", []):
                asset = balance.get("asset", "")
                free = float(balance.get("free", 0))
                locked = float(balance.get("locked", 0))
                
                total_asset = free + locked
                
                if total_asset > 0:
                    if asset == "USDT":
                        asset_value_usd = total_asset
                    else:
                        # Try to find the price for this asset/USDT pair
                        symbol = f"{asset}USDT"
                        if symbol in prices:
                            asset_value_usd = total_asset * prices[symbol]
                        else:
                            # If no direct pair with USDT, try BTC first then convert to USD
                            symbol_btc = f"{asset}BTC"
                            if symbol_btc in prices and "BTCUSDT" in prices:
                                asset_value_usd = total_asset * prices[symbol_btc] * prices["BTCUSDT"]
                            else:
                                # Skip assets we can't price
                                continue
                    
                    # Add to totals
                    total_balance_usd += asset_value_usd
                    
                    # Free balance is part of equity
                    if free > 0:
                        if asset == "USDT":
                            free_value_usd = free
                        else:
                            symbol = f"{asset}USDT"
                            if symbol in prices:
                                free_value_usd = free * prices[symbol]
                            else:
                                symbol_btc = f"{asset}BTC"
                                if symbol_btc in prices and "BTCUSDT" in prices:
                                    free_value_usd = free * prices[symbol_btc] * prices["BTCUSDT"]
                                else:
                                    free_value_usd = 0
                        
                        total_equity_usd += free_value_usd
                    
                    # Locked balance is considered margin used
                    if locked > 0:
                        if asset == "USDT":
                            locked_value_usd = locked
                        else:
                            symbol = f"{asset}USDT"
                            if symbol in prices:
                                locked_value_usd = locked * prices[symbol]
                            else:
                                symbol_btc = f"{asset}BTC"
                                if symbol_btc in prices and "BTCUSDT" in prices:
                                    locked_value_usd = locked * prices[symbol_btc] * prices["BTCUSDT"]
                                else:
                                    locked_value_usd = 0
                        
                        margin_used_usd += locked_value_usd
            
            # For simplicity in the testnet case, if we couldn't calculate values, use defaults
            if total_balance_usd <= 0:
                logger.warning("Could not calculate account balance. Using default value for testnet.")
                total_balance_usd = 100000.0  # Default testnet balance
            
            if total_equity_usd <= 0:
                total_equity_usd = total_balance_usd - margin_used_usd
            
            free_margin_usd = total_equity_usd - margin_used_usd
            
            # Avoid division by zero
            margin_level = (total_equity_usd / margin_used_usd * 100) if margin_used_usd > 0 else 0
            
            # Format the results with 2 decimal places
            result["balance_usd"] = round(total_balance_usd, 2)
            result["equity_usd"] = round(total_equity_usd, 2)
            result["margin_used_usd"] = round(margin_used_usd, 2)
            result["free_margin_usd"] = round(free_margin_usd, 2)
            result["margin_level"] = round(margin_level, 2)
            
            logger.info(f"Account metrics: Balance=${result['balance_usd']}, Equity=${result['equity_usd']}, Margin Used=${result['margin_used_usd']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching account metrics: {str(e)}")
            
            # Return default values for testnet
            result["balance_usd"] = 100000.0
            result["equity_usd"] = 100000.0
            result["margin_used_usd"] = 0.0
            result["free_margin_usd"] = 100000.0
            result["margin_level"] = 0.0
            
            logger.warning(f"Using default account metrics: {result}")
            return result

    def test_api_permissions(self) -> Dict:
        """
        Test API key permissions and provide diagnostics
        
        Returns:
            Dictionary with test results
        """
        logger.info("Testing API key permissions...")
        
        results = {
            "connectivity": False,
            "read_info": False,
            "trading": False,
            "errors": []
        }
        
        # Test basic connectivity with API key
        try:
            # Test ping endpoint (doesn't require signature)
            self._send_request(
                method="GET", 
                endpoint="/v3/ping"
            )
            results["connectivity"] = True
            logger.info("Basic connectivity test: PASSED")
        except Exception as e:
            results["connectivity"] = False
            error_msg = f"Basic connectivity test failed: {str(e)}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
        
        # Test reading account information (requires signature)
        if results["connectivity"]:
            try:
                # Check if we can get account information
                self._send_request(
                    method="GET",
                    endpoint="/v3/account",
                    signed=True
                )
                results["read_info"] = True
                logger.info("Read account information test: PASSED")
            except Exception as e:
                results["read_info"] = False
                error_msg = f"Read account information test failed: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
        
        # Test trading permissions by getting open orders (requires trading permission)
        if results["connectivity"]:
            try:
                # Check if we can get open orders
                self._send_request(
                    method="GET",
                    endpoint="/v3/openOrders",
                    params={"symbol": "BTCUSDT"},
                    signed=True
                )
                results["trading"] = True
                logger.info("Trading permissions test: PASSED")
            except Exception as e:
                results["trading"] = False
                error_msg = f"Trading permissions test failed: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
        
        # Summarize the results
        if results["connectivity"] and results["read_info"] and results["trading"]:
            logger.info("API key has all required permissions.")
        else:
            logger.warning("API key is missing some permissions. See logs for details.")
            
            # Provide detailed guidance based on the failures
            if not results["connectivity"]:
                logger.error("The API key is invalid or has been revoked. Generate a new API key.")
            elif not results["read_info"]:
                logger.error("The API key does not have 'Read Info' permission. Enable this permission for your API key.")
            elif not results["trading"]:
                logger.error("The API key does not have 'Enable Trading' permission. Enable this permission for your API key.")
        
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
        params = {}
        if symbol:
            params["symbol"] = symbol
            
        try:
            exchange_info = self._send_request(
                method="GET",
                endpoint="/v3/exchangeInfo",
                params=params
            )
            
            # Cache the result
            self._exchange_info = exchange_info
            self._exchange_info_timestamp = current_time
            
            # If symbol was specified, filter for that symbol
            if symbol:
                for sym_info in exchange_info.get("symbols", []):
                    if sym_info.get("symbol") == symbol:
                        self._symbol_info_cache[symbol] = sym_info
                        return sym_info
                return {}
                
            # Cache individual symbols
            for sym_info in exchange_info.get("symbols", []):
                sym = sym_info.get("symbol")
                if sym:
                    self._symbol_info_cache[sym] = sym_info
                    
            return exchange_info
            
        except Exception as e:
            logger.error(f"Error fetching exchange info: {str(e)}")
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
        # Get symbol info from cache or fetch it
        symbol_info = self._symbol_info_cache.get(symbol)
        if not symbol_info:
            symbol_info = self.get_exchange_info(symbol)
            
        # Find price filter
        price_filter = None
        if symbol_info and "filters" in symbol_info:
            for filter_info in symbol_info["filters"]:
                if filter_info.get("filterType") == "PRICE_FILTER":
                    price_filter = filter_info
                    break
                    
        if price_filter:
            # Get tick size
            tick_size = float(price_filter.get("tickSize", 0.01))
            
            # Round to nearest tick size
            if tick_size > 0:
                # Calculate precision from tick size
                precision = int(round(-math.log10(tick_size)))
                return round(round(price / tick_size) * tick_size, precision)
                
        # Default fallback - round to 2 decimal places
        logger.warning(f"Could not find price filter for {symbol}. Using default precision.")
        return round(price, 2) 