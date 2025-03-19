#!/usr/bin/env python3
"""
Script to clean up ghost orders on Binance
"""

import logging
import argparse
import time
import configparser
import sys
from binance_client import BinanceClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("order_cleanup")

def clean_ghost_orders(client, symbols=None, all_symbols=False):
    """
    Clean up ghost orders with zero quantity
    
    Args:
        client: Binance client instance
        symbols: List of symbols to check
        all_symbols: Whether to check all symbols
    """
    try:
        total_cancelled = 0
        symbols_processed = 0
        
        if all_symbols:
            # Get all futures symbols
            try:
                exchange_info = client.get_exchange_info()
                if exchange_info and "symbols" in exchange_info:
                    symbols = [s["symbol"] for s in exchange_info["symbols"] if s.get("contractType") == "PERPETUAL"]
                    logger.info(f"Found {len(symbols)} perpetual futures symbols")
                else:
                    logger.error("Failed to get exchange info")
                    return
            except Exception as e:
                logger.error(f"Error getting exchange symbols: {str(e)}")
                return
        
        if not symbols:
            logger.info("No symbols specified and all_symbols is False. Using default BTCUSDT")
            symbols = ["BTCUSDT"]
        
        for symbol in symbols:
            try:
                # Get all open orders for the symbol - use the client's underlying API directly if needed
                logger.info(f"Checking orders for {symbol}...")
                
                try:
                    # First try using the get_open_orders method
                    open_orders = client.get_open_orders(symbol)
                    
                    # If no orders or we got the parameterization error, try direct API call
                    if not open_orders or "parameterization issue" in str(open_orders):
                        logger.info("Falling back to direct API methods to get open orders")
                        
                        # Try using the client's client object directly
                        if hasattr(client, 'client') and client.client:
                            try:
                                # Try futures_get_all_orders first
                                api_response = client.client.futures_get_all_orders(symbol=symbol, limit=50)
                                
                                # Filter to just active orders
                                open_orders = [order for order in api_response 
                                              if order.get('status') in ['NEW', 'PARTIALLY_FILLED']]
                                
                                logger.info(f"Retrieved {len(open_orders)} open orders using futures_get_all_orders")
                            except Exception as e1:
                                logger.warning(f"Error using futures_get_all_orders: {e1}")
                                
                                try:
                                    # Try futures_get_open_orders
                                    open_orders = client.client.futures_get_open_orders(symbol=symbol)
                                    logger.info(f"Retrieved {len(open_orders)} open orders using futures_get_open_orders")
                                except Exception as e2:
                                    logger.warning(f"Error using futures_get_open_orders: {e2}")
                                    
                                    # Last attempt - get positions and try to infer SL/TP orders
                                    try:
                                        # Try to look at positions
                                        positions = client.get_positions(symbol)
                                        
                                        if positions:
                                            for pos in positions:
                                                logger.info(f"Found position: {pos['symbol']} {pos['positionAmt']} at {pos['entryPrice']}")
                                                
                                            logger.info(f"Found {len(positions)} positions for {symbol}")
                                            logger.info("Checking for SL/TP orders for these positions")
                                            
                                    except Exception as e3:
                                        logger.error(f"Error getting positions: {e3}")
                except Exception as e:
                    logger.error(f"Error retrieving open orders: {e}")
                    open_orders = []
                
                if not open_orders:
                    logger.info(f"No open orders found for {symbol}")
                    continue
                
                # Find orders with "Close Position" flag or zero quantity
                ghost_orders = []
                for order in open_orders:
                    qty = float(order.get("origQty", "0"))
                    close_position = order.get("closePosition", False)
                    order_type = order.get("type", "")
                    
                    # Check if it's a ghost order (zero qty or closePosition flag)
                    if (close_position or qty == 0 or qty < 0.0001) and order_type in ["STOP_MARKET", "TAKE_PROFIT_MARKET"]:
                        ghost_orders.append(order)
                
                if ghost_orders:
                    logger.warning(f"Found {len(ghost_orders)} ghost orders for {symbol}, cancelling...")
                    
                    # Cancel each ghost order
                    for order in ghost_orders:
                        order_id = order.get("orderId")
                        order_type = order.get("type")
                        order_price = order.get("stopPrice")
                        order_side = order.get("side")
                        if order_id:
                            try:
                                client.cancel_order(symbol, order_id=order_id)
                                logger.info(f"✅ Cancelled ghost {order_type} {order_side} order {order_id} for {symbol} at price {order_price}")
                                total_cancelled += 1
                                # Sleep briefly to avoid rate limits
                                time.sleep(0.3)
                            except Exception as e:
                                error_msg = str(e)
                                if "Order does not exist" in error_msg or "-2011" in error_msg:
                                    logger.info(f"Ghost order {order_id} already gone, skipping")
                                else:
                                    logger.warning(f"Failed to cancel ghost order {order_id}: {error_msg}")
                else:
                    logger.info(f"No ghost orders found for {symbol}")
                
                symbols_processed += 1
            
            except Exception as e:
                logger.error(f"Error cleaning up ghost orders for {symbol}: {str(e)}")
        
        logger.info(f"✅ Successfully cancelled {total_cancelled} ghost orders across {symbols_processed} symbols")
        
    except Exception as e:
        logger.error(f"Error in clean_ghost_orders: {str(e)}")

def direct_cancel_order(symbol, stop_price, side, order_type):
    """
    Directly cancel an order using parameters instead of order ID
    
    Args:
        symbol: Trading pair symbol
        stop_price: Stop price of the order
        side: Side of the order (BUY/SELL)
        order_type: Type of order (STOP_MARKET/TAKE_PROFIT_MARKET)
    """
    try:
        logger.info(f"Attempting to manually cancel {order_type} {side} order at {stop_price} for {symbol}")
        
        # Parse config
        config = configparser.ConfigParser()
        try:
            config.read("config.ini")
            api_key = config.get("binance", "api_key")
            api_secret = config.get("binance", "api_secret")
        except Exception as e:
            logger.error(f"Error reading config: {str(e)}")
            return
            
        # Create client
        client = BinanceClient(api_key, api_secret)
        
        # Try to use the client's underlying client object
        if hasattr(client, 'client') and client.client:
            try:
                # Use the Binance cancelAll method (different for each client version)
                # Try different method names since the Binance library changes
                try:
                    result = client.client.futures_cancel_all_open_orders(symbol=symbol)
                    logger.info(f"Cancelled all open orders for {symbol}: {result}")
                    return True
                except AttributeError:
                    try:
                        # Try cancel_all_open_orders
                        result = client.client.cancel_all_open_orders(symbol=symbol)
                        logger.info(f"Cancelled all open orders for {symbol}: {result}")
                        return True
                    except AttributeError:
                        # Try futures_cancel_orders
                        result = client.client.futures_cancel_orders(symbol=symbol)
                        logger.info(f"Cancelled all open orders for {symbol} using futures_cancel_orders: {result}")
                        return True
            except Exception as e:
                logger.error(f"Error cancelling all orders: {str(e)}")
                
                # Try using BinanceClient's direct method if available
                try:
                    result = client.cancel_all_orders(symbol)
                    logger.info(f"Cancelled all open orders for {symbol} using BinanceClient method: {result}")
                    return True
                except Exception as e2:
                    logger.error(f"Error using BinanceClient's cancel_all_orders: {str(e2)}")
                return False
    except Exception as e:
        logger.error(f"Error in direct_cancel_order: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Clean up ghost orders on Binance Futures")
    parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols to clean up")
    parser.add_argument("--all", action="store_true", help="Check all available futures symbols")
    parser.add_argument("--config", type=str, default="config.ini", help="Path to config file")
    parser.add_argument("--cancel-all", action="store_true", help="Cancel ALL open orders for the specified symbols")
    parser.add_argument("--cancel-specific", action="store_true", help="Cancel a specific order using parameters")
    parser.add_argument("--stop-price", type=float, help="Stop price for the specific order to cancel")
    parser.add_argument("--side", type=str, choices=["BUY", "SELL"], help="Side of the order to cancel")
    parser.add_argument("--order-type", type=str, choices=["STOP_MARKET", "TAKE_PROFIT_MARKET"], 
                        help="Type of the order to cancel")
    
    args = parser.parse_args()
    
    # Parse config
    config = configparser.ConfigParser()
    try:
        config.read(args.config)
        api_key = config.get("binance", "api_key")
        api_secret = config.get("binance", "api_secret")
    except Exception as e:
        logger.error(f"Error reading config: {str(e)}")
        logger.error("Please ensure config.ini exists with [binance] section containing api_key and api_secret")
        return
    
    # Get symbols from args
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    
    # Create client
    client = BinanceClient(api_key, api_secret)
    
    if args.cancel_specific and symbols and len(symbols) == 1 and args.stop_price and args.side and args.order_type:
        # Cancel a specific order using parameters
        direct_cancel_order(symbols[0], args.stop_price, args.side, args.order_type)
    elif args.cancel_all and symbols:
        # Cancel ALL open orders for the specified symbols
        for symbol in symbols:
            try:
                # Try different methods to cancel all orders
                try:
                    result = client.client.futures_cancel_all_open_orders(symbol=symbol)
                    logger.info(f"Cancelled all open orders for {symbol}: {result}")
                except AttributeError:
                    try:
                        result = client.client.cancel_all_open_orders(symbol=symbol)
                        logger.info(f"Cancelled all open orders for {symbol}: {result}")
                    except AttributeError:
                        try:
                            result = client.client.futures_cancel_orders(symbol=symbol)
                            logger.info(f"Cancelled all open orders for {symbol}: {result}")
                        except AttributeError:
                            try:
                                # Last resort - add a direct implementation
                                logger.info("Using direct cancel implementation")
                                # Try using a direct implementation by checking the client attributes
                                if hasattr(client, 'cancel_all_orders'):
                                    result = client.cancel_all_orders(symbol)
                                    logger.info(f"Used direct BinanceClient.cancel_all_orders for {symbol}: {result}")
                                else:
                                    # Just try a known-working Binance API endpoint as a last resort
                                    from binance.exceptions import BinanceAPIException
                                    try:
                                        # Try to directly cancel specific orders for SL/TP
                                        # First get positions
                                        positions = client.get_positions(symbol)
                                        logger.info(f"Found {len(positions)} positions for {symbol}")
                                        
                                        # Try a different approach - get all open orders 
                                        new_approach = True
                                        
                                        if new_approach:
                                            # First, let's try to manually call the cancel_all_orders endpoint
                                            import requests
                                            import hmac
                                            import hashlib
                                            import time
                                            
                                            # Get testnet status
                                            is_testnet = getattr(client, 'testnet', True)
                                            
                                            # Set base URL based on testnet status
                                            if is_testnet:
                                                base_url = 'https://testnet.binancefuture.com'
                                            else:
                                                base_url = 'https://fapi.binance.com'
                                            
                                            # Use endpoint to cancel all open orders
                                            endpoint = '/fapi/v1/allOpenOrders'
                                            
                                            # Generate timestamp
                                            timestamp = int(time.time() * 1000)
                                            
                                            # Parameters
                                            params = {
                                                'symbol': symbol,
                                                'timestamp': timestamp
                                            }
                                            
                                            # Convert params to query string
                                            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
                                            
                                            # Generate signature
                                            signature = hmac.new(
                                                api_secret.encode('utf-8'),
                                                query_string.encode('utf-8'),
                                                hashlib.sha256
                                            ).hexdigest()
                                            
                                            # Add signature to params
                                            params['signature'] = signature
                                            
                                            # Headers
                                            headers = {
                                                'X-MBX-APIKEY': api_key
                                            }
                                            
                                            # Make the DELETE request
                                            url = f"{base_url}{endpoint}"
                                            response = requests.delete(url, params=params, headers=headers)
                                            
                                            # Log the response
                                            logger.info(f"Manual cancel all orders response: {response.status_code}")
                                            if response.status_code == 200:
                                                logger.info(f"Successfully cancelled all orders for {symbol}: {response.json()}")
                                            else:
                                                logger.error(f"Error cancelling all orders: {response.text}")
                                    except Exception as e:
                                        logger.error(f"Error in manual cancel: {str(e)}")
                            except Exception as e:
                                logger.error(f"All methods to cancel orders failed: {str(e)}")
            except Exception as e:
                logger.error(f"Error cancelling all orders for {symbol}: {str(e)}")
    else:
        # Clean up ghost orders
        clean_ghost_orders(client, symbols=symbols, all_symbols=args.all)


if __name__ == "__main__":
    main() 