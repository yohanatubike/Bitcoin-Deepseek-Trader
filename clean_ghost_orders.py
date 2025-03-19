#!/usr/bin/env python3
"""
Focused script to cancel ghost SL/TP orders by hardcoding the known order details.
This script is specifically targeting the ghost orders that appear in the logs.
"""

import os
import sys
import configparser
import logging
import time
import json
import requests
import hmac
import hashlib
from urllib.parse import urlencode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("ghost_order_cleanup")

# Known TP/SL levels from the error logs
KNOWN_TP_LEVEL = 86000.00
KNOWN_SL_LEVEL = 83800.00

def make_binance_futures_request(api_key, api_secret, method, endpoint, params=None, testnet=True):
    """
    Make a direct request to the Binance Futures API
    
    Args:
        api_key: Binance API key
        api_secret: Binance API secret
        method: HTTP method (GET, POST, DELETE)
        endpoint: API endpoint
        params: Request parameters
        testnet: Whether to use testnet
        
    Returns:
        Response JSON or error message
    """
    # Set base URL based on testnet status
    if testnet:
        base_url = 'https://testnet.binancefuture.com'
    else:
        base_url = 'https://fapi.binance.com'
    
    # Prepare parameters
    if params is None:
        params = {}
    
    # Add timestamp
    params['timestamp'] = int(time.time() * 1000)
    
    # Create query string
    query_string = urlencode(params)
    
    # Generate signature
    signature = hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    # Add signature to query string
    query_string = f"{query_string}&signature={signature}"
    
    # Prepare URL
    url = f"{base_url}{endpoint}?{query_string}"
    
    # Prepare headers
    headers = {
        'X-MBX-APIKEY': api_key
    }
    
    # Make request
    try:
        if method == 'GET':
            response = requests.get(url, headers=headers)
        elif method == 'POST':
            response = requests.post(url, headers=headers)
        elif method == 'DELETE':
            response = requests.delete(url, headers=headers)
        else:
            return {'error': f'Unsupported method: {method}'}
        
        # Process response
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error {response.status_code}: {response.text}")
            return {'error': f'Error {response.status_code}: {response.text}'}
    except Exception as e:
        logger.error(f"Request error: {str(e)}")
        return {'error': str(e)}

def get_open_orders(api_key, api_secret, symbol, testnet=True):
    """
    Get all open orders for a symbol directly from the API
    
    Args:
        api_key: Binance API key
        api_secret: Binance API secret
        symbol: Trading pair symbol
        testnet: Whether to use testnet
        
    Returns:
        List of open orders
    """
    endpoint = '/fapi/v1/openOrders'
    params = {'symbol': symbol}
    
    return make_binance_futures_request(api_key, api_secret, 'GET', endpoint, params, testnet)

def cancel_all_orders(api_key, api_secret, symbol, testnet=True):
    """
    Cancel all open orders for a symbol directly from the API
    
    Args:
        api_key: Binance API key
        api_secret: Binance API secret
        symbol: Trading pair symbol
        testnet: Whether to use testnet
        
    Returns:
        Result of cancellation
    """
    endpoint = '/fapi/v1/allOpenOrders'
    params = {'symbol': symbol}
    
    return make_binance_futures_request(api_key, api_secret, 'DELETE', endpoint, params, testnet)

def cancel_specific_order(api_key, api_secret, symbol, order_id, testnet=True):
    """
    Cancel a specific order by ID
    
    Args:
        api_key: Binance API key
        api_secret: Binance API secret
        symbol: Trading pair symbol
        order_id: Order ID to cancel
        testnet: Whether to use testnet
        
    Returns:
        Result of cancellation
    """
    endpoint = '/fapi/v1/order'
    params = {
        'symbol': symbol,
        'orderId': order_id
    }
    
    return make_binance_futures_request(api_key, api_secret, 'DELETE', endpoint, params, testnet)

def get_positions(api_key, api_secret, testnet=True):
    """
    Get all positions directly from the API
    
    Args:
        api_key: Binance API key
        api_secret: Binance API secret
        testnet: Whether to use testnet
        
    Returns:
        List of positions
    """
    endpoint = '/fapi/v2/positionRisk'
    
    return make_binance_futures_request(api_key, api_secret, 'GET', endpoint, {}, testnet)

def find_ghost_orders(orders):
    """
    Find ghost orders from a list of orders
    
    Args:
        orders: List of orders
        
    Returns:
        List of ghost orders
    """
    ghost_orders = []
    
    for order in orders:
        # Check for specific TP/SL levels that we know are problematic
        if order.get('type') == 'TAKE_PROFIT_MARKET' and float(order.get('stopPrice', 0)) == KNOWN_TP_LEVEL:
            ghost_orders.append(order)
        elif order.get('type') == 'STOP_MARKET' and float(order.get('stopPrice', 0)) == KNOWN_SL_LEVEL:
            ghost_orders.append(order)
            
        # Also check general criteria
        if order.get('closePosition') == True:
            # If it has closePosition=true, check if we have an actual position
            ghost_orders.append(order)
            
    return ghost_orders

def main():
    # Load config
    config = configparser.ConfigParser()
    try:
        config.read('config.ini')
        api_key = config.get('binance', 'api_key')
        api_secret = config.get('binance', 'api_secret')
        testnet = config.getboolean('binance', 'testnet', fallback=True)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        logger.error("Make sure config.ini exists with [binance] section containing api_key and api_secret")
        return
    
    # Define symbols to check
    symbols = ['BTCUSDT']
    
    # Process each symbol
    for symbol in symbols:
        logger.info(f"Processing {symbol}...")
        
        # First, get current positions
        positions = get_positions(api_key, api_secret, testnet)
        
        if isinstance(positions, list):
            # Filter to current symbol
            symbol_positions = [p for p in positions if p.get('symbol') == symbol]
            
            if symbol_positions:
                logger.info(f"Found {len(symbol_positions)} positions for {symbol}:")
                for pos in symbol_positions:
                    pos_amt = float(pos.get('positionAmt', 0))
                    if abs(pos_amt) > 0:
                        logger.info(f"Active position: {pos.get('symbol')} {pos_amt} @ {pos.get('entryPrice')}")
            else:
                logger.info(f"No active positions for {symbol}")
        
        # Get open orders
        orders = get_open_orders(api_key, api_secret, symbol, testnet)
        
        if isinstance(orders, list):
            logger.info(f"Found {len(orders)} open orders for {symbol}")
            
            # Find ghost orders
            ghost_orders = find_ghost_orders(orders)
            
            if ghost_orders:
                logger.info(f"Found {len(ghost_orders)} ghost orders to cancel:")
                
                for order in ghost_orders:
                    order_id = order.get('orderId')
                    order_type = order.get('type')
                    stop_price = order.get('stopPrice')
                    side = order.get('side')
                    
                    logger.info(f"Cancelling {order_type} {side} order {order_id} at {stop_price}")
                    
                    # Cancel the order
                    result = cancel_specific_order(api_key, api_secret, symbol, order_id, testnet)
                    
                    if not result.get('error'):
                        logger.info(f"Success: {result}")
                    else:
                        logger.error(f"Failed: {result}")
                        
                    # Brief delay to avoid rate limits
                    time.sleep(0.5)
            else:
                logger.info("No ghost orders found based on criteria")
                
                # If no ghost orders found by criteria but we still have orders with the known problematic stop prices,
                # try cancelling all orders as a last resort
                has_problem_orders = any(
                    (order.get('type') in ['TAKE_PROFIT_MARKET', 'STOP_MARKET'] and
                    (float(order.get('stopPrice', 0)) == KNOWN_TP_LEVEL or 
                     float(order.get('stopPrice', 0)) == KNOWN_SL_LEVEL))
                    for order in orders
                )
                
                if has_problem_orders:
                    logger.info("Found orders with known problematic stop prices. Cancelling all orders.")
                    result = cancel_all_orders(api_key, api_secret, symbol, testnet)
                    
                    if not result.get('error'):
                        logger.info(f"Successfully cancelled all orders: {result}")
                    else:
                        logger.error(f"Failed to cancel all orders: {result}")
        else:
            logger.error(f"Error getting open orders: {orders}")
            
            # If getting orders failed, try to cancel all orders directly
            logger.info("Attempting to cancel all orders as fallback...")
            result = cancel_all_orders(api_key, api_secret, symbol, testnet)
            
            if not result.get('error'):
                logger.info(f"Successfully cancelled all orders: {result}")
            else:
                logger.error(f"Failed to cancel all orders: {result}")

if __name__ == "__main__":
    main() 