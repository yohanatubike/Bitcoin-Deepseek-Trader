"""
Module for executing trades based on predictions
"""

import logging
import time
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class TradeExecutor:
    def __init__(self, binance_client):
        """
        Initialize the trade executor
        
        Args:
            binance_client: BinanceClient instance
        """
        self.binance_client = binance_client
        self.last_trade_time = 0
        self.min_trade_interval = 60  # Minimum seconds between trades
        self.max_concurrent_trades = 3  # Maximum number of concurrent trades
        
        # Track all active positions
        self.active_positions = []
        
        # Single position tracking for backward compatibility
        self.position = {
            "symbol": None,
            "side": None,
            "quantity": 0,
            "entry_price": 0,
            "stop_loss": 0,
            "take_profit": 0,
            "timestamp": 0
        }
    
    def execute_trade(self, symbol: str, action: str, confidence: float, 
                     confidence_threshold: float, stop_loss: Optional[float] = None, 
                     take_profit: Optional[float] = None, volatility: float = 0.01) -> bool:
        """
        Execute a trade based on prediction
        
        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)
            action: Trade action (BUY, SELL, HOLD)
            confidence: Confidence score (0-1)
            confidence_threshold: Minimum confidence required to execute trade
            stop_loss: Stop loss price
            take_profit: Take profit price
            volatility: Volatility measure (WVO) for position sizing
            
        Returns:
            Whether a trade was executed
        """
        logger.info(f"Executing trade: {action} for {symbol} with confidence {confidence}")
        
        # Check if confidence meets threshold
        if confidence < confidence_threshold:
            logger.info(f"Confidence {confidence} below threshold {confidence_threshold}. No trade executed.")
            return False
        
        # Check if enough time has passed since the last trade
        current_time = time.time()
        if current_time - self.last_trade_time < self.min_trade_interval:
            logger.info(f"Minimum trade interval not met. No trade executed.")
            return False
        
        # Check if we've reached the maximum number of concurrent trades
        if len(self.active_positions) >= self.max_concurrent_trades and action != "HOLD":
            logger.warning(f"Maximum concurrent trades ({self.max_concurrent_trades}) reached. Cannot execute new trade.")
            return False
        
        try:
            # Get account information
            account_info = self.binance_client.get_account_info()
            
            # Calculate position size based on volatility
            position_size = self._calculate_position_size(volatility, account_info)
            
            if action == "BUY":
                # Execute BUY order
                result = self._execute_buy(symbol, position_size, stop_loss, take_profit)
                if result:
                    self.last_trade_time = current_time
                    return True
                return False
                
            elif action == "SELL":
                # Execute SELL order
                result = self._execute_sell(symbol, position_size, stop_loss, take_profit)
                if result:
                    self.last_trade_time = current_time
                    return True
                return False
                
            elif action == "HOLD":
                logger.info(f"HOLD signal received. No trade executed.")
                return False
                
            else:
                logger.warning(f"Unknown action: {action}. No trade executed.")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return False
    
    def _calculate_position_size(self, volatility: float, account_info: Dict[str, Any]) -> float:
        """
        Calculate position size based on volatility
        
        Args:
            volatility: Volatility measure (WVO)
            account_info: Account information from Binance
            
        Returns:
            Position size in BTC
        """
        # Get available USDT balance
        usdt_balance = 0
        for balance in account_info.get("balances", []):
            if balance.get("asset") == "USDT":
                usdt_balance = float(balance.get("free", 0))
                break
        
        # If balance is too low, use a small default value for testing
        if usdt_balance < 10:
            usdt_balance = 1000  # Default for testing
            logger.warning(f"Low USDT balance. Using default value: {usdt_balance}")
        
        # Risk 2% of balance per trade, adjusted by volatility
        risk_amount = usdt_balance * 0.02
        
        # Adjust position size based on volatility (1 / (1 + WVO))
        # Higher volatility = smaller position
        volatility_factor = 1 / (1 + volatility)
        position_size_usdt = risk_amount * volatility_factor
        
        # Get current price to convert to BTC
        try:
            ticker = self.binance_client._send_request(
                method="GET",
                endpoint="/v3/ticker/price",
                params={"symbol": "BTCUSDT"}
            )
            
            if ticker and "price" in ticker:
                btc_price = float(ticker["price"])
                position_size_btc = position_size_usdt / btc_price
                
                # Round to 5 decimal places
                position_size_btc = round(position_size_btc, 5)
                
                # Ensure minimum position size of 0.001 BTC
                return max(position_size_btc, 0.001)
            else:
                logger.warning("Could not get BTC price. Using default position size.")
                return 0.001  # Default minimum position
                
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.001  # Default minimum position
    
    def _execute_buy(self, symbol: str, quantity: float, stop_loss: Optional[float], 
                    take_profit: Optional[float]) -> bool:
        """
        Execute a BUY order
        
        Args:
            symbol: Trading pair symbol
            quantity: Order quantity
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Executing BUY order for {symbol}, quantity: {quantity}")
        
        # Format stop loss and take profit prices
        if stop_loss is not None:
            stop_loss = self.binance_client.format_price(symbol, stop_loss)
            
        if take_profit is not None:
            take_profit = self.binance_client.format_price(symbol, take_profit)
        
        # Check if we already have an open position for this symbol with the opposite side
        existing_position = self._find_position_by_symbol_and_side(symbol, "SELL")
        if existing_position:
            logger.info(f"Closing existing SELL position before opening BUY position")
            
            # Close the existing position
            try:
                self.binance_client.execute_order(
                    symbol=symbol,
                    side="BUY",  # To close a SELL position, we BUY
                    quantity=existing_position["quantity"],
                    order_type="MARKET"
                )
                
                # Remove the closed position
                self._remove_position(existing_position)
                logger.info(f"Closed existing SELL position for {symbol}")
            except Exception as e:
                logger.error(f"Error closing existing position: {str(e)}")
                return False
        
        # Execute the BUY order
        try:
            response = self.binance_client.execute_order(
                symbol=symbol,
                side="BUY",
                quantity=quantity,
                order_type="MARKET",
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Create new position
            if response and "executedQty" in response:
                # Get the actual entry price from fills
                entry_price = 0.0
                if "fills" in response and response["fills"]:
                    # Calculate weighted average price from fills
                    total_qty = 0.0
                    total_cost = 0.0
                    for fill in response["fills"]:
                        qty = float(fill["qty"])
                        price = float(fill["price"])
                        total_qty += qty
                        total_cost += qty * price
                    if total_qty > 0:
                        entry_price = total_cost / total_qty
                else:
                    # Fallback to getting current price
                    try:
                        ticker = self.binance_client._send_request(
                            method="GET",
                            endpoint="/v3/ticker/price",
                            params={"symbol": symbol}
                        )
                        if ticker and "price" in ticker:
                            entry_price = float(ticker["price"])
                    except Exception as e:
                        logger.error(f"Error getting entry price: {str(e)}")
                
                new_position = {
                    "symbol": symbol,
                    "side": "BUY",
                    "quantity": float(response.get("executedQty", quantity)),
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "timestamp": int(time.time()),
                    "order_id": response.get("orderId", 0)
                }
                
                # Add to active positions
                self.active_positions.append(new_position)
                
                # Update single position tracker for backward compatibility
                self.position = new_position.copy()
                
                logger.info(f"BUY position opened: {new_position}")
                logger.info(f"Active positions count: {len(self.active_positions)}")
                return True
            else:
                logger.error("Failed to execute BUY order: No execution details received")
                return False
                
        except Exception as e:
            logger.error(f"Error executing BUY order: {str(e)}")
            return False
    
    def _execute_sell(self, symbol: str, quantity: float, stop_loss: Optional[float], 
                     take_profit: Optional[float]) -> bool:
        """
        Execute a SELL order
        
        Args:
            symbol: Trading pair symbol
            quantity: Order quantity
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Executing SELL order for {symbol}, quantity: {quantity}")
        
        # Format stop loss and take profit prices
        if stop_loss is not None:
            stop_loss = self.binance_client.format_price(symbol, stop_loss)
            
        if take_profit is not None:
            take_profit = self.binance_client.format_price(symbol, take_profit)
        
        # Check if we already have an open position for this symbol with the opposite side
        existing_position = self._find_position_by_symbol_and_side(symbol, "BUY")
        if existing_position:
            logger.info(f"Closing existing BUY position before opening SELL position")
            
            # Close the existing position
            try:
                self.binance_client.execute_order(
                    symbol=symbol,
                    side="SELL",  # To close a BUY position, we SELL
                    quantity=existing_position["quantity"],
                    order_type="MARKET"
                )
                
                # Remove the closed position
                self._remove_position(existing_position)
                logger.info(f"Closed existing BUY position for {symbol}")
            except Exception as e:
                logger.error(f"Error closing existing position: {str(e)}")
                return False
        
        # Execute the SELL order
        try:
            response = self.binance_client.execute_order(
                symbol=symbol,
                side="SELL",
                quantity=quantity,
                order_type="MARKET",
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Create new position
            if response and "executedQty" in response:
                # Get the actual entry price from fills
                entry_price = 0.0
                if "fills" in response and response["fills"]:
                    # Calculate weighted average price from fills
                    total_qty = 0.0
                    total_cost = 0.0
                    for fill in response["fills"]:
                        qty = float(fill["qty"])
                        price = float(fill["price"])
                        total_qty += qty
                        total_cost += qty * price
                    if total_qty > 0:
                        entry_price = total_cost / total_qty
                else:
                    # Fallback to getting current price
                    try:
                        ticker = self.binance_client._send_request(
                            method="GET",
                            endpoint="/v3/ticker/price",
                            params={"symbol": symbol}
                        )
                        if ticker and "price" in ticker:
                            entry_price = float(ticker["price"])
                    except Exception as e:
                        logger.error(f"Error getting entry price: {str(e)}")
                
                new_position = {
                    "symbol": symbol,
                    "side": "SELL",
                    "quantity": float(response.get("executedQty", quantity)),
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "timestamp": int(time.time()),
                    "order_id": response.get("orderId", 0)
                }
                
                # Add to active positions
                self.active_positions.append(new_position)
                
                # Update single position tracker for backward compatibility
                self.position = new_position.copy()
                
                logger.info(f"SELL position opened: {new_position}")
                logger.info(f"Active positions count: {len(self.active_positions)}")
                return True
            else:
                logger.error("Failed to execute SELL order: No execution details received")
                return False
                
        except Exception as e:
            logger.error(f"Error executing SELL order: {str(e)}")
            return False
    
    def _find_position_by_symbol_and_side(self, symbol: str, side: str) -> Optional[Dict]:
        """
        Find a position by symbol and side
        
        Args:
            symbol: Trading pair symbol to find
            side: Position side (BUY or SELL) to find
            
        Returns:
            Position dictionary if found, None otherwise
        """
        for position in self.active_positions:
            if position["symbol"] == symbol and position["side"] == side:
                return position
        return None
    
    def _remove_position(self, position: Dict) -> None:
        """
        Remove a position from active positions
        
        Args:
            position: Position to remove
        """
        # Filter out the position
        self.active_positions = [p for p in self.active_positions if 
                                p.get("order_id") != position.get("order_id")]
    
    def close_all_positions(self) -> None:
        """
        Close all active positions
        """
        logger.info(f"Closing all {len(self.active_positions)} active positions")
        
        for position in self.active_positions[:]:  # Create a copy to iterate over
            try:
                symbol = position["symbol"]
                side = position["side"]
                quantity = position["quantity"]
                
                # Determine the closing side
                close_side = "SELL" if side == "BUY" else "BUY"
                
                # Execute the closing order
                self.binance_client.execute_order(
                    symbol=symbol,
                    side=close_side,
                    quantity=quantity,
                    order_type="MARKET"
                )
                
                logger.info(f"Closed {side} position for {symbol}, quantity: {quantity}")
                
                # Remove from active positions
                self._remove_position(position)
                
            except Exception as e:
                logger.error(f"Error closing position: {str(e)}")
    
    def get_position_status(self) -> Dict[str, Any]:
        """
        Get current position status (for backward compatibility)
        
        Returns:
            Current position information
        """
        return self.position.copy()
    
    def get_active_positions(self) -> List[Dict]:
        """
        Get all active positions
        
        Returns:
            List of active positions
        """
        return self.active_positions.copy()
    
    def get_active_positions_count(self) -> int:
        """
        Get the number of active positions
        
        Returns:
            Number of active positions
        """
        return len(self.active_positions)
        
    def calculate_positions_pnl(self) -> Dict[str, Any]:
        """
        Calculate the profit/loss of all active positions
        
        Returns:
            Dictionary with total PnL and PnL by position
        """
        result = {
            "total_pnl_usd": 0.0,
            "positions": []
        }
        
        try:
            # Skip if no active positions
            if not self.active_positions:
                return result
                
            # Get current market prices
            current_prices = {}
            for position in self.active_positions:
                symbol = position["symbol"]
                if symbol not in current_prices:
                    try:
                        ticker = self.binance_client._send_request(
                            method="GET",
                            endpoint="/v3/ticker/price",
                            params={"symbol": symbol}
                        )
                        if ticker and "price" in ticker:
                            current_prices[symbol] = float(ticker["price"])
                    except Exception as e:
                        logger.error(f"Error fetching price for {symbol}: {str(e)}")
                        current_prices[symbol] = 0.0
            
            total_pnl = 0.0
            positions_pnl = []
            
            # Calculate PnL for each position
            for position in self.active_positions:
                symbol = position["symbol"]
                side = position["side"]
                quantity = position["quantity"]
                entry_price = position["entry_price"]
                current_price = current_prices.get(symbol, 0.0)
                
                if current_price > 0 and entry_price > 0:
                    # Calculate PnL based on position side
                    if side == "BUY":
                        pnl = (current_price - entry_price) * quantity
                        pnl_percent = ((current_price - entry_price) / entry_price) * 100
                    else:  # SELL
                        pnl = (entry_price - current_price) * quantity
                        pnl_percent = ((entry_price - current_price) / entry_price) * 100
                        
                    # Add to total PnL
                    total_pnl += pnl
                    
                    # Add position details
                    position_info = {
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "pnl_usd": round(pnl, 2),
                        "pnl_percent": round(pnl_percent, 2)
                    }
                    
                    positions_pnl.append(position_info)
            
            # Round the total PnL
            result["total_pnl_usd"] = round(total_pnl, 2)
            result["positions"] = positions_pnl
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating positions PnL: {str(e)}")
            return result 