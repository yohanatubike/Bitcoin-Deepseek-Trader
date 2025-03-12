"""
Module for executing trades based on predictions
"""

import logging
import time
import json
import math
from typing import Dict, Any, Optional, List
import traceback

logger = logging.getLogger(__name__)

class TradeExecutor:
    """
    Executes trades based on AI predictions
    """
    
    def __init__(self, binance_client, max_positions: int = 3, risk_per_trade: float = 0.02, 
                 leverage: int = 10, symbol: str = "BTCUSDT"):
        """
        Initialize trade executor for futures trading
        
        Args:
            binance_client: Binance client instance
            max_positions: Maximum number of concurrent positions
            risk_per_trade: Risk per trade as a percentage of account balance (0.02 = 2%)
            leverage: Leverage to use for futures trading (1-125)
            symbol: Trading pair symbol to focus on
        """
        self.client = binance_client
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        self.leverage = min(125, max(1, leverage))  # Ensure leverage is between 1-125
        self.symbol = symbol
                
        # Initialize positions tracking
        self.positions = []
        
        # Load existing positions
        self._load_existing_positions()
        
        logger.info(f"🤖 Trade executor initialized with max_positions={max_positions}, risk_per_trade={risk_per_trade*100}%, leverage={leverage}x, symbol={symbol}")

    def _load_existing_positions(self):
        """
        Load existing futures positions from the exchange
        """
        try:
            # Get positions from the API
            positions = self.client.get_positions(self.symbol)
            for position in positions:
                # Only add positions with non-zero amount
                position_amt = float(position.get("positionAmt", 0))
                if position_amt != 0:
                    symbol = position.get("symbol")
                    side = "BUY" if position_amt > 0 else "SELL"
                    entry_price = float(position.get("entryPrice", 0))
                    amount = abs(position_amt)
                    leverage = float(position.get("leverage", self.leverage))
                    margin_type = position.get("marginType", "ISOLATED")
                    
                    # Get position ID if available
                    position_id = position.get("positionId", None)
                    if not position_id:
                        position_id = f"{symbol}_{side}_{int(time.time() * 1000)}"
                    
                    self.positions.append({
                        "symbol": symbol,
                        "side": side,
                        "entry_price": entry_price,
                        "quantity": amount,
                        "timestamp": int(time.time() * 1000),
                        "position_id": position_id,
                        "leverage": leverage,
                        "margin_type": margin_type
                    })
                    logger.info(f"📊 Loaded existing futures position: {symbol} {side} {amount} @ {entry_price} with {leverage}x leverage")
        
        except Exception as e:
            logger.error(f"❌ Error loading existing positions: {str(e)}")
            traceback.print_exc()

    def execute_trade(self, symbol: str, action: str, confidence: float, 
                     confidence_threshold: float, stop_loss: Optional[float] = None, 
                     take_profit: Optional[float] = None, volatility: float = 0.01) -> bool:
        """
        Execute a trade based on AI prediction
        
        Args:
            symbol: Trading pair symbol
            action: Trade action (BUY, SELL, HOLD)
            confidence: Prediction confidence (0-1)
            confidence_threshold: Minimum confidence to execute trade
            stop_loss: Stop loss price
            take_profit: Take profit price
            volatility: Market volatility for position sizing
            
        Returns:
            Whether the trade was executed
        """
        # Skip if action is HOLD or confidence is below threshold
        if action == "HOLD" or confidence < confidence_threshold:
            logger.info(f"No trade: action={action}, confidence={confidence:.2f} < threshold={confidence_threshold:.2f}")
            return False
            
        # Check if we already have too many positions
        if len(self.positions) >= self.max_positions:
            logger.warning(f"Maximum positions reached ({self.max_positions}). Cannot open new position.")
            return False
            
        # Check if we already have a position for this symbol and side (informational only)
        has_existing = self.has_position(symbol, action)
        existing_count = len(self._find_position_by_symbol_and_side(symbol, action))
        if has_existing:
            # Log that we're adding another position
            logger.info(f"Adding another {action} position for {symbol} (currently have {existing_count}).")
            
        try:
            # Get account information for position sizing
            account_info = self.client.get_account_info()
            
            # Calculate position size based on risk management
            quantity = self._calculate_position_size(volatility, account_info)
            
            if quantity <= 0:
                logger.warning(f"Calculated quantity is too small: {quantity}. Skipping trade.")
                return False
                
            # Execute the trade
            if action == "BUY":
                success = self._execute_buy(symbol, quantity, stop_loss, take_profit)
            elif action == "SELL":
                success = self._execute_sell(symbol, quantity, stop_loss, take_profit)
            else:
                logger.error(f"Unknown action: {action}")
                return False
                
            return success
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return False

    def _calculate_position_size(self, volatility: float, account_info: Dict[str, Any]) -> float:
        """
        Calculate position size based on advanced risk management for futures
        
        Args:
            volatility: Market volatility (ATR or other volatility measure)
            account_info: Account information
            
        Returns:
            Position size in base currency
        """
        try:
            # Get account balance
            total_balance = 0
            unrealized_pnl = 0
            
            assets = account_info.get("assets", [])
            for asset in assets:
                if asset.get("asset") == "USDT":
                    total_balance = float(asset.get("walletBalance", 0))
                    unrealized_pnl = float(asset.get("unrealizedProfit", 0))
                    break
            
            # Get current market price using the new method
            current_price = self.client.get_current_price(self.symbol)
            
            if current_price <= 0:
                logger.error(f"❌ Invalid current price: {current_price}")
                return 0.0
            
            # Calculate risk amount (% of total balance)
            risk_amount = total_balance * self.risk_per_trade
            
            # Calculate stop loss distance as % of price based on volatility
            # Higher volatility = wider stop loss to avoid noise-triggered stops
            stop_loss_pct = max(0.005, volatility * 1.5)  # Minimum 0.5% SL distance
            
            # Calculate position size based on risk amount and stop loss distance
            position_value_without_leverage = risk_amount / stop_loss_pct
            
            # Apply leverage (with safety factor to prevent liquidation)
            safety_factor = 0.5  # Reduce position size for safety
            position_value = position_value_without_leverage * self.leverage * safety_factor
            
            # Binance Futures requires minimum order value of 100 USDT
            min_notional = 100.0
            if position_value < min_notional:
                logger.warning(f"Position value {position_value:.2f} USDT is less than minimum notional {min_notional} USDT")
                logger.info(f"Increasing position value to meet minimum notional requirement")
                position_value = min_notional * 1.01  # Add 1% buffer
            
            # Calculate quantity in the base asset
            quantity = position_value / current_price
            
            # Log details for transparency
            logger.info(f"💰 Risk calculation: Balance={total_balance:.2f} USDT, Risk={self.risk_per_trade*100:.1f}%, Risk amount={risk_amount:.2f} USDT")
            logger.info(f"📏 Position sizing: Volatility={volatility:.4f}, SL%={stop_loss_pct*100:.2f}%, Leverage={self.leverage}x")
            logger.info(f"🔢 Position size: {position_value:.2f} USDT ({quantity:.8f} {self.symbol[:-4]}) at price {current_price:.2f}")
            
            return quantity
            
        except Exception as e:
            logger.error(f"❌ Error calculating position size: {str(e)}")
            return 0.0

    def _execute_buy(self, symbol: str, quantity: float, stop_loss: Optional[float], 
                    take_profit: Optional[float]) -> bool:
        """
        Execute a buy order
        
        Args:
            symbol: Trading pair symbol
            quantity: Order quantity in quote currency (USDT)
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Whether the order was successful
        """
        try:
            # Get current market price using the new method
            current_price = self.client.get_current_price(symbol)
            
            if current_price <= 0:
                logger.error(f"Invalid current price: {current_price}")
                return False
                
            # Convert USDT quantity to asset quantity
            asset_quantity = quantity / current_price
            
            # Check minimum notional value (Binance Futures requires minimum order value of 100 USDT)
            min_notional = 100.0  # Minimum order value in USDT
            order_value = asset_quantity * current_price
            
            if order_value < min_notional:
                logger.warning(f"Order value {order_value:.2f} USDT is below minimum {min_notional} USDT. Adjusting quantity.")
                asset_quantity = (min_notional * 1.01) / current_price  # Add 1% buffer
                logger.info(f"Adjusted quantity to {asset_quantity} to meet minimum notional value")
            
            # Get symbol info for precision
            symbol_info = self.client.get_exchange_info(symbol)
            
            if not symbol_info or not isinstance(symbol_info, dict) or not symbol_info.get("filters"):
                logger.error(f"Symbol info not found or invalid for {symbol}")
                # Try directly accessing symbol filters from the client
                try:
                    # Use default precision as fallback
                    logger.warning(f"Using default precision values for {symbol}")
                    step_size = 0.001  # Standard BTC step size
                    min_qty = 0.001    # Standard BTC minimum quantity
                    
                    # Round down to the nearest step size
                    asset_quantity = math.floor(asset_quantity / step_size) * step_size
                    
                    # Ensure minimum quantity
                    if asset_quantity < min_qty:
                        logger.warning(f"Calculated quantity {asset_quantity} is below minimum {min_qty}. Using minimum.")
                        asset_quantity = min_qty
                except Exception as e:
                    logger.error(f"Error using default precision: {str(e)}")
                    return False
            else:
                # Find the LOT_SIZE filter
                lot_size_filter = None
                for f in symbol_info.get("filters", []):
                    if f.get("filterType") == "LOT_SIZE":
                        lot_size_filter = f
                        break
                
                if not lot_size_filter:
                    logger.error(f"LOT_SIZE filter not found for {symbol}")
                    # Use default values
                    step_size = 0.001
                    min_qty = 0.001
                else:
                    # Get the step size and minimum quantity
                    step_size = float(lot_size_filter.get("stepSize", 0.001))
                    min_qty = float(lot_size_filter.get("minQty", 0.001))
                
                # Round to the nearest step size - use ceil for buy orders to ensure minimum notional
                asset_quantity = math.ceil(asset_quantity / step_size) * step_size
                
                # Ensure minimum quantity
                if asset_quantity < min_qty:
                    logger.warning(f"Calculated quantity {asset_quantity} is below minimum {min_qty}. Using minimum.")
                    asset_quantity = min_qty
                
            # Final check: ensure minimum notional value is met
            final_order_value = asset_quantity * current_price
            if final_order_value < min_notional:
                logger.warning(f"Final order value {final_order_value:.2f} USDT still below minimum {min_notional} USDT after adjustments.")
                # Force minimum notional value with buffer
                asset_quantity = (min_notional / current_price) * 1.05  # Add 5% buffer to be safe
                # Use ceiling to round up to the nearest step size
                asset_quantity = math.ceil(asset_quantity / step_size) * step_size
                logger.info(f"Forced quantity to {asset_quantity} to ensure minimum notional value")
                
            # Execute the order
            logger.info(f"Executing BUY order for {symbol}: {asset_quantity} at ~{current_price} USDT (value: {asset_quantity * current_price:.2f} USDT)")
            
            # For futures, we need to set the leverage first
            if self.leverage != float(self.client.get_leverage(symbol)):
                try:
                    self.client.set_leverage(symbol, self.leverage)
                    logger.info(f"Set leverage to {self.leverage}x for {symbol}")
                except Exception as e:
                    logger.warning(f"Could not set leverage: {str(e)}")
            
            # Execute the order
            response = self.client.execute_order(
                symbol=symbol,
                side="BUY",
                quantity=asset_quantity,
                order_type="MARKET",
                stop_loss=stop_loss,
                take_profit=take_profit,
                raw_quantity=False  # Let binance_client handle formatting
            )
            
            if response and response.get("orderId"):
                # Generate a unique position ID
                timestamp = int(time.time() * 1000)
                position_id = f"{symbol}_BUY_{timestamp}_{response.get('orderId')}"
                
                # Add to positions with the unique ID
                self.positions.append({
                    "symbol": symbol,
                    "side": "BUY",
                    "entry_price": current_price,
                    "quantity": asset_quantity,
                    "timestamp": timestamp,
                    "position_id": position_id,
                    "order_id": response.get("orderId")
                })
                
                logger.info(f"BUY order executed successfully: {symbol} {asset_quantity} @ ~{current_price}, Position ID: {position_id}")
                return True
            else:
                logger.error(f"BUY order failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing BUY order: {str(e)}")
            return False

    def _execute_sell(self, symbol: str, quantity: float, stop_loss: Optional[float], 
                     take_profit: Optional[float]) -> bool:
        """
        Execute a sell order
        
        Args:
            symbol: Trading pair symbol
            quantity: Order quantity in quote currency (USDT)
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Whether the order was successful
        """
        try:
            # For futures, we can directly sell (short)
            if self.leverage != float(self.client.get_leverage(symbol)):
                # Get current market price using the new method
                current_price = self.client.get_current_price(symbol)
                
                if current_price <= 0:
                    logger.error(f"Invalid current price: {current_price}")
                    return False
                    
                # Convert USDT quantity to asset quantity
                asset_quantity = quantity / current_price
                
                # Check minimum notional value (Binance Futures requires minimum order value of 100 USDT)
                min_notional = 100.0  # Minimum order value in USDT
                order_value = asset_quantity * current_price
                
                if order_value < min_notional:
                    logger.warning(f"Order value {order_value:.2f} USDT is below minimum {min_notional} USDT. Adjusting quantity.")
                    asset_quantity = (min_notional * 1.01) / current_price  # Add 1% buffer
                    logger.info(f"Adjusted quantity to {asset_quantity} to meet minimum notional value")
                
                # Get symbol info for precision
                symbol_info = self.client.get_exchange_info(symbol)
                
                if not symbol_info or not isinstance(symbol_info, dict) or not symbol_info.get("filters"):
                    logger.error(f"Symbol info not found or invalid for {symbol}")
                    # Try directly accessing symbol filters from the client
                    try:
                        # Use default precision as fallback
                        logger.warning(f"Using default precision values for {symbol}")
                        step_size = 0.001  # Standard BTC step size
                        min_qty = 0.001    # Standard BTC minimum quantity
                        
                        # Round to the nearest step size using ceiling to ensure min notional
                        asset_quantity = math.ceil(asset_quantity / step_size) * step_size
                        
                        # Ensure minimum quantity
                        if asset_quantity < min_qty:
                            logger.warning(f"Calculated quantity {asset_quantity} is below minimum {min_qty}. Using minimum.")
                            asset_quantity = min_qty
                    except Exception as e:
                        logger.error(f"Error using default precision: {str(e)}")
                        return False
                else:
                    # Find the LOT_SIZE filter
                    lot_size_filter = None
                    for f in symbol_info.get("filters", []):
                        if f.get("filterType") == "LOT_SIZE":
                            lot_size_filter = f
                            break
                    
                    if not lot_size_filter:
                        logger.error(f"LOT_SIZE filter not found for {symbol}")
                        # Use default values
                        step_size = 0.001
                        min_qty = 0.001
                    else:
                        # Get the step size and minimum quantity
                        step_size = float(lot_size_filter.get("stepSize", 0.001))
                        min_qty = float(lot_size_filter.get("minQty", 0.001))
                    
                    # Round to the nearest step size using ceiling to ensure min notional
                    asset_quantity = math.ceil(asset_quantity / step_size) * step_size
                    
                    # Ensure minimum quantity
                    if asset_quantity < min_qty:
                        logger.warning(f"Calculated quantity {asset_quantity} is below minimum {min_qty}. Using minimum.")
                        asset_quantity = min_qty
                
                # Final check: ensure minimum notional value is met
                final_order_value = asset_quantity * current_price
                if final_order_value < min_notional:
                    logger.warning(f"Final order value {final_order_value:.2f} USDT still below minimum {min_notional} USDT after adjustments.")
                    # Force minimum notional value with safety buffer
                    asset_quantity = (min_notional / current_price) * 1.05  # Add 5% buffer to be safe
                    # Round up to nearest step size
                    asset_quantity = math.ceil(asset_quantity / step_size) * step_size
                    logger.info(f"Forced quantity to {asset_quantity} to ensure minimum notional value")
                    
                # Set leverage first
                try:
                    self.client.set_leverage(symbol, self.leverage)
                    logger.info(f"Set leverage to {self.leverage}x for {symbol}")
                except Exception as e:
                    logger.warning(f"Could not set leverage: {str(e)}")
                
                # Execute the order
                logger.info(f"Executing SELL (SHORT) order for {symbol}: {asset_quantity} at ~{current_price} USDT (value: {asset_quantity * current_price:.2f} USDT)")
                
                response = self.client.execute_order(
                    symbol=symbol,
                    side="SELL",
                    quantity=asset_quantity,
                    order_type="MARKET",
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    raw_quantity=False  # Let binance_client handle formatting
                )
                
                if response and response.get("orderId"):
                    # Generate a unique position ID
                    timestamp = int(time.time() * 1000)
                    position_id = f"{symbol}_SELL_{timestamp}_{response.get('orderId')}"
                    
                    # Add to positions with the unique ID
                    self.positions.append({
                        "symbol": symbol,
                        "side": "SELL",
                        "entry_price": current_price,
                        "quantity": asset_quantity,
                        "timestamp": timestamp,
                        "position_id": position_id,
                        "order_id": response.get("orderId")
                    })
                    
                    logger.info(f"SELL (SHORT) order executed successfully: {symbol} {asset_quantity} @ ~{current_price}, Position ID: {position_id}")
                    return True
                else:
                    logger.error(f"SELL (SHORT) order failed: {response}")
                    return False
            else:
                # For futures, we need to check if we have the asset to sell
                logger.warning("SELL signal received but futures market doesn't support shorting. Ignoring.")
                return False
                
        except Exception as e:
            logger.error(f"Error executing SELL order: {str(e)}")
            return False
    
    def _find_position_by_symbol_and_side(self, symbol: str, side: str) -> List[Dict]:
        """
        Find all positions matching a symbol and side
        
        Args:
            symbol: Trading pair symbol to find
            side: Position side (BUY or SELL) to find
            
        Returns:
            List of matching position dictionaries
        """
        matching_positions = []
        for position in self.positions:
            if position["symbol"] == symbol and position["side"] == side:
                matching_positions.append(position)
        return matching_positions
        
    def has_position(self, symbol: str, side: str) -> bool:
        """
        Check if any positions exist for a given symbol and side
        
        Args:
            symbol: Trading pair symbol to check
            side: Position side (BUY or SELL) to check
            
        Returns:
            True if any matching positions exist
        """
        return len(self._find_position_by_symbol_and_side(symbol, side)) > 0
        
    def get_positions_by_symbol(self, symbol: str) -> List[Dict]:
        """
        Get all positions for a specific symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            List of positions for the symbol
        """
        return [pos for pos in self.positions if pos["symbol"] == symbol]
    
    def close_all_positions(self) -> None:
        """
        Close all active positions
        """
        logger.info(f"Closing all {len(self.positions)} active positions")
        success_count = 0
        
        for position in self.positions[:]:  # Create a copy to iterate over
            try:
                symbol = position["symbol"]
                side = position["side"]
                quantity = position["quantity"]
                position_id = position.get("position_id", None)
                
                # Determine the closing side
                close_side = "SELL" if side == "BUY" else "BUY"
                
                # For spot positions, check the actual available balance
                if side == "BUY":
                    # Extract the base asset (e.g., "BTC" from "BTCUSDT")
                    base_asset = symbol[:-4] if symbol.endswith("USDT") else symbol.split("USDT")[0]
                    
                    # Get the actual free balance
                    balance_info = self.client.get_asset_balance(base_asset)
                    free_balance = balance_info.get("free", 0)
                    
                    logger.info(f"Actual {base_asset} free balance: {free_balance}, recorded position quantity: {quantity}")
                    
                    if free_balance <= 0:
                        logger.warning(f"No free balance available for {base_asset}. Cannot close position.")
                        continue
                        
                    # Use the actual free balance instead of the recorded quantity
                    if abs(free_balance - quantity) > 0.001:  # If there's a significant difference
                        logger.warning(f"Adjusting sell quantity from {quantity} to {free_balance} based on actual balance")
                        quantity = free_balance
                
                # For BTCUSDT specifically, ensure correct precision formatting
                if symbol.upper() == "BTCUSDT":
                    # Force 3 decimal precision for BTCUSDT
                    quantity = math.floor(quantity * 1000) / 1000
                    logger.info(f"Adjusted BTCUSDT quantity for closing: {quantity}")
                
                if quantity <= 0:
                    logger.warning(f"Zero or negative quantity for {symbol}. Skipping position closure.")
                    continue
                
                # Execute the closing order
                response = self.client.execute_order(
                    symbol=symbol,
                    side=close_side,
                    quantity=quantity,
                    order_type="MARKET",
                    raw_quantity=True,  # Use exact quantity without formatting
                    reduce_only=True  # Use reduce_only only for futures
                )
                
                if response and response.get("orderId"):
                    logger.info(f"Closed {side} position for {symbol}, quantity: {quantity}")
                    success_count += 1
                    
                    # Remove from positions list using position_id if available, otherwise use index
                    if position_id:
                        self.positions = [p for p in self.positions if p.get("position_id") != position_id]
                    else:
                        # Fallback to old method
                        self.positions.remove(position)
                else:
                    logger.error(f"Failed to close position for {symbol}: {response}")
                
            except Exception as e:
                logger.error(f"Error closing position: {str(e)}")
        
        logger.info(f"Successfully closed {success_count} out of {len(self.positions)} positions")
        
        # If any positions remain after trying to close them all, log a warning
        if self.positions:
            logger.warning(f"{len(self.positions)} positions could not be closed")
            for pos in self.positions:
                logger.warning(f"Remaining position: {pos['symbol']} {pos['side']} {pos['quantity']}")
    
    def get_position_status(self) -> Dict[str, Any]:
        """
        Get current position status (for backward compatibility)
        
        Returns:
            Current position information
        """
        return self.positions[-1] if self.positions else {}
    
    def get_active_positions(self) -> List[Dict]:
        """
        Get all active positions
        
        Returns:
            List of active positions
        """
        return self.positions.copy()
    
    def get_active_positions_count(self) -> int:
        """
        Get the number of active positions
        
        Returns:
            Number of active positions
        """
        return len(self.positions)
        
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
            if not self.positions:
                return result
                
            # Get current market prices
            current_prices = {}
            for position in self.positions:
                symbol = position["symbol"]
                if symbol not in current_prices:
                    try:
                        # Use get_current_price method instead of _send_request
                        current_price = self.client.get_current_price(symbol)
                        if current_price > 0:
                            current_prices[symbol] = current_price
                        else:
                            logger.warning(f"Invalid price (0 or negative) received for {symbol}")
                            current_prices[symbol] = 0.0
                    except Exception as e:
                        logger.error(f"Error fetching price for {symbol}: {str(e)}")
                        current_prices[symbol] = 0.0
            
            total_pnl = 0.0
            positions_pnl = []
            
            # Calculate PnL for each position
            for position in self.positions:
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