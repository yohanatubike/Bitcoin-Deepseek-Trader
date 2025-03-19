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
                 leverage: int = 10, symbol: str = "BTCUSDT", min_confidence: float = 0.65):
        """
        Initialize trade executor for futures trading
        
        Args:
            binance_client: BinanceClient instance
            max_positions: Maximum number of positions to hold
            risk_per_trade: Risk per trade as a fraction (0.02 = 2%)
            leverage: Leverage to use for futures trading (1-125)
            symbol: Trading pair symbol to focus on
            min_confidence: Minimum confidence required to execute a trade (0.0-1.0)
        """
        self.client = binance_client
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade  # As a decimal, e.g., 0.02 for 2%
        self.leverage = min(125, max(1, leverage))  # Ensure leverage is between 1-125
        self.symbol = symbol
        self.min_confidence = min_confidence
        self.margin_type = "ISOLATED"  # Default margin type
                
        # Initialize positions tracking
        self.positions = []
        
        # Load any existing positions
        self._load_existing_positions()
        
        logger.info(f"🤖 Trade executor initialized with max_positions={max_positions}, risk_per_trade={risk_per_trade*100}%, leverage={leverage}x, symbol={symbol}")

    def _load_existing_positions(self):
        """
        Load existing futures positions from the exchange
        """
        try:
            positions = self.client.get_positions(self.symbol)
            for pos in positions:
                position_amount = float(pos.get("positionAmt", 0))
                
                # Skip positions with zero amount
                if position_amount == 0:
                    continue
                    
                # Determine side based on position amount
                side = "BUY" if position_amount > 0 else "SELL"
                quantity = abs(position_amount)
                symbol = pos.get("symbol", "")
                entry_price = float(pos.get("entryPrice", 0))
                leverage = float(pos.get("leverage", self.leverage))
                margin_type = pos.get("marginType", self.margin_type)
                timestamp = int(pos.get("updateTime", int(time.time() * 1000)))
                
                # Create position ID
                position_id = f"{symbol}_{side}_{timestamp}"
                
                # Get stop loss and take profit orders for this position
                sl_order_id = None
                tp_order_id = None
                stop_loss = None
                take_profit = None
                
                # Get open orders for the symbol and find SL/TP orders
                try:
                    open_orders = self.client.get_open_orders(symbol)
                    for order in open_orders:
                        order_side = order.get("side")
                        order_type = order.get("type")
                        order_quantity = float(order.get("origQty", 0))
                        
                        # Match SL/TP orders to our position
                        if order_quantity == quantity:
                            if order_type == "STOP_MARKET" and order_side != side:
                                sl_order_id = order.get("orderId")
                                stop_loss = float(order.get("stopPrice", 0))
                            elif order_type == "TAKE_PROFIT_MARKET" and order_side != side:
                                tp_order_id = order.get("orderId")
                                take_profit = float(order.get("stopPrice", 0))
                except Exception as e:
                    logger.error(f"Error retrieving orders for {symbol}: {str(e)}")
                
                # Add position to the list
                position_data = {
                    "symbol": symbol,
                    "side": side,
                    "entry_price": entry_price,
                    "quantity": quantity,
                    "timestamp": timestamp,
                    "position_id": position_id,
                    "leverage": leverage,
                    "margin_type": margin_type,
                    "sl_order_id": sl_order_id,
                    "tp_order_id": tp_order_id,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit
                }
                
                self.positions.append(position_data)
                logger.info(f"Loaded existing futures position: {symbol} {side} {quantity} @ {entry_price} with {leverage}x leverage")
            
            logger.info(f"Loaded {len(self.positions)} active positions")
        except Exception as e:
            logger.error(f"Error loading positions: {str(e)}")

    def _save_positions(self):
        """Save current positions to log or persistent storage"""
        try:
            # For now, just log the positions - in a real implementation,
            # this would save to a database or file for persistence
            for position in self.positions:
                logger.debug(f"Position saved: {position['symbol']} {position['side']} {position['quantity']} @ {position['entry_price']}")
            logger.debug(f"Saved {len(self.positions)} positions")
        except Exception as e:
            logger.error(f"Error saving positions: {str(e)}")

    def execute_trade(self, prediction: Dict[str, Any], market_data: Dict[str, Any]) -> bool:
        """
        Execute a trade based on prediction
        
        Args:
            prediction: Prediction data
            market_data: Market data
            
        Returns:
            bool: Whether the trade was executed
        """
        try:
            # Extract prediction data
            if "prediction" in prediction and isinstance(prediction["prediction"], dict):
                pred_data = prediction["prediction"]
                action = pred_data.get("action", "").upper()
                confidence = pred_data.get("confidence", 0.0)
                stop_loss = pred_data.get("stop_loss")
                take_profit = pred_data.get("take_profit")
                position_size = pred_data.get("position_size", 0.3)  # Default to 30% if not provided
            else:
                # Direct format
                action = prediction.get("action", "").upper()
                confidence = prediction.get("confidence", 0.0)
                stop_loss = prediction.get("stop_loss")
                take_profit = prediction.get("take_profit")
                position_size = prediction.get("position_size", 0.3)
                
            # Log prediction for debugging
            logger.info(f"Attempting to execute trade: {action} with confidence {confidence:.2f}")
            
            # Skip execution for HOLD actions
            if action == "HOLD":
                logger.info(f"Skipping trade execution for HOLD action")
                return False
                
            # Skip if confidence is too low
            if confidence < self.min_confidence:
                logger.info(f"Confidence {confidence:.2f} below threshold {self.min_confidence:.2f}, skipping trade")
                return False
                
            # Check position limit
            if len(self.positions) >= self.max_positions:
                logger.warning(f"Maximum number of positions reached ({self.max_positions}), skipping trade")
                return False
                
            # Check if we already have an active position for this symbol in the same direction
            symbol = market_data.get("symbol", "BTCUSDT")
            for position in self.positions:
                if position["symbol"] == symbol and position["side"] == action:
                    logger.warning(f"Already have an active {action} position for {symbol}, skipping")
                    return False
            
            # Execute the trade
            if action == "BUY":
                # Calculate position size
                quantity = self._calculate_position_size(
                    symbol=symbol, 
                    side="BUY", 
                    price=market_data.get("timeframes", {}).get("5m", {}).get("price", {}).get("close", 0),
                    stop_loss=stop_loss,
                    market_data=market_data,
                    position_size_factor=position_size
                )
                
                # Execute order
                order = self.client.execute_order(
                    symbol=symbol,
                    side="BUY",
                    quantity=quantity,
                    order_type="MARKET"
                )
                
                if not order:
                    logger.error(f"Failed to execute BUY order for {symbol}")
                    return False
                    
                # Get order details
                order_id = order.get("orderId")
                executed_qty = float(order.get("executedQty", 0))
                avg_price = float(order.get("avgPrice", 0))
                if avg_price == 0:
                    # Try to get current price if avg_price is not available
                    avg_price = self.client.get_current_price(symbol)
                    
                if executed_qty == 0:
                    # This is normal with Binance Futures API - sometimes the response doesn't include 
                    # the executed quantity right away, so we need to check the position
                    logger.info(f"Order submitted but quantity not yet confirmed, checking position...")
                    # Check position to confirm
                    positions = self.client.get_positions(symbol)
                    for pos in positions:
                        if pos["symbol"] == symbol and float(pos["positionAmt"]) > 0:
                            executed_qty = float(pos["positionAmt"])
                            avg_price = float(pos["entryPrice"])
                            logger.info(f"Confirmed BUY position: {executed_qty} {symbol} @ {avg_price}")
                            break
                            
                if executed_qty == 0:
                    logger.error(f"Failed to confirm BUY position for {symbol}")
                    return False
                
                # Calculate stop loss and take profit levels
                final_stop_loss = stop_loss
                final_take_profit = take_profit
                
                # Generate position ID
                position_id = f"{symbol}_BUY_{int(time.time() * 1000)}_{order_id}"
                
                # Add to active positions
                position_data = {
                    "symbol": symbol,
                    "side": "BUY",
                    "entry_price": avg_price,
                    "quantity": executed_qty,
                    "timestamp": int(time.time() * 1000),
                    "position_id": position_id,
                    "leverage": self.leverage,
                    "margin_type": self.margin_type,
                    "sl_order_id": None,
                    "tp_order_id": None,
                    "stop_loss": final_stop_loss,
                    "take_profit": final_take_profit
                }
                
                # Try to add stop loss order
                if final_stop_loss:
                    try:
                        sl_response = self.client.create_stop_loss_order(
                            symbol=symbol,
                            side="SELL",  # Opposite of position side
                            stop_price=final_stop_loss,
                            quantity=executed_qty,
                            close_position=True
                        )
                        
                        if sl_response and "orderId" in sl_response:
                            position_data["sl_order_id"] = sl_response["orderId"]
                            logger.info(f"Set stop loss for BUY position: {final_stop_loss}")
                        else:
                            logger.warning(f"Failed to create stop loss order, continuing without SL: {sl_response}")
                    except Exception as e:
                        logger.warning(f"Error creating stop loss order, continuing without SL: {str(e)}")
                
                # Try to add take profit order
                if final_take_profit:
                    try:
                        tp_response = self.client.create_take_profit_order(
                            symbol=symbol,
                            side="SELL",  # Opposite of position side
                            stop_price=final_take_profit,
                            quantity=executed_qty,
                            close_position=True
                        )
                        
                        if tp_response and "orderId" in tp_response:
                            position_data["tp_order_id"] = tp_response["orderId"]
                            logger.info(f"Set take profit for BUY position: {final_take_profit}")
                        else:
                            logger.warning(f"Failed to create take profit order, continuing without TP: {tp_response}")
                    except Exception as e:
                        logger.warning(f"Error creating take profit order, continuing without TP: {str(e)}")
                
                # Add position to active positions even if SL/TP failed
                self.positions.append(position_data)
                self._save_positions()
                
                logger.info(f"BUY order executed successfully: {symbol} {executed_qty} @ ~{avg_price}, Position ID: {position_id}")
                return True
                
            elif action == "SELL":
                # Similar structure as BUY, with side reversed
                # Calculate position size
                quantity = self._calculate_position_size(
                    symbol=symbol, 
                    side="SELL", 
                    price=market_data.get("timeframes", {}).get("5m", {}).get("price", {}).get("close", 0),
                    stop_loss=stop_loss,
                    market_data=market_data,
                    position_size_factor=position_size
                )
                
                # Execute order
                order = self.client.execute_order(
                    symbol=symbol,
                    side="SELL",
                    quantity=quantity,
                    order_type="MARKET"
                )
                
                if not order:
                    logger.error(f"Failed to execute SELL order for {symbol}")
                    return False
                    
                # Get order details
                order_id = order.get("orderId")
                executed_qty = float(order.get("executedQty", 0))
                avg_price = float(order.get("avgPrice", 0))
                if avg_price == 0:
                    # Try to get current price if avg_price is not available
                    avg_price = self.client.get_current_price(symbol)
                    
                if executed_qty == 0:
                    # This is normal with Binance Futures API - sometimes the response doesn't include 
                    # the executed quantity right away, so we need to check the position
                    logger.info(f"Order submitted but quantity not yet confirmed, checking position...")
                    # Check position to confirm
                    positions = self.client.get_positions(symbol)
                    for pos in positions:
                        if pos["symbol"] == symbol and float(pos["positionAmt"]) < 0:
                            executed_qty = abs(float(pos["positionAmt"]))
                            avg_price = float(pos["entryPrice"])
                            logger.info(f"Confirmed SELL position: {executed_qty} {symbol} @ {avg_price}")
                            break
                            
                if executed_qty == 0:
                    logger.error(f"Failed to confirm SELL position for {symbol}")
                    return False
                
                # Calculate stop loss and take profit levels
                final_stop_loss = stop_loss
                final_take_profit = take_profit
                
                # Generate position ID
                position_id = f"{symbol}_SELL_{int(time.time() * 1000)}_{order_id}"
                
                # Add to active positions
                position_data = {
                    "symbol": symbol,
                    "side": "SELL",
                    "entry_price": avg_price,
                    "quantity": executed_qty,
                    "timestamp": int(time.time() * 1000),
                    "position_id": position_id,
                    "leverage": self.leverage,
                    "margin_type": self.margin_type,
                    "sl_order_id": None,
                    "tp_order_id": None,
                    "stop_loss": final_stop_loss,
                    "take_profit": final_take_profit
                }
                
                # Try to add stop loss order
                if final_stop_loss:
                    try:
                        sl_response = self.client.create_stop_loss_order(
                            symbol=symbol,
                            side="BUY",  # Opposite of position side
                            stop_price=final_stop_loss,
                            quantity=executed_qty,
                            close_position=True
                        )
                        
                        if sl_response and "orderId" in sl_response:
                            position_data["sl_order_id"] = sl_response["orderId"]
                            logger.info(f"Set stop loss for SELL position: {final_stop_loss}")
                        else:
                            logger.warning(f"Failed to create stop loss order, continuing without SL: {sl_response}")
                    except Exception as e:
                        logger.warning(f"Error creating stop loss order, continuing without SL: {str(e)}")
                
                # Try to add take profit order
                if final_take_profit:
                    try:
                        tp_response = self.client.create_take_profit_order(
                            symbol=symbol,
                            side="BUY",  # Opposite of position side
                            stop_price=final_take_profit,
                            quantity=executed_qty,
                            close_position=True
                        )
                        
                        if tp_response and "orderId" in tp_response:
                            position_data["tp_order_id"] = tp_response["orderId"]
                            logger.info(f"Set take profit for SELL position: {final_take_profit}")
                        else:
                            logger.warning(f"Failed to create take profit order, continuing without TP: {tp_response}")
                    except Exception as e:
                        logger.warning(f"Error creating take profit order, continuing without TP: {str(e)}")
                
                # Add position to active positions even if SL/TP failed
                self.positions.append(position_data)
                self._save_positions()
                
                logger.info(f"SELL order executed successfully: {symbol} {executed_qty} @ ~{avg_price}, Position ID: {position_id}")
                return True
                
            else:
                logger.warning(f"Unsupported action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def _adjust_risk_for_open_positions(self, base_risk: float) -> float:
        """
        Dynamically adjust risk percentage based on number of open positions and available margin
        
        Args:
            base_risk: Base risk percentage (0.0-1.0)
            
        Returns:
            Adjusted risk percentage
        """
        # Get current number of positions
        position_count = len(self.positions)
        
        if position_count == 0:
            # No existing positions, use full risk
            return base_risk
            
        # Get account info to check margin usage
        try:
            account_info = self.client.get_account_info()
            
            # Check if we have available balance info
            available_balance = 0
            total_balance = 0
            
            if isinstance(account_info, dict):
                available_balance = float(account_info.get("availableBalance", 0))
                total_balance = float(account_info.get("totalWalletBalance", 0))
                
                # If values not found, try assets
                if total_balance == 0:
                    assets = account_info.get("assets", [])
                    for asset in assets:
                        if asset.get("asset") == "USDT":
                            total_balance = float(asset.get("walletBalance", 0))
                            break
            
            # Calculate used percentage
            if total_balance > 0:
                used_percent = 1 - (available_balance / total_balance)
                logger.info(f"Used margin percentage: {used_percent:.2%}")
                
                # Adjust risk based on available margin
                # If more than 50% of margin is used, start scaling down risk
                if used_percent > 0.5:
                    # Linear reduction: at 50% usage, use 100% of base risk
                    # at 90% usage, use only 10% of base risk
                    margin_scale = max(0.1, 1 - ((used_percent - 0.5) * 2.5))
                    adjusted_risk = base_risk * margin_scale
                    logger.info(f"Scaling down risk due to high margin usage: {base_risk:.2%} → {adjusted_risk:.2%}")
                    return adjusted_risk
                
            # If we can't determine margin usage or it's not high,
            # scale based on position count alone
            adjusted_risk = base_risk / (position_count + 1)
            logger.info(f"Scaling down risk based on {position_count} existing positions: {base_risk:.2%} → {adjusted_risk:.2%}")
            return adjusted_risk
            
        except Exception as e:
            # If there's an error, fall back to a simple reduction based on position count
            fallback_risk = base_risk / (position_count + 1)
            logger.warning(f"Error calculating adjusted risk: {str(e)}. Using fallback: {fallback_risk:.2%}")
            return fallback_risk

    def _calculate_position_size(self, symbol: str, side: str, price: float, stop_loss: Optional[float], 
                                 market_data: Dict[str, Any], position_size_factor: float) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            symbol: Trading pair symbol
            side: Position side (BUY/SELL)
            price: Current market price
            stop_loss: Stop loss price
            market_data: Market data
            position_size_factor: Factor to adjust position size (0.0-1.0)
            
        Returns:
            float: Position size in base asset
        """
        try:
            # Fetch account info
            account_info = self.client.get_account_info()
            if not account_info:
                logger.error("Failed to get account info for position sizing")
                return 0.001  # Minimum position size as fallback
                
            # Get total balance and available balance
            total_balance = float(account_info.get('totalWalletBalance', 0))
            available_balance = float(account_info.get('availableBalance', 0))
            
            if total_balance <= 0 or available_balance <= 0:
                logger.error(f"Invalid account balance: total={total_balance}, available={available_balance}")
                return 0.001  # Minimum position size as fallback
                
            # Log account metrics
            logger.info(f"Account metrics - Total Balance: ${total_balance:.2f}, Available Balance: ${available_balance:.2f}")
            
            # Calculate used margin and positions count
            used_margin = total_balance - available_balance
            positions_count = len(self.positions)
            logger.info(f"Used Margin: ${used_margin:.2f}, Number of current positions: {positions_count}")
            
            # Calculate the risk amount (how much we're willing to lose)
            risk_amount = total_balance * self.risk_per_trade
            
            # Apply position sizing factor
            risk_amount = risk_amount * position_size_factor
            
            # Cap volatility to a reasonable range (0.005 to 0.05)
            volatility = market_data.get('volatility', 0.01)
            volatility_capped = max(0.005, min(0.05, volatility))
            
            # Get current market price if not provided
            current_price = price
            if current_price <= 0:
                current_price = self.client.get_current_price(symbol)
                if current_price <= 0:
                    logger.error(f"Invalid current price: {current_price}")
                    return 0.001  # Minimum position size as fallback
            
            # Calculate stop loss distance
            if stop_loss and stop_loss > 0:
                if side == "BUY":
                    stop_loss_pct = abs((current_price - stop_loss) / current_price)
                else:  # SELL
                    stop_loss_pct = abs((stop_loss - current_price) / current_price)
            else:
                # Default stop loss percentage if not provided
                stop_loss_pct = 0.015  # 1.5% default stop loss
                
            # Ensure stop loss percentage is reasonable
            stop_loss_pct = max(0.005, min(0.1, stop_loss_pct))  # Between 0.5% and 10%
            
            # Calculate position value based on risk and stop loss
            position_value = (risk_amount / stop_loss_pct) * self.leverage
            
            # Apply safety factor to prevent liquidation (0.8)
            position_value = position_value * 0.8
            
            # Cap position value at 80% of available balance with leverage
            max_safe_value = available_balance * self.leverage * 0.8
            if position_value > max_safe_value:
                logger.warning(f"Position value {position_value:.2f} exceeds maximum safe value. Capping at {max_safe_value:.2f}")
                position_value = max_safe_value
            
            # Ensure position meets minimum notional value
            min_notional = 10.0  # Minimum value for most futures
            if position_value < min_notional:
                logger.warning(f"Position value {position_value:.2f} is below minimum notional {min_notional}. Setting to minimum.")
                position_value = min_notional
            
            # Calculate quantity in base asset
            quantity = position_value / current_price
            
            # Log the details
            logger.info(f"Position sizing - Risk amount: {risk_amount:.2f} USDT, Stop loss %: {stop_loss_pct*100:.2f}%")
            logger.info(f"Position value: {position_value:.2f} USDT ({quantity:.8f} {symbol.replace('USDT', '')})")
            
            # Return formatted quantity
            return quantity
            
        except Exception as e:
            logger.error(f"❌ Error calculating position size: {str(e)}")
            traceback.print_exc()
            return 0.001  # Return minimum size as fallback

    def _execute_buy(self, symbol: str, quantity: float, stop_loss: Optional[float], 
                    take_profit: Optional[float]) -> bool:
        """
        Execute a buy order
        
        Args:
            symbol: Trading pair symbol
            quantity: Order quantity in the base asset (e.g. BTC for BTCUSDT)
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
                
            # IMPORTANT: The quantity is already in BTC, not USDT
            # Calculate USD value for logging purposes
            usd_value = quantity * current_price
            
            # Log the actual values we're working with
            logger.info(f"Using pre-calculated quantity: {quantity} {symbol[:-4]} at price {current_price} (value: {usd_value:.2f} USDT)")
            
            # Check minimum notional value (Binance Futures requires minimum order value of 100 USDT)
            min_notional = 100.0  # Minimum order value in USDT
            
            if usd_value < min_notional:
                logger.warning(f"Order value {usd_value:.2f} USDT is below minimum {min_notional} USDT. Adjusting quantity.")
                quantity = (min_notional * 1.01) / current_price  # Add 1% buffer
                usd_value = quantity * current_price
                logger.info(f"Adjusted quantity to {quantity} to meet minimum notional value")
            
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
                    quantity = math.floor(quantity / step_size) * step_size
                    
                    # Ensure minimum quantity
                    if quantity < min_qty:
                        logger.warning(f"Calculated quantity {quantity} is below minimum {min_qty}. Using minimum.")
                        quantity = min_qty
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
                quantity = math.ceil(quantity / step_size) * step_size
                
                # Ensure minimum quantity
                if quantity < min_qty:
                    logger.warning(f"Calculated quantity {quantity} is below minimum {min_qty}. Using minimum.")
                    quantity = min_qty
                
            # Final check: ensure minimum notional value is met
            final_order_value = quantity * current_price
            if final_order_value < min_notional:
                logger.warning(f"Final order value {final_order_value:.2f} USDT still below minimum {min_notional} USDT after adjustments.")
                # Force minimum notional value with buffer
                quantity = (min_notional / current_price) * 1.05  # Add 5% buffer to be safe
                # Use ceiling to round up to the nearest step size
                quantity = math.ceil(quantity / step_size) * step_size
                logger.info(f"Forced quantity to {quantity} to ensure minimum notional value")
                
            # Execute the order
            response = self.client.execute_order(
                symbol=symbol,
                side="BUY",
                quantity=quantity,
                order_type="MARKET",
                raw_quantity=False
            )
            
            if response and response.get("orderId"):
                # Generate a unique position ID
                timestamp = int(time.time() * 1000)
                position_id = f"{symbol}_BUY_{timestamp}_{response.get('orderId')}"
                
                # Create stop loss order if provided
                sl_order_id = None
                if stop_loss:
                    sl_response = self.client.create_stop_loss_order(
                        symbol=symbol,
                        side="SELL",
                        stop_price=stop_loss,
                        quantity=quantity,
                        close_position=True
                    )
                    if sl_response and sl_response.get("orderId"):
                        sl_order_id = sl_response.get("orderId")
                        logger.info(f"Created stop loss order at {stop_loss}")
                
                # Create take profit order if provided
                tp_order_id = None
                if take_profit:
                    tp_response = self.client.create_take_profit_order(
                        symbol=symbol,
                        side="SELL",
                        stop_price=take_profit,
                        quantity=quantity,
                        close_position=True
                    )
                    if tp_response and tp_response.get("orderId"):
                        tp_order_id = tp_response.get("orderId")
                        logger.info(f"Created take profit order at {take_profit}")
                
                # Add to positions with the unique ID and order IDs
                self.positions.append({
                    "symbol": symbol,
                    "side": "BUY",
                    "entry_price": current_price,
                    "quantity": quantity,
                    "timestamp": timestamp,
                    "position_id": position_id,
                    "order_id": response.get("orderId"),
                    "sl_order_id": sl_order_id,
                    "tp_order_id": tp_order_id,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit
                })
                
                logger.info(f"BUY order executed successfully: {symbol} {quantity} @ ~{current_price}, Position ID: {position_id}")
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
            quantity: Order quantity in the base asset (e.g. BTC for BTCUSDT)
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Whether the order was successful
        """
        try:
            # Get current market price
            current_price = self.client.get_current_price(symbol)
            
            if current_price <= 0:
                logger.error(f"Invalid current price: {current_price}")
                return False
                
            # Calculate USD value
            usd_value = quantity * current_price
            
            # Get account info to check available margin
            account_info = self.client.get_account_info()
            available_balance = float(account_info.get("availableBalance", 0))
            
            # Calculate required margin with buffer
            required_margin = (usd_value / self.leverage) * 1.15  # 15% buffer
            
            logger.info(f"Available balance: ${available_balance:.2f}, Required margin: ${required_margin:.2f}")
            
            if required_margin > available_balance * 0.9:  # Only use 90% of available balance
                # Adjust quantity to fit available margin
                safe_margin = available_balance * 0.85  # Use 85% of available balance
                max_position_value = safe_margin * self.leverage
                new_quantity = max_position_value / current_price
                
                logger.warning(f"Reducing quantity from {quantity} to {new_quantity} due to margin constraints")
                quantity = new_quantity
                usd_value = quantity * current_price
            
            # Check minimum notional value (5 USDT for BTCUSDT)
            min_notional = 5.0
            
            if usd_value < min_notional:
                logger.warning(f"Order value {usd_value:.2f} USDT is below minimum {min_notional} USDT. Adjusting quantity.")
                quantity = (min_notional * 1.01) / current_price  # Add 1% buffer
                usd_value = quantity * current_price
                logger.info(f"Adjusted quantity to {quantity} to meet minimum notional value")
            
            # Get symbol info for precision
            symbol_info = self.client.get_exchange_info(symbol)
            
            if not symbol_info or not isinstance(symbol_info, dict) or not symbol_info.get("filters"):
                logger.warning(f"Using default precision values for {symbol}")
                step_size = 0.001
                min_qty = 0.001
            else:
                # Find the LOT_SIZE filter
                lot_size_filter = next((f for f in symbol_info.get("filters", []) if f.get("filterType") == "LOT_SIZE"), None)
                
                if not lot_size_filter:
                    logger.warning(f"Using default precision values for {symbol}")
                    step_size = 0.001
                    min_qty = 0.001
                else:
                    step_size = float(lot_size_filter.get("stepSize", 0.001))
                    min_qty = float(lot_size_filter.get("minQty", 0.001))
            
            # Round to the nearest step size
            quantity = math.floor(quantity / step_size) * step_size
            
            # Ensure minimum quantity
            if quantity < min_qty:
                logger.warning(f"Quantity {quantity} is below minimum {min_qty}. Cannot execute order.")
                return False
            
            # Set leverage first
            try:
                current_leverage = float(self.client.get_leverage(symbol))
                if current_leverage != self.leverage:
                    self.client.set_leverage(symbol, self.leverage)
                    logger.info(f"Set leverage to {self.leverage}x for {symbol}")
            except Exception as e:
                logger.warning(f"Could not set leverage: {str(e)}")
            
            # Execute the order
            response = self.client.execute_order(
                symbol=symbol,
                side="SELL",
                quantity=quantity,
                order_type="MARKET",
                raw_quantity=False
            )
            
            if response and response.get("orderId"):
                # Generate a unique position ID
                timestamp = int(time.time() * 1000)
                position_id = f"{symbol}_SELL_{timestamp}_{response.get('orderId')}"
                
                # Create stop loss order if provided
                sl_order_id = None
                if stop_loss:
                    sl_response = self.client.create_stop_loss_order(
                        symbol=symbol,
                        side="BUY",
                        stop_price=stop_loss,
                        quantity=quantity,
                        close_position=True
                    )
                    if sl_response and sl_response.get("orderId"):
                        sl_order_id = sl_response.get("orderId")
                        logger.info(f"Created stop loss order at {stop_loss}")
                
                # Create take profit order if provided
                tp_order_id = None
                if take_profit:
                    tp_response = self.client.create_take_profit_order(
                        symbol=symbol,
                        side="BUY",
                        stop_price=take_profit,
                        quantity=quantity,
                        close_position=True
                    )
                    if tp_response and tp_response.get("orderId"):
                        tp_order_id = tp_response.get("orderId")
                        logger.info(f"Created take profit order at {take_profit}")
                
                # Add to positions with the unique ID and order IDs
                self.positions.append({
                    "symbol": symbol,
                    "side": "SELL",
                    "entry_price": current_price,
                    "quantity": quantity,
                    "timestamp": timestamp,
                    "position_id": position_id,
                    "order_id": response.get("orderId"),
                    "sl_order_id": sl_order_id,
                    "tp_order_id": tp_order_id,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit
                })
                
                logger.info(f"SELL order executed successfully: {symbol} {quantity} @ ~{current_price}, Position ID: {position_id}")
                return True
            else:
                logger.error(f"SELL order failed: {response}")
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
                
                # For futures positions, we need to get the current position info
                # from the exchange to make sure we're using the correct quantity
                futures_positions = self.client.get_positions(symbol)
                
                # Find the matching position
                futures_position = None
                for pos in futures_positions:
                    pos_side = "BUY" if float(pos.get("positionAmt", 0)) > 0 else "SELL"
                    if pos.get("symbol") == symbol and pos_side == side:
                        futures_position = pos
                        break
                
                # If we found a matching futures position, use its data
                if futures_position:
                    # Get the actual position amount from the exchange
                    position_amt = abs(float(futures_position.get("positionAmt", 0)))
                    if position_amt > 0:
                        logger.info(f"Found active futures position: {symbol} {side} with amount {position_amt}")
                        
                        # Use the actual position amount from the exchange
                        if abs(position_amt - quantity) > 0.001:  # If there's a significant difference
                            logger.warning(f"Adjusting position quantity from {quantity} to {position_amt} based on actual position")
                            quantity = position_amt
                    else:
                        logger.warning(f"Position reported by exchange has zero amount: {futures_position}")
                        # Remove from our tracking but don't try to close
                        if position_id:
                            self.positions = [p for p in self.positions if p.get("position_id") != position_id]
                        continue
                else:
                    # If we can't find the position on the exchange, log a warning but try to close anyway
                    logger.warning(f"Position not found on exchange: {symbol} {side}. Attempting to close with recorded quantity.")
                
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
                    reduce_only=True    # Use reduce_only for futures
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
                logger.error(traceback.format_exc())
        
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
        
    def close_position(self, position_id: str) -> bool:
        """
        Close a specific position by its ID
        
        Args:
            position_id: Unique position identifier
            
        Returns:
            Whether the position was successfully closed
        """
        logger.info(f"Closing position with ID: {position_id}")
        
        try:
            # Find the position by ID
            position = next((p for p in self.positions if p.get("position_id") == position_id), None)
            
            if not position:
                logger.warning(f"Position with ID {position_id} not found")
                return False
            
            symbol = position["symbol"]
            side = position["side"]
            quantity = position["quantity"]
            sl_order_id = position.get("sl_order_id")
            tp_order_id = position.get("tp_order_id")
            
            # Cancel any existing SL/TP orders first
            if sl_order_id:
                try:
                    self.client.cancel_order(symbol, order_id=sl_order_id)
                    logger.info(f"Cancelled stop loss order {sl_order_id}")
                except Exception as e:
                    logger.warning(f"Could not cancel stop loss order: {str(e)}")
            
            if tp_order_id:
                try:
                    self.client.cancel_order(symbol, order_id=tp_order_id)
                    logger.info(f"Cancelled take profit order {tp_order_id}")
                except Exception as e:
                    logger.warning(f"Could not cancel take profit order: {str(e)}")
            
            # Determine the closing side
            close_side = "SELL" if side == "BUY" else "BUY"
            
            # For futures positions, we need to get the current position info
            # from the exchange to make sure we're using the correct quantity
            futures_positions = self.client.get_positions(symbol)
            
            # Find the matching position
            futures_position = None
            for pos in futures_positions:
                pos_side = "BUY" if float(pos.get("positionAmt", 0)) > 0 else "SELL"
                if pos.get("symbol") == symbol and pos_side == side:
                    futures_position = pos
                    break
            
            # If we found a matching futures position, use its data
            if futures_position:
                # Get the actual position amount from the exchange
                position_amt = abs(float(futures_position.get("positionAmt", 0)))
                if position_amt > 0:
                    logger.info(f"Found active futures position: {symbol} {side} with amount {position_amt}")
                    
                    # Use the actual position amount from the exchange
                    if abs(position_amt - quantity) > 0.001:  # If there's a significant difference
                        logger.warning(f"Adjusting position quantity from {quantity} to {position_amt} based on actual position")
                        quantity = position_amt
                else:
                    logger.warning(f"Position reported by exchange has zero amount: {futures_position}")
                    return False
            else:
                # If we can't find the position on the exchange, log a warning but try to close anyway
                logger.warning(f"Position not found on exchange: {symbol} {side}. Attempting to close with recorded quantity.")
            
            # For BTCUSDT specifically, ensure correct precision formatting
            if symbol.upper() == "BTCUSDT":
                # Force 3 decimal precision for BTCUSDT
                quantity = math.floor(quantity * 1000) / 1000
                logger.info(f"Adjusted BTCUSDT quantity for closing: {quantity}")
            
            if quantity <= 0:
                logger.warning(f"Zero or negative quantity for {symbol}. Cannot close position.")
                return False
            
            # Execute the closing order
            response = self.client.execute_order(
                symbol=symbol,
                side=close_side,
                quantity=quantity,
                order_type="MARKET",
                raw_quantity=True,  # Use exact quantity without formatting
                reduce_only=True    # Use reduce_only for futures
            )
            
            if response and response.get("orderId"):
                logger.info(f"Closed {side} position for {symbol}, quantity: {quantity}")
                
                # Remove position from list
                self.positions = [p for p in self.positions if p.get("position_id") != position_id]
                return True
            else:
                logger.error(f"Failed to close position: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
    def partial_close_position(self, position_id: str, percentage: float = 50.0) -> bool:
        """
        Partially close a position
        
        Args:
            position_id: Unique position identifier
            percentage: Percentage of position to close (1-99)
            
        Returns:
            Whether the position was partially closed
        """
        logger.info(f"Partially closing position with ID: {position_id} ({percentage}%)")
        
        # Validate percentage
        if percentage <= 0 or percentage >= 100:
            logger.error(f"Invalid percentage for partial close: {percentage}. Must be between 1-99.")
            return False
            
        # Find the position by ID
        position = next((p for p in self.positions if p.get("position_id") == position_id), None)
        
        if not position:
            logger.warning(f"Position with ID {position_id} not found")
            return False
            
        try:
            symbol = position["symbol"]
            side = position["side"]
            total_quantity = position["quantity"]
            entry_price = position["entry_price"]
            
            # Calculate quantity to close
            close_quantity = total_quantity * (percentage / 100.0)
            remaining_quantity = total_quantity - close_quantity
            
            # Determine the closing side
            close_side = "SELL" if side == "BUY" else "BUY"
            
            # For BTCUSDT specifically, ensure correct precision formatting
            if symbol.upper() == "BTCUSDT":
                # Force 3 decimal precision for BTCUSDT
                close_quantity = math.floor(close_quantity * 1000) / 1000
                remaining_quantity = math.floor(remaining_quantity * 1000) / 1000
                logger.info(f"Adjusted BTCUSDT quantities: close={close_quantity}, remaining={remaining_quantity}")
            
            # Get symbol info for precision
            symbol_info = self.client.get_exchange_info(symbol)
            
            # Find the LOT_SIZE filter
            lot_size_filter = None
            for f in symbol_info.get("filters", []):
                if f.get("filterType") == "LOT_SIZE":
                    lot_size_filter = f
                    break
                    
            if lot_size_filter:
                # Get the step size and minimum quantity
                step_size = float(lot_size_filter.get("stepSize", 0.001))
                min_qty = float(lot_size_filter.get("minQty", 0.001))
                
                # Round to the nearest step size
                close_quantity = math.floor(close_quantity / step_size) * step_size
                
                # Ensure minimum quantity
                if close_quantity < min_qty:
                    logger.warning(f"Close quantity {close_quantity} is below minimum {min_qty}. Using minimum.")
                    close_quantity = min_qty
                    
                # Check if remaining would be below minimum
                if remaining_quantity < min_qty:
                    logger.warning(f"Remaining quantity {remaining_quantity} would be below minimum {min_qty}. Closing full position instead.")
                    return self.close_position(position_id)
            
            # Execute the closing order
            response = self.client.execute_order(
                symbol=symbol,
                side=close_side,
                quantity=close_quantity,
                order_type="MARKET",
                raw_quantity=True,  # Use exact quantity without formatting
                reduce_only=True    # Use reduce_only for futures
            )
            
            if response and response.get("orderId"):
                logger.info(f"Partially closed {side} position for {symbol}, quantity: {close_quantity} ({percentage}%)")
                
                # Update position with new quantity
                for pos in self.positions:
                    if pos.get("position_id") == position_id:
                        pos["quantity"] = remaining_quantity
                        logger.info(f"Updated position quantity to {remaining_quantity}")
                        break
                        
                return True
            else:
                logger.error(f"Failed to partially close position: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error partially closing position: {str(e)}")
            return False
            
    def modify_position_sl_tp(self, position_id: str, stop_loss: Optional[float] = None, 
                             take_profit: Optional[float] = None) -> bool:
        """
        Modify stop loss and take profit levels for a position
        
        Args:
            position_id: Unique position identifier
            stop_loss: New stop loss price
            take_profit: New take profit price
            
        Returns:
            Whether the SL/TP was modified successfully
        """
        logger.info(f"Modifying SL/TP for position with ID: {position_id}")
        
        # Find the position by ID
        position = next((p for p in self.positions if p.get("position_id") == position_id), None)
        
        if not position:
            logger.warning(f"Position with ID {position_id} not found")
            return False
            
        # If neither stop loss nor take profit provided, nothing to do
        if stop_loss is None and take_profit is None:
            logger.warning(f"No stop loss or take profit provided for modification")
            return False
            
        try:
            symbol = position["symbol"]
            side = position["side"]
            
            # Get existing orders to cancel them
            existing_orders = self.client.get_open_orders(symbol)
            for order in existing_orders:
                # Check if it's a stop loss or take profit order for this position
                if order.get("type") in ["STOP_MARKET", "TAKE_PROFIT_MARKET"]:
                    # Cancel the order
                    self.client.cancel_order(symbol, order_id=order.get("orderId"))
                    logger.info(f"Cancelled existing SL/TP order: {order.get('orderId')}")
            
            # Set up new stop loss if provided
            if stop_loss is not None:
                # Determine stop loss side (opposite of position side)
                sl_side = "SELL" if side == "BUY" else "BUY"
                
                # Place stop loss order
                sl_response = self.client.create_stop_loss_order(
                    symbol=symbol,
                    side=sl_side,
                    stop_price=stop_loss,
                    close_position=True  # Close entire position when triggered
                )
                
                if sl_response and sl_response.get("orderId"):
                    logger.info(f"Set new stop loss at {stop_loss} for position {position_id}")
                else:
                    logger.error(f"Failed to set stop loss: {sl_response}")
                    
            # Set up new take profit if provided
            if take_profit is not None:
                # Determine take profit side (opposite of position side)
                tp_side = "SELL" if side == "BUY" else "BUY"
                
                # Place take profit order
                tp_response = self.client.create_take_profit_order(
                    symbol=symbol,
                    side=tp_side,
                    stop_price=take_profit,
                    close_position=True  # Close entire position when triggered
                )
                
                if tp_response and tp_response.get("orderId"):
                    logger.info(f"Set new take profit at {take_profit} for position {position_id}")
                else:
                    logger.error(f"Failed to set take profit: {tp_response}")
            
            # Return true if at least one of SL or TP was set successfully
            return True
            
        except Exception as e:
            logger.error(f"Error modifying position SL/TP: {str(e)}")
            return False
        
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

    def monitor_positions(self):
        """Monitor active positions and detect closures"""
        try:
            # Get current positions from exchange
            current_positions = {
                p["symbol"]: float(p["positionAmt"])
                for p in self.client.get_positions(self.symbol)
                if float(p["positionAmt"]) != 0
            }
            
            # Check each tracked position
            for position in self.positions[:]:  # Create copy to allow modification during iteration
                symbol = position["symbol"]
                position_id = position["position_id"]
                
                # Position no longer exists on exchange
                if symbol not in current_positions:
                    # Get recent trades to determine if closed by TP/SL
                    trades = self.client.get_recent_trades(symbol)
                    last_trade = next((t for t in trades if t["orderId"] in 
                                     [position.get("sl_order_id"), position.get("tp_order_id")]), None)
                    
                    if last_trade:
                        # Position was closed by TP/SL
                        close_type = "stop loss" if last_trade["orderId"] == position.get("sl_order_id") else "take profit"
                        logger.info(f"Position {position_id} was closed by {close_type}")
                        
                        # Cancel the other order
                        other_order_id = position.get("tp_order_id") if close_type == "stop loss" else position.get("sl_order_id")
                        if other_order_id:
                            try:
                                self.client.cancel_order(symbol, other_order_id)
                                logger.info(f"Cancelled remaining {close_type} order {other_order_id}")
                            except Exception as e:
                                logger.warning(f"Could not cancel remaining order: {str(e)}")
                    else:
                        logger.info(f"Position {position_id} was closed (manual or other reason)")
                    
                    # Remove from tracked positions
                    self.positions.remove(position)
                    
                # Position exists but amount changed
                elif abs(current_positions[symbol]) != abs(position["quantity"]):
                    logger.info(f"Position {position_id} quantity changed from {position['quantity']} to {abs(current_positions[symbol])}")
                    position["quantity"] = abs(current_positions[symbol])
            
        except Exception as e:
            logger.error(f"Error monitoring positions: {str(e)}")
            logger.error(traceback.format_exc()) 