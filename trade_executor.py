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
            
            # Clear existing positions list
            self.positions = []
            
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
                    
                    # Get associated orders (SL/TP)
                    orders = self.client.get_open_orders(symbol)
                    sl_order = None
                    tp_order = None
                    
                    for order in orders:
                        if order.get("type") == "STOP_MARKET" and order.get("closePosition", False):
                            sl_order = order
                        elif order.get("type") == "TAKE_PROFIT_MARKET" and order.get("closePosition", False):
                            tp_order = order
                    
                    self.positions.append({
                        "symbol": symbol,
                        "side": side,
                        "entry_price": entry_price,
                        "quantity": amount,
                        "timestamp": int(time.time() * 1000),
                        "position_id": position_id,
                        "leverage": leverage,
                        "margin_type": margin_type,
                        "sl_order_id": sl_order.get("orderId") if sl_order else None,
                        "tp_order_id": tp_order.get("orderId") if tp_order else None,
                        "stop_loss": float(sl_order.get("stopPrice", 0)) if sl_order else None,
                        "take_profit": float(tp_order.get("stopPrice", 0)) if tp_order else None
                    })
                    logger.info(f"📊 Loaded existing futures position: {symbol} {side} {amount} @ {entry_price} with {leverage}x leverage")
            
            # Log loaded positions
            logger.info(f"Loaded {len(self.positions)} active positions")
            
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
        # Skip if action is HOLD
        if action == "HOLD":
            logger.info("No trade execution needed for HOLD action")
            return True
            
        # Skip if confidence is below threshold
        if confidence < confidence_threshold:
            logger.info(f"No trade: confidence={confidence:.2f} < threshold={confidence_threshold:.2f}")
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
            logger.info("Fetching account info for position sizing...")
            account_info = self.client.get_account_info()
            
            # DEBUGGING: Check account info
            if account_info:
                logger.info(f"Account info retrieved, keys: {account_info.keys() if isinstance(account_info, dict) else 'Not a dict'}")
            else:
                logger.warning("Account info is None or empty")
            
            # Adjust risk based on open positions
            original_risk = self.risk_per_trade
            adjusted_risk = self._adjust_risk_for_open_positions(original_risk)
            
            # Temporarily set the adjusted risk
            old_risk = self.risk_per_trade
            self.risk_per_trade = adjusted_risk
            
            # Check if this is a high confidence trade (>0.9 confidence)
            # If so, we'll use a large percentage of our risk amount
            high_confidence = confidence > 0.9
            
            # Calculate position size based on risk management
            logger.info(f"Calculating position size with volatility: {volatility}, high_confidence: {high_confidence}, adjusted_risk: {adjusted_risk:.2f}")
            quantity = self._calculate_position_size(volatility, account_info, high_confidence)
            
            # Restore original risk setting
            self.risk_per_trade = old_risk
            
            if quantity <= 0:
                logger.warning(f"Calculated quantity is too small: {quantity}. Skipping trade.")
                return False
                
            # DEBUGGING: Log the calculated quantity
            logger.info(f"Calculated quantity: {quantity}")
            
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
            error_trace = traceback.format_exc()
            logger.error(f"Error executing trade: {str(e)}")
            logger.debug(f"Error trace: {error_trace}")
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

    def _calculate_position_size(self, volatility: float, account_info: Dict[str, Any], high_confidence: bool = False) -> float:
        """
        Calculate position size based on advanced risk management for futures
        
        Args:
            volatility: Market volatility (ATR or other volatility measure)
            account_info: Account information
            high_confidence: Whether this is a high confidence trade (use larger position)
            
        Returns:
            Position size in base currency
        """
        # DEBUGGING: Log that we've entered the function
        print("DEBUG: Entered _calculate_position_size function")
        logger.debug("DEBUG: Entered _calculate_position_size function")
        
        try:
            # Get account balance
            total_balance = 0
            unrealized_pnl = 0
            available_balance = 0
            margin_balance = 0
            
            # DEBUGGING: Log account info structure
            logger.debug(f"Account info structure: {account_info.keys() if isinstance(account_info, dict) else 'Not a dict'}")
            
            # Get all important account metrics
            if isinstance(account_info, dict):
                # Get total wallet balance
                total_balance = float(account_info.get("totalWalletBalance", 0))
                
                # Get available balance (this is the free margin)
                available_balance = float(account_info.get("availableBalance", 0))
                
                # Get margin balance (equity)
                margin_balance = float(account_info.get("totalMarginBalance", 0))
                
                # Calculate used margin
                used_margin = margin_balance - available_balance
                
                # Get unrealized PnL
                unrealized_pnl = float(account_info.get("totalUnrealizedProfit", 0))
                
                # If we couldn't find the values in the expected fields, try assets
                if total_balance == 0:
                    assets = account_info.get("assets", [])
                    for asset in assets:
                        if asset.get("asset") == "USDT":
                            total_balance = float(asset.get("walletBalance", 0))
                            unrealized_pnl = float(asset.get("unrealizedProfit", 0))
                            break
            
            # Log available margin for better decision making
            logger.info(f"Account metrics - Total Balance: ${total_balance:.2f}, Available Balance: ${available_balance:.2f}")
            logger.info(f"Used Margin: ${used_margin:.2f}, Number of current positions: {len(self.positions)}")
            
            # DEBUGGING: Force log at INFO level
            logger.info(f"POSITION SIZING - Balance: {total_balance}, volatility: {volatility}")
            
            # Get current market price using the new method
            current_price = self.client.get_current_price(self.symbol)
            
            if current_price <= 0:
                logger.error(f"❌ Invalid current price: {current_price}")
                return 0.0
            
            # Calculate risk amount (% of total balance)
            risk_amount = total_balance * self.risk_per_trade
            
            # Calculate stop loss distance as % of price based on volatility
            # Cap the volatility impact to ensure reasonable position sizes
            # Higher volatility = wider stop loss to avoid noise-triggered stops
            volatility_capped = min(volatility, 0.02)  # Cap volatility at 2%
            stop_loss_pct = max(0.005, volatility_capped * 1.5)  # Minimum 0.5% SL distance
            
            # For high confidence trades, we use a much higher percentage of the account
            if high_confidence:
                logger.info("🚀 HIGH CONFIDENCE TRADE - Using larger position size")
                # Use 80% of the risk amount directly
                min_position_pct = 0.8  # 80% of risk amount for high confidence
                safety_factor = 1.0  # No safety reduction for high confidence
            else:
                # Regular trades
                min_position_pct = 0.3  # At least 30% of risk amount
                safety_factor = 0.9  # Slight safety factor for regular trades
            
            # Hard minimum position size - ensure we're using a significant portion of risk amount
            min_position_value = risk_amount * min_position_pct * self.leverage
            
            # Calculate position size based on risk amount and stop loss distance
            position_value_without_leverage = risk_amount / stop_loss_pct
            
            # Apply leverage (with safety factor to prevent liquidation)
            position_value = position_value_without_leverage * self.leverage * safety_factor
            
            # For high confidence trades, ensure we're using a large position
            if high_confidence:
                # Use at least 50% of the risk amount (with leverage)
                position_value = max(position_value, risk_amount * 0.5 * self.leverage)
            
            # Ensure position value meets our minimum threshold
            position_value = max(position_value, min_position_value)
            
            # Additional log to show the impact of changing the safety factor
            logger.info(f"Risk amount: {risk_amount:.2f} USDT, Volatility (capped): {volatility_capped:.4f}")
            logger.info(f"Position value without safety factor would be: {position_value_without_leverage * self.leverage:.2f} USDT")
            logger.info(f"Position value with safety factor ({safety_factor}) is: {position_value:.2f} USDT")
            logger.info(f"Minimum position value: {min_position_value:.2f} USDT")
            if high_confidence:
                logger.info("Using high confidence position sizing")
            
            # For very large positions, cap at 80% of the account balance to prevent overexposure
            max_position_value = total_balance * 0.8 * self.leverage
            if position_value > max_position_value:
                logger.warning(f"Position value {position_value:.2f} exceeds maximum safe value. Capping at {max_position_value:.2f}")
                position_value = max_position_value
            
            # IMPROVED MARGIN CHECK: Ensure we account for fees and potential price movement
            # For futures trading, the initial margin required is approximately position_value / leverage
            required_margin = position_value / self.leverage
            
            # Add margin for fees and potential price impact - use 15% buffer
            required_margin_with_buffer = required_margin * 1.15  # 15% buffer
            
            # Ensure required margin doesn't exceed 90% of available balance to avoid rejections
            if available_balance > 0:
                max_safe_margin = available_balance * 0.9  # Leave 10% of available balance free
                if required_margin_with_buffer > max_safe_margin:
                    # Calculate the safe position value based on available margin
                    safe_position_value = (max_safe_margin / 1.15) * self.leverage
                    
                    old_position_value = position_value
                    position_value = safe_position_value
                    
                    logger.warning(f"Scaling down position due to insufficient margin: Available=${available_balance:.2f}, Required=${required_margin_with_buffer:.2f}")
                    logger.warning(f"Position value reduced from {old_position_value:.2f} to {position_value:.2f}")
            
            # Binance Futures requires minimum order value of 5 USDT for BTCUSDT
            min_notional = 5.0  # Minimum for BTCUSDT is 5 USDT
            if position_value < min_notional:
                logger.warning(f"Position value {position_value:.2f} USDT is less than minimum notional {min_notional} USDT")
                logger.info(f"Increasing position value to meet minimum notional requirement")
                position_value = min_notional * 1.01  # Add 1% buffer
            
            # Calculate quantity in the base asset
            quantity = position_value / current_price
            
            # FORCE Log details for transparency at all levels
            print(f"💰 Risk calculation: Balance={total_balance:.2f} USDT, Risk={self.risk_per_trade*100:.1f}%, Risk amount={risk_amount:.2f} USDT")
            print(f"📏 Position sizing: Volatility={volatility:.4f}, SL%={stop_loss_pct*100:.2f}%, Leverage={self.leverage}x")
            print(f"🔢 Position size: {position_value:.2f} USDT ({quantity:.8f} {self.symbol[:-4]}) at price {current_price:.2f}")
            
            logger.info(f"💰 Risk calculation: Balance={total_balance:.2f} USDT, Risk={self.risk_per_trade*100:.1f}%, Risk amount={risk_amount:.2f} USDT")
            logger.info(f"📏 Position sizing: Volatility={volatility:.4f}, SL%={stop_loss_pct*100:.2f}%, Leverage={self.leverage}x")
            logger.info(f"🔢 Position size: {position_value:.2f} USDT ({quantity:.8f} {self.symbol[:-4]}) at price {current_price:.2f}")
            
            return quantity
            
        except Exception as e:
            # DEBUGGING: Print and log the full exception
            error_trace = traceback.format_exc()
            print(f"❌ Error calculating position size: {str(e)}\n{error_trace}")
            logger.error(f"❌ Error calculating position size: {str(e)}")
            logger.debug(f"Error trace: {error_trace}")
            return 0.0

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