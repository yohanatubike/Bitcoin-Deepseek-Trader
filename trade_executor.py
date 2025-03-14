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
        # DEBUGGING: Log that execute_trade was called
        print(f"DEBUG: Execute trade called: action={action}, confidence={confidence:.2f}, threshold={confidence_threshold:.2f}, volatility={volatility}")
        logger.info(f"DEBUG: Execute trade called: action={action}, confidence={confidence:.2f}, threshold={confidence_threshold:.2f}, volatility={volatility}")
        
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
            
            # NEW: Check available margin and adjust position size if needed
            # We want to ensure we have enough margin for this trade
            if available_balance > 0:
                # For futures trading, the margin required is approximately position_value / leverage
                required_margin = position_value / self.leverage
                
                # Add a 10% buffer to prevent just barely hitting the limit
                required_margin_with_buffer = required_margin * 1.1
                
                # If required margin exceeds available balance, scale down the position
                if required_margin_with_buffer > available_balance:
                    margin_ratio = available_balance / required_margin_with_buffer
                    old_position_value = position_value
                    position_value = position_value * margin_ratio
                    
                    logger.warning(f"Scaling down position due to margin constraints: Available=${available_balance:.2f}, Required=${required_margin_with_buffer:.2f}")
                    logger.warning(f"Position value reduced from {old_position_value:.2f} to {position_value:.2f}")
            
            # Binance Futures requires minimum order value of 100 USDT
            min_notional = 100.0
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
            logger.info(f"Executing BUY order for {symbol}: {quantity} at ~{current_price} USDT (value: {quantity * current_price:.2f} USDT)")
            
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
                quantity=quantity,
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
                    "quantity": quantity,
                    "timestamp": timestamp,
                    "position_id": position_id,
                    "order_id": response.get("orderId")
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
            # For futures, we can directly sell (short)
            if self.leverage != float(self.client.get_leverage(symbol)):
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
                        
                        # Round to the nearest step size using ceiling to ensure min notional
                        quantity = math.ceil(quantity / step_size) * step_size
                        
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
                    
                    # Round to the nearest step size using ceiling to ensure min notional
                    quantity = math.ceil(quantity / step_size) * step_size
                    
                    # Ensure minimum quantity
                    if quantity < min_qty:
                        logger.warning(f"Calculated quantity {quantity} is below minimum {min_qty}. Using minimum.")
                        quantity = min_qty
                
                # Final check: ensure minimum notional value is met
                final_order_value = quantity * current_price
                if final_order_value < min_notional:
                    logger.warning(f"Final order value {final_order_value:.2f} USDT still below minimum {min_notional} USDT after adjustments.")
                    # Force minimum notional value with safety buffer
                    quantity = (min_notional / current_price) * 1.05  # Add 5% buffer to be safe
                    # Round up to nearest step size
                    quantity = math.ceil(quantity / step_size) * step_size
                    logger.info(f"Forced quantity to {quantity} to ensure minimum notional value")
                    
                # Set leverage first
                try:
                    self.client.set_leverage(symbol, self.leverage)
                    logger.info(f"Set leverage to {self.leverage}x for {symbol}")
                except Exception as e:
                    logger.warning(f"Could not set leverage: {str(e)}")
                
                # Execute the order
                logger.info(f"Executing SELL (SHORT) order for {symbol}: {quantity} at ~{current_price} USDT (value: {quantity * current_price:.2f} USDT)")
                
                response = self.client.execute_order(
                    symbol=symbol,
                    side="SELL",
                    quantity=quantity,
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
                        "quantity": quantity,
                        "timestamp": timestamp,
                        "position_id": position_id,
                        "order_id": response.get("orderId")
                    })
                    
                    logger.info(f"SELL (SHORT) order executed successfully: {symbol} {quantity} @ ~{current_price}, Position ID: {position_id}")
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