#!/usr/bin/env python3
"""
Bitcoin Trading Bot using Binance Testnet and DeepSeek API for trade signals
"""

import os
import json
import time
import logging
import configparser
from typing import Dict, Any
import traceback

# Local modules
from binance_client import BinanceClient
from data_enricher import DataEnricher
from deepseek_api import DeepSeekAPI
from santiment_api import SantimentAPI
from trade_executor import TradeExecutor
from utils import setup_logging

# Setup logging
logger = setup_logging()

class TradingBot:
    """
    Automated trading bot for cryptocurrency markets
    """
    
    def __init__(self, config_path: str = "config.ini"):
        """
        Initialize Futures Trading Bot
        
        Args:
            config_path: Path to configuration file
        """
        # Setup logging
        self._setup_logging()
        
        # Initialize attributes
        self.session_start_time = int(time.time())
        self.session_total_pnl = 0.0
        self.session_trades_count = 0
        self.session_winning_trades = 0
        self.session_losing_trades = 0
        self.last_run_time = 0
        self.running = False
        self.symbol = "BTCUSDT"  # Default, will be overridden from config
        
        # Load configuration
        if not os.path.exists(config_path):
            logger.warning(f"Configuration file {config_path} not found. Creating default configuration.")
            self._create_default_config(config_path)
            
        self.config = self._load_config(config_path)
        
        # Load trading parameters from config
        self.cycle_interval = self.config.getint("trading", "run_interval_seconds", fallback=180)
        self.max_positions = self.config.getint("trading", "max_positions", fallback=3)
        self.risk_per_trade = self.config.getfloat("trading", "risk_per_trade", fallback=0.02)
        self.confidence_threshold = self.config.getfloat("trading", "confidence_threshold", fallback=0.85)
        
        # Futures trading parameters
        self.leverage = self.config.getint("futures", "leverage", fallback=10)
        self.margin_type = self.config.get("futures", "margin_type", fallback="ISOLATED")
        
        # Initialize clients
        self._init_clients()
        
        # Close any active positions on startup if configured
        close_positions_on_start = self.config.getboolean("trading", "close_positions_on_start", fallback=True)
        if close_positions_on_start:
            active_positions = self.trade_executor.get_active_positions()
            if active_positions:
                logger.info(f"🔄 Active positions found at startup: {active_positions}. Closing them.")
                self.trade_executor.close_all_positions()
        
        logger.info(f"🚀 Futures Trading bot initialized for {self.symbol}")
        logger.info(f"💰 Leverage: {self.leverage}x, Margin Type: {self.margin_type}")
        
    def _setup_logging(self):
        """
        Set up logging configuration
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("trading_bot.log")
            ]
        )
        
    def _init_clients(self):
        """
        Initialize API clients for futures trading
        """
        try:
            # Get API keys from config
            binance_api_key = self.config.get("binance", "api_key")
            binance_api_secret = self.config.get("binance", "api_secret")
            deepseek_api_key = self.config.get("deepseek", "api_key")
            
            # Get API URLs and settings
            binance_testnet = self.config.getboolean("binance", "testnet", fallback=True)
            deepseek_api_url = self.config.get("deepseek", "api_url", fallback="https://api.deepseek.com")
            deepseek_test_mode = self.config.getboolean("deepseek", "test_mode", fallback=False)
            
            # Get futures settings
            leverage = self.config.getint("futures", "leverage", fallback=10)
            
            # Get trading symbol from config or default to BTCUSDT
            self.symbol = self.config.get("trading", "symbol", fallback="BTCUSDT")
            
            logger.info(f"🔧 Initializing clients for {self.symbol} with {leverage}x leverage")
            
            # Initialize Binance client (now futures-only)
            self.binance_client = BinanceClient(
                api_key=binance_api_key,
                api_secret=binance_api_secret,
                testnet=binance_testnet,
                leverage=leverage,
                symbol=self.symbol
            )
            
            # Initialize DeepSeek API
            self.deepseek_api = DeepSeekAPI(
                api_key=deepseek_api_key,
                api_url=deepseek_api_url,
                test_mode=deepseek_test_mode
            )
            
            # Santiment API is disabled for now - initialize data enricher without it
            self.data_enricher = DataEnricher(santiment_api=None)
            
            # Initialize trade executor (now futures-only)
            self.trade_executor = TradeExecutor(
                binance_client=self.binance_client,
                max_positions=self.max_positions,
                risk_per_trade=self.risk_per_trade,
                leverage=leverage,
                symbol=self.symbol
            )
            
            # Close existing positions if enabled in config
            close_positions_on_start = self.config.getboolean("trading", "close_positions_on_start", fallback=True)
            if close_positions_on_start:
                # Check if we have any active positions
                active_positions = self.trade_executor.get_active_positions()
                if active_positions:
                    logger.info(f"🔄 Active positions found at startup: {active_positions}. Closing them.")
                    self.trade_executor.close_all_positions()
            
            logger.info(f"✅ Clients initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing clients: {str(e)}")
            traceback.print_exc()
            raise
        
    def _load_config(self, config_path: str) -> configparser.ConfigParser:
        """
        Load configuration from file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            ConfigParser object
        """
        config = configparser.ConfigParser()
        
        # Check if config file exists
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found at {config_path}. Creating default config.")
            self._create_default_config(config_path)
            
        config.read(config_path)
        return config
        
    def _create_default_config(self, config_path: str):
        """
        Create default configuration file for futures trading
        
        Args:
            config_path: Path to configuration file
        """
        config = configparser.ConfigParser()
        
        # Binance API settings
        config["binance"] = {
            "api_key": "YOUR_BINANCE_API_KEY",
            "api_secret": "YOUR_BINANCE_API_SECRET",
            "testnet": "true"
        }
        
        # DeepSeek API settings
        config["deepseek"] = {
            "api_key": "YOUR_DEEPSEEK_API_KEY",
            "api_url": "https://api.deepseek.com/v1/predictions",
            "test_mode": "false"
        }
        
        # Santiment API settings
        config["santiment"] = {
            "api_key": "YOUR_SANTIMENT_API_KEY",
            "api_url": "https://api.santiment.net/graphql"
        }
        
        # Trading settings
        config["trading"] = {
            "symbol": "BTCUSDT",
            "run_interval_seconds": "180",
            "max_positions": "3",
            "risk_per_trade": "0.02",
            "confidence_threshold": "0.85",
            "close_positions_on_start": "true"
        }
        
        # Futures settings
        config["futures"] = {
            "leverage": "10",
            "margin_type": "ISOLATED"
        }
        
        # Write to file
        with open(config_path, "w") as f:
            config.write(f)
            
        logger.info(f"📝 Created default configuration file at {config_path}")
        logger.info("⚠️ Please edit the configuration file with your API keys and settings")
    
    def run_cycle(self):
        """
        Run a single trading cycle for futures market
        """
        try:
            logger.info(f"🔄 Starting trading cycle for {self.symbol}")
            
            # 1. Monitor positions first
            self.trade_executor.monitor_positions()
            
            # 2. Fetch account metrics
            try:
                account_metrics = self.binance_client.get_account_metrics()
                balance = account_metrics.get("balance_usd", 0)
                equity = account_metrics.get("equity_usd", 0)
                margin_used = account_metrics.get("margin_used_usd", 0)
                free_margin = equity - margin_used
                
                logger.info(f"💼 Account metrics: Balance=${balance:.2f}, Equity=${equity:.2f}, Margin Used=${margin_used:.2f}, Free Margin=${free_margin:.2f}")
                
                # Update session PnL
                positions_pnl = self.trade_executor.calculate_positions_pnl()
                current_pnl = positions_pnl.get("total_pnl_usd", 0)
                self.session_total_pnl = current_pnl
                
                logger.info(f"💰 Current PnL: ${current_pnl:.2f}")
                
            except Exception as e:
                logger.error(f"❌ Error fetching account metrics: {str(e)}")
            
            # 3. Fetch market data (only once)
            market_data = self.binance_client.fetch_market_data(self.symbol)
            if not market_data:
                logger.error("❌ Failed to fetch market data")
                return
                
            logger.info(f"📊 Fetched market data for {self.symbol}")
            
            # 4. Enrich data with indicators
            enriched_data = self.data_enricher.enrich_data(market_data)
            logger.info("🔍 Enriched market data with technical indicators")
            
            # 5. Check for existing positions
            existing_positions = self.trade_executor.get_active_positions()
            existing_count = len(existing_positions)
            
            # 6. Get prediction from DeepSeek based on position state
            if existing_count >= self.max_positions:
                logger.info(f"Maximum positions reached ({existing_count}/{self.max_positions}). Switching to position management mode.")
                # Add position data to enriched data
                enriched_data["active_positions"] = existing_positions
                enriched_data["request_type"] = "position_management"
                
                logger.info("Requesting position management advice from DeepSeek API")
                prediction = self.deepseek_api.get_prediction(enriched_data)
            else:
                # Normal prediction request for new trades
                logger.info("Requesting trading signals from DeepSeek API")
                prediction = self.deepseek_api.get_prediction(enriched_data)
            
            if not prediction:
                logger.error("❌ No prediction received")
                return
            
            # Check if prediction has nested structure with 'prediction' key
            if "prediction" in prediction:
                pred_data = prediction["prediction"]
                action = pred_data.get("action", "HOLD")
                confidence = pred_data.get("confidence", 0)
                stop_loss = pred_data.get("stop_loss")
                take_profit = pred_data.get("take_profit")
                volatility = pred_data.get("volatility", 0.01)
            elif "action" in prediction:
                # Direct structure without nesting
                action = prediction.get("action", "HOLD")
                confidence = prediction.get("confidence", 0)
                stop_loss = prediction.get("stop_loss")
                take_profit = prediction.get("take_profit")
                volatility = prediction.get("volatility", 0.01)
            else:
                logger.error("❌ Invalid prediction format received")
                return
            
            # Log all prediction details including volatility
            logger.info(f"🤖 Prediction: {action} with {confidence:.2f} confidence")
            logger.info(f"📊 Market volatility: {volatility:.6f}")
            print(f"CRITICAL DEBUG - Market volatility value: {volatility:.6f}")
            
            if stop_loss:
                logger.info(f"🛑 Stop Loss: {stop_loss}")
            if take_profit:
                logger.info(f"💰 Take Profit: {take_profit}")
            
            # 7. Execute trade if confidence is high enough
            if confidence >= self.confidence_threshold:
                logger.info(f"✅ Confidence {confidence:.2f} meets threshold {self.confidence_threshold}")
                
                # Handle prediction based on request type
                if enriched_data.get("request_type") == "position_management":
                    # Handle position management advice
                    logger.info("Processing position management advice")
                    
                    # Check if we have position-specific recommendations
                    position_recommendations = prediction.get("position_recommendations", [])
                    
                    if position_recommendations:
                        # Process each position recommendation
                        for rec in position_recommendations:
                            position_id = rec.get("position_id")
                            action = rec.get("action", "HOLD")
                            confidence = rec.get("confidence", 0)
                            
                            if not position_id or confidence < self.confidence_threshold:
                                continue
                                
                            # Find the position
                            position = next((p for p in existing_positions if p.get("position_id") == position_id), None)
                            
                            if not position:
                                logger.warning(f"Position with ID {position_id} not found")
                                continue
                                
                            # Apply the recommendation
                            if action == "CLOSE":
                                logger.info(f"Closing position {position_id} based on recommendation")
                                # TODO: Implement close_position method in trade_executor
                                # self.trade_executor.close_position(position_id)
                                self.trade_executor.close_position(position_id)
                            elif action == "PARTIAL_CLOSE":
                                percentage = rec.get("percentage", 50)
                                logger.info(f"Partially closing position {position_id} ({percentage}%)")
                                # TODO: Implement partial_close_position in trade_executor
                                # self.trade_executor.partial_close_position(position_id, percentage)
                                self.trade_executor.partial_close_position(position_id, percentage)
                            elif action == "MODIFY_SL_TP":
                                stop_loss = rec.get("stop_loss")
                                take_profit = rec.get("take_profit")
                                logger.info(f"Modifying SL/TP for position {position_id}")
                                # TODO: Implement modify_position_sl_tp in trade_executor
                                # self.trade_executor.modify_position_sl_tp(position_id, stop_loss, take_profit)
                                self.trade_executor.modify_position_sl_tp(position_id, stop_loss, take_profit)
                            else:
                                logger.info(f"Holding position {position_id} as recommended")
                    else:
                        # Generic recommendation for all positions
                        action = prediction.get("action", "HOLD")
                        
                        if action == "CLOSE_ALL":
                            logger.info("Closing all positions based on recommendation")
                            self.trade_executor.close_all_positions()
                else:
                    # Normal trade execution (existing code)
                    # Check for existing positions first
                    if existing_count >= self.max_positions and action != "HOLD":
                        logger.warning(f"⚠️ Maximum positions reached ({existing_count}/{self.max_positions})")
                    else:
                        # Skip trade execution for HOLD actions
                        if action == "HOLD":
                            logger.info("⏸️ No trade execution needed for HOLD action")
                        else:
                            # Execute the trade
                            success = self.trade_executor.execute_trade(
                                symbol=self.symbol,
                                action=action,
                                confidence=confidence,
                                confidence_threshold=self.confidence_threshold,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                volatility=volatility
                            )
                            
                            if success:
                                logger.info(f"✅ Successfully executed {action} trade for {self.symbol}")
                                self.session_trades_count += 1
                            else:
                                logger.warning(f"⚠️ Failed to execute {action} trade for {self.symbol}")
            else:
                logger.info(f"⏸️ Confidence {confidence:.2f} below threshold {self.confidence_threshold}, no trade executed")
                
            # 8. Log active positions
            active_positions = self.trade_executor.get_active_positions()
            if active_positions:
                logger.info(f"📋 Active positions ({len(active_positions)}):")
                for pos in active_positions:
                    symbol = pos.get("symbol", "")
                    side = pos.get("side", "")
                    quantity = pos.get("quantity", 0)
                    entry_price = pos.get("entry_price", 0)
                    leverage = pos.get("leverage", self.leverage)
                    logger.info(f"   - {symbol} {side} {quantity} @ {entry_price} ({leverage}x)")
            else:
                logger.info("📋 No active positions")
                
            logger.info(f"✅ Trading cycle completed for {self.symbol}")
            
        except Exception as e:
            logger.error(f"❌ Error in trading cycle: {str(e)}")
            logger.error(traceback.format_exc())
    
    def start(self):
        """Start the trading bot"""
        logger.info("Starting Trading Bot")
        self.running = True
        
        try:
            while self.running:
                cycle_start_time = time.time()
                
                try:
                    # Run the trading cycle
                    self.run_cycle()
                    
                    # Calculate how long to sleep
                    cycle_duration = time.time() - cycle_start_time
                    sleep_time = max(0, self.cycle_interval - cycle_duration)
                    
                    if sleep_time > 0:
                        logger.info(f"Sleeping for {sleep_time:.1f} seconds until next cycle")
                        time.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    logger.error(traceback.format_exc())
                    time.sleep(5)  # Wait before retrying
                
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            self.running = False
            
            # Log final session statistics
            session_duration = int(time.time()) - self.session_start_time
            hours = session_duration // 3600
            minutes = (session_duration % 3600) // 60
            
            logger.info("\nFinal Session Summary:")
            logger.info(f"Duration: {hours}h {minutes}m")
            logger.info(f"Total P/L: {'+'if self.session_total_pnl >= 0 else ''}" +
                       f"${self.session_total_pnl:,.2f}")
            if self.session_trades_count > 0:
                win_rate = (self.session_winning_trades / self.session_trades_count * 100)
                avg_pnl_per_trade = self.session_total_pnl / self.session_trades_count
                logger.info(f"Total Trades: {self.session_trades_count}")
                logger.info(f"Win Rate: {win_rate:.1f}%")
                logger.info(f"Average P/L per Trade: {'+'if avg_pnl_per_trade >= 0 else ''}" +
                           f"${avg_pnl_per_trade:,.2f}")
            
            # Ask user if they want to close all positions
            try:
                response = input("Do you want to close all active positions? (y/n): ").strip().lower()
                if response == 'y' or response == 'yes':
                    logger.info("Closing all active positions...")
                    self.trade_executor.close_all_positions()
                    logger.info("All positions closed.")
                else:
                    logger.info("Keeping all positions open.")
                    active_positions = self.trade_executor.get_active_positions()
                    logger.info(f"Current active positions: {active_positions}")
            except Exception as e:
                logger.error(f"Error while closing positions: {str(e)}")
                
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            logger.error(traceback.format_exc())
            self.running = False
            raise

if __name__ == "__main__":
    bot = TradingBot()
    bot.start()
