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
    def __init__(self, config_path: str = "config.ini"):
        """Initialize the trading bot with configuration"""
        logger.info("Initializing Trading Bot")
        
        # Initialize session P/L tracking
        self.session_start_time = int(time.time())
        self.session_total_pnl = 0.0
        self.session_trades_count = 0
        self.session_winning_trades = 0
        self.session_losing_trades = 0
        
        try:
            # Load configuration
            self.config = self._load_config(config_path)
            
            # Initialize components
            logger.info("Initializing Binance client...")
            self.binance_client = BinanceClient(
                api_key=self.config.get('binance', 'api_key'),
                api_secret=self.config.get('binance', 'api_secret'),
                testnet=self.config.getboolean('binance', 'use_testnet')
            )
            
            # Test API permissions
            logger.info("Testing Binance API permissions...")
            api_test_results = self.binance_client.test_api_permissions()
            
            if not (api_test_results["connectivity"] and api_test_results["read_info"]):
                logger.error("Critical API permissions are missing. Please check your API key settings.")
                logger.error("See the guide in BINANCE_SETUP.md for instructions on creating proper API keys.")
                if api_test_results["errors"]:
                    for error in api_test_results["errors"]:
                        logger.error(f"API Error: {error}")
                logger.error("The bot can't function without proper API permissions. Exiting...")
                exit(1)
            
            logger.info("Initializing Santiment API client...")
            self.santiment_api = SantimentAPI(
                api_key=self.config.get('santiment', 'api_key'),
                api_url=self.config.get('santiment', 'api_url')
            )
            
            logger.info("Initializing data enricher...")
            self.data_enricher = DataEnricher(santiment_api=self.santiment_api)
            
            logger.info("Initializing DeepSeek API client...")
            self.deepseek_api = DeepSeekAPI(
                api_key=self.config.get('deepseek', 'api_key'),
                api_url=self.config.get('deepseek', 'api_url'),
                model_name=self.config.get('deepseek', 'model_name', fallback='deepseek-chat')
            )
            
            logger.info("Initializing trade executor...")
            self.trade_executor = TradeExecutor(self.binance_client)
            
            # Trading parameters
            self.symbol = self.config.get('trading', 'symbol')
            self.confidence_threshold = self.config.getfloat('trading', 'confidence_threshold')
            self.run_interval = self.config.getint('trading', 'run_interval_seconds')
            
            logger.info("Trading Bot initialization complete")
            
        except ValueError as e:
            logger.error(f"Configuration error: {str(e)}")
            logger.error("Please check your API keys in the config.ini file")
            exit(1)
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            logger.error(traceback.format_exc())
            exit(1)
        
    def _load_config(self, config_path: str) -> configparser.ConfigParser:
        """Load configuration from file"""
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            self._create_default_config(config_path)
            logger.info(f"Created default configuration file: {config_path}")
            logger.error("Please update the configuration file with your API keys and settings")
            exit(1)
            
        config = configparser.ConfigParser()
        config.read(config_path)
        return config
    
    def _create_default_config(self, config_path: str):
        """Create a default configuration file"""
        config = configparser.ConfigParser()
        
        config['binance'] = {
            'api_key': 'YOUR_BINANCE_API_KEY',
            'api_secret': 'YOUR_BINANCE_API_SECRET',
            'use_testnet': 'True'
        }
        
        config['deepseek'] = {
            'api_key': 'YOUR_DEEPSEEK_API_KEY',
            'api_url': 'https://api.deepseek.com',
            'model_name': 'deepseek-chat'
        }
        
        config['santiment'] = {
            'api_key': 'YOUR_SANTIMENT_API_KEY',
            'api_url': 'https://api.santiment.net/graphql'
        }
        
        config['trading'] = {
            'symbol': 'BTCUSDT',
            'confidence_threshold': '0.80',
            'run_interval_seconds': '300'
        }
        
        with open(config_path, 'w') as f:
            config.write(f)
            
        # Add comments after writing the basic config
        with open(config_path, 'r') as f:
            content = f.read()
            
        # Add comments to the config
        content = content.replace(
            "[trading]", 
            "[trading]\n# Trading parameters configuration"
        )
        content = content.replace(
            "run_interval_seconds = 300", 
            "# Time between trading cycles in seconds (5 minutes)\nrun_interval_seconds = 300"
        )
        content = content.replace(
            "[deepseek]",
            "[deepseek]\n# DeepSeek API configuration (uses OpenAI-compatible API)"
        )
        content = content.replace(
            "api_key = YOUR_DEEPSEEK_API_KEY",
            "# Your DeepSeek API key\napi_key = YOUR_DEEPSEEK_API_KEY"
        )
        content = content.replace(
            "api_url = https://api.deepseek.com",
            "# Base URL for the DeepSeek API (no trailing slashes or path segments)\napi_url = https://api.deepseek.com"
        )
        content = content.replace(
            "model_name = deepseek-chat",
            "# DeepSeek model name to use\nmodel_name = deepseek-chat"
        )
        content = content.replace(
            "[santiment]",
            "[santiment]\n# Santiment API configuration for sentiment and on-chain metrics"
        )
        content = content.replace(
            "api_key = YOUR_SANTIMENT_API_KEY",
            "# Your Santiment API key\napi_key = YOUR_SANTIMENT_API_KEY"
        )
        content = content.replace(
            "api_url = https://api.santiment.net/graphql",
            "# Santiment GraphQL API endpoint\napi_url = https://api.santiment.net/graphql"
        )
        
        with open(config_path, 'w') as f:
            f.write(content)
    
    def run_cycle(self):
        """Execute a single trading cycle"""
        try:
            logger.info(f"Starting trading cycle for {self.symbol}")
            
            # Log account metrics and session performance
            try:
                account_metrics = self.binance_client.get_account_metrics()
                logger.info(f"Account Status: Balance=${account_metrics['balance_usd']:,.2f}, " +
                           f"Equity=${account_metrics['equity_usd']:,.2f}, " +
                           f"Margin Used=${account_metrics['margin_used_usd']:,.2f}, " +
                           f"Free Margin=${account_metrics['free_margin_usd']:,.2f}")
                
                # Log session performance
                session_duration = int(time.time()) - self.session_start_time
                hours = session_duration // 3600
                minutes = (session_duration % 3600) // 60
                
                win_rate = (self.session_winning_trades / self.session_trades_count * 100) if self.session_trades_count > 0 else 0
                
                logger.info(f"Session Performance (Duration: {hours}h {minutes}m):")
                logger.info(f"  Total P/L: {'+'if self.session_total_pnl >= 0 else ''}" +
                           f"${self.session_total_pnl:,.2f}")
                logger.info(f"  Trades: {self.session_trades_count} total, " +
                           f"{self.session_winning_trades} winners, {self.session_losing_trades} losers")
                if self.session_trades_count > 0:
                    logger.info(f"  Win Rate: {win_rate:.1f}%")
                
            except Exception as e:
                logger.error(f"Error getting account metrics: {str(e)}")
            
            # Log current active positions
            active_positions_count = self.trade_executor.get_active_positions_count()
            logger.info(f"Currently managing {active_positions_count} active positions (max: {self.trade_executor.max_concurrent_trades})")
            
            # Check stop loss and take profit for positions without algo orders
            active_positions = self.trade_executor.get_active_positions()
            for position in active_positions:
                try:
                    # Get current market price
                    ticker = self.binance_client._send_request(
                        method="GET",
                        endpoint="/v3/ticker/price",
                        params={"symbol": position["symbol"]}
                    )
                    
                    if ticker and "price" in ticker:
                        current_price = float(ticker["price"])
                        stop_loss = position.get("stop_loss")
                        take_profit = position.get("take_profit")
                        
                        # Format stop loss and take profit prices according to Binance's price filter
                        if stop_loss is not None:
                            stop_loss = self.binance_client.format_price(position["symbol"], stop_loss)
                            
                        if take_profit is not None:
                            take_profit = self.binance_client.format_price(position["symbol"], take_profit)
                        
                        # Calculate current P/L for position
                        entry_price = position["entry_price"]
                        quantity = position["quantity"]
                        
                        if position["side"] == "BUY":
                            position_pnl = (current_price - entry_price) * quantity
                            # Check if we need to close position due to stop loss or take profit
                            if stop_loss and current_price <= stop_loss:
                                logger.info(f"Stop loss triggered for {position['symbol']} BUY position at {current_price}")
                                if self.trade_executor._execute_sell(
                                    symbol=position["symbol"],
                                    quantity=position["quantity"],
                                    stop_loss=None,
                                    take_profit=None
                                ):
                                    # Update session statistics
                                    self.session_total_pnl += position_pnl
                                    self.session_trades_count += 1
                                    if position_pnl >= 0:
                                        self.session_winning_trades += 1
                                    else:
                                        self.session_losing_trades += 1
                                        
                            elif take_profit and current_price >= take_profit:
                                logger.info(f"Take profit triggered for {position['symbol']} BUY position at {current_price}")
                                if self.trade_executor._execute_sell(
                                    symbol=position["symbol"],
                                    quantity=position["quantity"],
                                    stop_loss=None,
                                    take_profit=None
                                ):
                                    # Update session statistics
                                    self.session_total_pnl += position_pnl
                                    self.session_trades_count += 1
                                    if position_pnl >= 0:
                                        self.session_winning_trades += 1
                                    else:
                                        self.session_losing_trades += 1
                        else:  # SELL position
                            position_pnl = (entry_price - current_price) * quantity
                            if stop_loss and current_price >= stop_loss:
                                logger.info(f"Stop loss triggered for {position['symbol']} SELL position at {current_price}")
                                if self.trade_executor._execute_buy(
                                    symbol=position["symbol"],
                                    quantity=position["quantity"],
                                    stop_loss=None,
                                    take_profit=None
                                ):
                                    # Update session statistics
                                    self.session_total_pnl += position_pnl
                                    self.session_trades_count += 1
                                    if position_pnl >= 0:
                                        self.session_winning_trades += 1
                                    else:
                                        self.session_losing_trades += 1
                                        
                            elif take_profit and current_price <= take_profit:
                                logger.info(f"Take profit triggered for {position['symbol']} SELL position at {current_price}")
                                if self.trade_executor._execute_buy(
                                    symbol=position["symbol"],
                                    quantity=position["quantity"],
                                    stop_loss=None,
                                    take_profit=None
                                ):
                                    # Update session statistics
                                    self.session_total_pnl += position_pnl
                                    self.session_trades_count += 1
                                    if position_pnl >= 0:
                                        self.session_winning_trades += 1
                                    else:
                                        self.session_losing_trades += 1
                except Exception as e:
                    logger.error(f"Error checking stop loss/take profit for position {position['symbol']}: {str(e)}")
            
            # Log positions profit/loss
            if active_positions_count > 0:
                try:
                    positions_pnl = self.trade_executor.calculate_positions_pnl()
                    total_pnl = positions_pnl["total_pnl_usd"]
                    pnl_sign = "+" if total_pnl >= 0 else ""
                    logger.info(f"Current P/L: {pnl_sign}${total_pnl:,.2f}")
                    
                    # Log individual position details
                    for position in positions_pnl["positions"]:
                        symbol = position["symbol"]
                        side = position["side"]
                        pnl = position["pnl_usd"]
                        pnl_percent = position["pnl_percent"]
                        pnl_sign = "+" if pnl >= 0 else ""
                        logger.info(f"  {side} {symbol}: {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_percent:.2f}%)")
                except Exception as e:
                    logger.error(f"Error calculating positions P/L: {str(e)}")
            
            # Step 1: Fetch market data from Binance
            market_data = self.binance_client.fetch_market_data(self.symbol)
            
            # Step 2: Enrich data with indicators
            enriched_data = self.data_enricher.enrich_data(market_data)
            
            # Step 3: Send data to DeepSeek API
            prediction = self.deepseek_api.get_prediction(enriched_data)
            
            # Step 4: Log prediction
            logger.info(f"Received prediction: {json.dumps(prediction)}")
            
            # Step 5: Execute trade based on prediction
            if prediction and 'prediction' in prediction:
                action = prediction['prediction']['action']
                confidence = prediction['prediction']['confidence']
                stop_loss = prediction['prediction'].get('stop_loss')
                take_profit = prediction['prediction'].get('take_profit')
                
                # Format stop loss and take profit prices according to Binance's price filter
                if stop_loss is not None:
                    stop_loss = self.binance_client.format_price(self.symbol, stop_loss)
                    
                if take_profit is not None:
                    take_profit = self.binance_client.format_price(self.symbol, take_profit)
                
                trade_executed = self.trade_executor.execute_trade(
                    symbol=self.symbol,
                    action=action,
                    confidence=confidence,
                    confidence_threshold=self.confidence_threshold,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    volatility=enriched_data['timeframes']['5m']['indicators'].get('WVO', 0.01)
                )
                
                if trade_executed:
                    logger.info(f"Successfully executed {action} trade")
                    # Log updated position count
                    new_positions_count = self.trade_executor.get_active_positions_count()
                    logger.info(f"Now managing {new_positions_count} active positions")
                    
                    # Log updated account metrics after trade
                    try:
                        updated_metrics = self.binance_client.get_account_metrics()
                        logger.info(f"Updated Account Status: Balance=${updated_metrics['balance_usd']:,.2f}, " +
                                   f"Equity=${updated_metrics['equity_usd']:,.2f}, " +
                                   f"Margin Used=${updated_metrics['margin_used_usd']:,.2f}")
                    except Exception as e:
                        logger.error(f"Error getting updated account metrics: {str(e)}")
                else:
                    logger.info(f"No trade was executed")
            else:
                logger.warning("Invalid prediction format received")
                
        except Exception as e:
            logger.error(f"Error in trading cycle: {str(e)}")
            logger.error(traceback.format_exc())
    
    def start(self):
        """Start the trading bot"""
        logger.info("Starting Trading Bot")
        
        try:
            while True:
                self.run_cycle()
                logger.info(f"Sleeping for {self.run_interval} seconds")
                time.sleep(self.run_interval)
        except KeyboardInterrupt:
            logger.info("Trading Bot stopped by user")
            
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
                    
                    # Calculate final P/L for all positions before closing
                    positions_pnl = self.trade_executor.calculate_positions_pnl()
                    for position in positions_pnl["positions"]:
                        pnl = position["pnl_usd"]
                        self.session_total_pnl += pnl
                        self.session_trades_count += 1
                        if pnl >= 0:
                            self.session_winning_trades += 1
                        else:
                            self.session_losing_trades += 1
                    
                    self.trade_executor.close_all_positions()
                    logger.info("All positions closed.")
                    
                    # Log updated final statistics
                    logger.info("\nUpdated Final Session Summary:")
                    logger.info(f"Total P/L: {'+'if self.session_total_pnl >= 0 else ''}" +
                               f"${self.session_total_pnl:,.2f}")
                    if self.session_trades_count > 0:
                        win_rate = (self.session_winning_trades / self.session_trades_count * 100)
                        avg_pnl_per_trade = self.session_total_pnl / self.session_trades_count
                        logger.info(f"Total Trades: {self.session_trades_count}")
                        logger.info(f"Win Rate: {win_rate:.1f}%")
                        logger.info(f"Average P/L per Trade: {'+'if avg_pnl_per_trade >= 0 else ''}" +
                                   f"${avg_pnl_per_trade:,.2f}")
                else:
                    logger.info("Keeping all positions open.")
                    active_positions = self.trade_executor.get_active_positions()
                    logger.info(f"Current active positions: {active_positions}")
            except Exception as e:
                logger.error(f"Error while closing positions: {str(e)}")
                
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            logger.error(traceback.format_exc())
            raise

if __name__ == "__main__":
    bot = TradingBot()
    bot.start()
