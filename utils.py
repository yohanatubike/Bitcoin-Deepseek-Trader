"""
Utility functions for the trading bot
"""

import os
import logging
import json
from logging.handlers import RotatingFileHandler
from typing import Dict, Any

def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    """
    Set up stylish logging configuration with colored output
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        
    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Set up logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Define color codes for different log levels
    class ColoredFormatter(logging.Formatter):
        COLORS = {
            'DEBUG': '\033[36m',  # Cyan
            'INFO': '\033[32m',   # Green
            'WARNING': '\033[33m', # Yellow
            'ERROR': '\033[31m',  # Red
            'CRITICAL': '\033[1;31m', # Bold Red
            'RESET': '\033[0m'    # Reset
        }
        
        SYMBOLS = {
            'DEBUG': '🔍',
            'INFO': '✅',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'CRITICAL': '🔥'
        }
        
        def format(self, record):
            log_symbol = self.SYMBOLS.get(record.levelname, '')
            log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            log_reset = self.COLORS['RESET']
            
            # Format the message with color
            record.levelname = f"{log_color}{record.levelname}{log_reset}"
            record.msg = f"{log_symbol} {record.msg}"
            
            return super().format(record)
    
    # Create formatters
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(console_formatter)
    
    # Create file handler (rotating log file)
    log_file = os.path.join(log_dir, 'trading_bot.log')
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(file_formatter)
    
    # Get logger and add handlers
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized at level {log_level}")
    logger.info(f"Log file: {log_file}")
    
    return logger

def save_json(data: Any, filename: str) -> None:
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        filename: Output filename
    """
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def load_json(filename: str) -> Any:
    """
    Load data from JSON file
    
    Args:
        filename: Input filename
        
    Returns:
        Loaded data
    """
    if not os.path.exists(filename):
        return None
        
    with open(filename, "r") as f:
        return json.load(f)

def calculate_profit_loss(entry_price: float, current_price: float, side: str, quantity: float) -> float:
    """
    Calculate profit/loss for a position
    
    Args:
        entry_price: Entry price
        current_price: Current price
        side: Position side (BUY or SELL)
        quantity: Position quantity
        
    Returns:
        Profit/loss in quote currency
    """
    if side == "BUY":
        return (current_price - entry_price) * quantity
    elif side == "SELL":
        return (entry_price - current_price) * quantity
    else:
        return 0.0

def format_currency(amount: float, currency: str = "USDT", decimals: int = 2) -> str:
    """
    Format currency amount
    
    Args:
        amount: Amount to format
        currency: Currency symbol
        decimals: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    return f"{amount:.{decimals}f} {currency}" 