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
    Set up logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        
    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    
    # Create file handler for all logs
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "trading_bot.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(numeric_level)
    
    # Create file handler for errors only
    error_handler = RotatingFileHandler(
        os.path.join(log_dir, "error.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Set formatters
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    
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