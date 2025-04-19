"""
Logging module for ScaleTent
"""

import logging
import os
import sys
import time
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import threading

# Global variables
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_DIR = "logs"
DEFAULT_MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5

# Thread-local storage for logger cache
_logger_cache = threading.local()


def setup_logger(name, level=None, log_dir=None, log_format=None, 
                 console=True, file=True, max_size=None, backup_count=None):
    """
    Configure and return a logger
    
    Args:
        name (str): Logger name
        level (int, optional): Logging level
        log_dir (str, optional): Log directory
        log_format (str, optional): Log format
        console (bool): Whether to log to console
        file (bool): Whether to log to file
        max_size (int, optional): Maximum log file size
        backup_count (int, optional): Number of backup files
    
    Returns:
        logging.Logger: Configured logger
    """
    # Use cache if available
    if hasattr(_logger_cache, 'loggers') and name in _logger_cache.loggers:
        return _logger_cache.loggers[name]
    
    # Create cache if it doesn't exist
    if not hasattr(_logger_cache, 'loggers'):
        _logger_cache.loggers = {}
    
    # Set defaults
    level = level or DEFAULT_LOG_LEVEL
    log_dir = log_dir or DEFAULT_LOG_DIR
    log_format = log_format or DEFAULT_LOG_FORMAT
    max_size = max_size or DEFAULT_MAX_LOG_SIZE
    backup_count = backup_count or DEFAULT_BACKUP_COUNT
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add console handler if enabled
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if enabled
    if file:
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Determine log file path
        log_file = Path(log_dir) / f"{name.replace('.', '_')}.log"
        
        # Create rotating file handler
        file_handler = RotatingFileHandler(
            str(log_file),
            maxBytes=max_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add logger to cache
    _logger_cache.loggers[name] = logger
    
    return logger


def setup_time_rotating_logger(name, level=None, log_dir=None, log_format=None,
                              console=True, file=True, when='midnight', interval=1, backup_count=None):
    """
    Configure and return a logger with time-based rotation
    
    Args:
        name (str): Logger name
        level (int, optional): Logging level
        log_dir (str, optional): Log directory
        log_format (str, optional): Log format
        console (bool): Whether to log to console
        file (bool): Whether to log to file
        when (str): Rotation interval type ('S', 'M', 'H', 'D', 'midnight')
        interval (int): Rotation interval
        backup_count (int, optional): Number of backup files
    
    Returns:
        logging.Logger: Configured logger
    """
    # Use cache if available
    if hasattr(_logger_cache, 'loggers') and name in _logger_cache.loggers:
        return _logger_cache.loggers[name]
    
    # Create cache if it doesn't exist
    if not hasattr(_logger_cache, 'loggers'):
        _logger_cache.loggers = {}
    
    # Set defaults
    level = level or DEFAULT_LOG_LEVEL
    log_dir = log_dir or DEFAULT_LOG_DIR
    log_format = log_format or DEFAULT_LOG_FORMAT
    backup_count = backup_count or DEFAULT_BACKUP_COUNT
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add console handler if enabled
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if enabled
    if file:
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Determine log file path
        log_file = Path(log_dir) / f"{name.replace('.', '_')}.log"
        
        # Create timed rotating file handler
        file_handler = TimedRotatingFileHandler(
            str(log_file),
            when=when,
            interval=interval,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add logger to cache
    _logger_cache.loggers[name] = logger
    
    return logger


def set_global_log_level(level):
    """
    Set log level for all loggers
    
    Args:
        level (int): Logging level
    """
    if hasattr(_logger_cache, 'loggers'):
        for logger in _logger_cache.loggers.values():
            logger.setLevel(level)


def get_logger(name):
    """
    Get a logger by name
    
    Args:
        name (str): Logger name
    
    Returns:
        logging.Logger: Logger instance
    """
    if hasattr(_logger_cache, 'loggers') and name in _logger_cache.loggers:
        return _logger_cache.loggers[name]
    
    # Create new logger if not found
    return setup_logger(name)