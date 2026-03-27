"""
ForesightX Logging Module
=========================
Production-grade logging system for the ForesightX stock prediction project.

Features:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- File and console output
- Rotating file handlers to prevent large log files
- Structured log format with timestamps
- Color-coded console output
- Separate logs for different modules
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler


# Define color codes for console output
class LogColors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Regular colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for different log levels"""
    
    COLORS = {
        'DEBUG': LogColors.CYAN,
        'INFO': LogColors.GREEN,
        'WARNING': LogColors.YELLOW,
        'ERROR': LogColors.RED,
        'CRITICAL': LogColors.BRIGHT_RED + LogColors.BOLD
    }
    
    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{LogColors.RESET}"
        
        # Add color to the message based on level
        if hasattr(record, 'msg'):
            if levelname in ['ERROR', 'CRITICAL']:
                record.msg = f"{LogColors.BRIGHT_RED}{record.msg}{LogColors.RESET}"
            elif levelname == 'WARNING':
                record.msg = f"{LogColors.YELLOW}{record.msg}{LogColors.RESET}"
        
        return super().format(record)


class LoggerSetup:
    """Setup and configure logging for the application"""
    
    def __init__(self, 
                 name: str = "ForesightX",
                 log_dir: str = "logs",
                 level: str = "INFO",
                 max_bytes: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        """
        Initialize logger configuration
        
        Args:
            name: Logger name (typically module or component name)
            log_dir: Directory to store log files
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            max_bytes: Maximum size of log file before rotation
            backup_count: Number of backup files to keep
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        
        # Create logs directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Configure and return logger instance"""
        
        # Create logger
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        
        # Remove existing handlers to avoid duplicates
        if logger.handlers:
            logger.handlers.clear()
        
        # Create formatters
        file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_formatter = ColoredFormatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler with rotation
        log_file = self.log_dir / f"{self.name.lower()}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(self.level)
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        return logger
    
    def get_logger(self) -> logging.Logger:
        """Return configured logger instance"""
        return self.logger


def get_logger(name: str = "ForesightX", level: str = "INFO") -> logging.Logger:
    """
    Convenience function to get a logger instance
    
    Args:
        name: Logger name (use module name, e.g., __name__)
        level: Logging level
    
    Returns:
        Configured logger instance
    
    Example:
        >>> from src.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting data processing...")
        >>> logger.warning("Missing some features")
        >>> logger.error("Failed to load model")
    """
    logger_setup = LoggerSetup(name=name, level=level)
    return logger_setup.get_logger()


def log_function_call(func):
    """
    Decorator to log function entry and exit
    
    Example:
        >>> @log_function_call
        >>> def train_model(data):
        >>>     # training code
        >>>     pass
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Entering function: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting function: {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Error in function {func.__name__}: {str(e)}")
            raise
    return wrapper


# Example usage and testing
if __name__ == "__main__":
    # Test the logger
    logger = get_logger("TestLogger", level="DEBUG")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test with different modules
    data_logger = get_logger("DataPipeline")
    model_logger = get_logger("ModelTraining")
    
    data_logger.info("Data pipeline initialized")
    model_logger.info("Model training started")
    
    print("\n✅ Logger setup complete! Check the 'logs' directory for log files.")
