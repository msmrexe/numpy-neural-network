import logging
import os
import sys
from logging.handlers import RotatingFileHandler

def setup_logger(log_dir="logs", log_file="app.log", level=logging.DEBUG):
    """
    Sets up a centralized logger that writes to both the console and a file.

    Args:
        log_dir (str): The directory to store log files.
        log_file (str): The name of the log file.
        level (int): The logging level (e.g., logging.DEBUG, logging.INFO).

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except OSError as e:
            print(f"Error creating log directory {log_dir}: {e}")
            sys.exit(1)

    log_path = os.path.join(log_dir, log_file)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Prevent duplicate handlers if already configured
    if logger.hasHandlers():
        return logger

    # --- File Handler ---
    # Rotates logs: max 5MB per file, keeps 5 backup files
    try:
        file_handler = RotatingFileHandler(
            log_path, maxBytes=5*1024*1024, backupCount=5
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except PermissionError:
        print(f"Error: Insufficient permissions to write log file at {log_path}")
        # Continue with console-only logging
    except Exception as e:
        print(f"Error setting up file handler: {e}")

    # --- Console (Stream) Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Less verbose for console
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info("Logger configured. Logging to console and %s", log_path)
    
    return logger
