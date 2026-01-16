import logging
import os
from datetime import datetime

LOG_DIRS = {}  # Store mapping of logger_id -> log directory

def setup_logging(logger_id="finch", group_name=None):
    """
    Setup logging for a specific project instance.
    """
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    group_name = f"/{group_name}" if group_name else ""
    log_dir = f"logs/{group_name}/{logger_id}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{logger_id}.log")
    LOG_DIRS[logger_id] = log_dir
    logger = logging.getLogger(logger_id)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info(f"Logging setup complete for {logger_id}. Log file: {log_file}")
    return log_dir

def get_logger(logger_id="finch"):
    """
    Retrieve the logger for the given project instance.
    """
    return logging.getLogger(logger_id)

def get_log_dir(logger_id="finch"):
    """
    Retrieve the log directory for a given project instance.
    """
    return LOG_DIRS.get(logger_id, None)



