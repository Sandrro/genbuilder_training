import sys
import logging
from loguru import logger
from iduconfig import Config

class InterceptHandler(logging.Handler):
    """
    Лог-обработчик для перенаправления сообщений из стандартного модуля logging в Loguru.
    """
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

class StreamToLogger:
    """
    Перехватывает вывод из print() (stdout/stderr) и направляет его в Loguru.
    """
    def __init__(self, level: str):
        self.level = level

    def write(self, message: str):
        message = message.rstrip()
        if message:  
            logger.log(self.level, message)

    def flush(self):
        pass 

def setup_logger(config: Config, log_level: str = "INFO") -> None:
    logger.remove()
    
    console_log_format = "<green>{time:MM-DD HH:mm}</green> | <level>{level:<8}</level> | <cyan>{message}</cyan>"
    logger.add(
        sys.__stdout__,
        format=console_log_format,
        level=log_level,
        colorize=True,
    )
    
    file_log_format = console_log_format
    log_file = config.get("LOG_FILE")
    logger.add(
        f"{log_file}.log",
        level=log_level,
        format=file_log_format,
        colorize=False,
        backtrace=True,
        diagnose=True,
    )
    
    logging.basicConfig(handlers=[InterceptHandler()], level=0)
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).handlers = [InterceptHandler()]

    sys.stdout = StreamToLogger("INFO")
    sys.stderr = StreamToLogger("ERROR")
