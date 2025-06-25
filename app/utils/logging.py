import logging
from logging.config import dictConfig
from app.config import settings
import sys

def configure_logging():
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": settings.LOG_FORMAT,
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
        },
        "handlers": {
            "console": {
                "level": settings.LOG_LEVEL,
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
            },
            "file": {
                "level": settings.LOG_LEVEL,
                "formatter": "standard",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "app.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            },
        },
        "loggers": {
            "app": {
                "handlers": ["console", "file"],
                "level": settings.LOG_LEVEL,
                "propagate": False
            },
            "sqlalchemy": {
                "handlers": ["console"],
                "level": "WARNING",
                "propagate": False
            },
        }
    }
    
    dictConfig(logging_config)
    logger = logging.getLogger("app")
    logger.info("Logging configured successfully")
    return logger

logger = configure_logging()