import logging
from pathlib import Path


def create_logger(
    name="trading_bot",
    log_file="logs/trading.log"
):

    Path(log_file).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)

    if not logger.handlers:

        handler = logging.FileHandler(log_file)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger

