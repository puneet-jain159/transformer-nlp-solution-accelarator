import logging


def get_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
        datefmt="%m/%d/%Y %H:%M:%S",
        filename="pipeline.log",
        filemode='a')
    logger = logging.getLogger(__name__)
    return logger
