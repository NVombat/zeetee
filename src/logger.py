import os
import logging

from .utils import get_file_path


def create_logger(f_name="zt.log", l_name="zt_logger"):
    '''
    Logger Configuration

    Args:
        f_name: Logfile name
        l_name: Logger name

    Returns:
        logger
    '''
    logger = logging.getLogger(l_name)
    logger.setLevel(logging.DEBUG)

    target_dir = "data"
    fs_foldername = "logfiles"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%m-%d %H:%M",
        filename=get_file_path(target_dir, fs_foldername, f_name),
        filemode='w'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)s: %(levelname)s %(message)s")
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger