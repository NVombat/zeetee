import os
import sys


def get_file_path(target_dir: str, foldername: str, filename: str) -> str:
    '''
    Get the path to files in either the data folder (/data/(config/log/test)files)
    or the assets folder (/src/assets)

    Args:
        target_dir: Main Target Directory
        filename: File Name
        foldername: Name of Folder Within the Target Directory

    Returns:
        str: Path to file
    '''
    if not isinstance(target_dir, str) or target_dir not in ["data", "assets"]:
        logger.error("The Target Directory Must Be A String & Must Be Either 'data' or 'assets'!")
        sys.exit(1)

    # Get the directory of the current file
    src_dir = os.path.dirname(__file__)

    # Construct the path to the directory
    if target_dir is "data":
        final_dir = os.path.abspath(os.path.join(src_dir, '..', target_dir, foldername))

    else:
        final_dir = os.path.abspath(os.path.join(src_dir, target_dir, foldername))

    # Combine the directory with the filename to get the full path
    file_path = os.path.join(final_dir, filename)

    return file_path