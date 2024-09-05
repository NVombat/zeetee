import os


def get_key_by_value(dictionary, target_value):
    '''
    Takes a dictionary and target value as input and
    returns the key value associated with that
    particular target value

    Args:
        dictionary
        target_value

    Returns:
        dict.key: key value
    '''
    for key, value in dictionary.items():
        if value == target_value:
            return key
    return None


def get_file_path(foldername: str, filename: str, ) -> str:
    '''
    Get the path to files in the data folder (/data/(log/test)files)

    Args:
        filename: File Name
        foldername: Name of Folder Within the Data Folder

    Returns: Path to file
    '''
    # Get the directory of the current file
    src_dir = os.path.dirname(__file__)

    # Construct the path to the 'data/foldername' directory
    data_dir = os.path.abspath(os.path.join(src_dir, '..', 'data', foldername))

    # Combine the directory with the filename to get the full path
    file_path = os.path.join(data_dir, filename)

    return file_path