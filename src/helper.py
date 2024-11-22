import os
import sys
import json
import itertools
from datetime import datetime

from . import default_jfp
from .logger import create_logger

logger = create_logger(l_name="zt_helper")


def json_to_rgp(json_file_path=default_jfp) -> list:
    '''
    Converts a JSON object to a list of RGP dictionary object(s)

    Args:
        json_file_path: Path to JSON file

    Returns:
        list: list of RGP instance(s) dicts
    '''
    logger.debug(f"File Path: {json_file_path}")

    try:
        with open(json_file_path, "r") as fh:
            rgp_obj = json.load(fh)

    except FileNotFoundError:
        logger.error(f"{json_file_path}: File Not Found")

    except PermissionError:
        logger.error(f"{json_file_path}: File Access Not Permitted")

    except Exception as e:
        logger.error(f"Error Accessing RGP Instances from File: {json_file_path}")

    logger.debug(f"RGP Object: {rgp_obj}")

    instances = list(rgp_obj.keys())
    assert len(instances) != 0, logger.error(f"No RGP Instances Present In The RGP Object: {rgp_obj}")

    logger.debug(f"Number of Instances in the RGP Object: {len(instances)}")

    rgp_instances = []

    for inst in instances:
        val = rgp_obj[inst]
        rgp_instances.append(val)

    logger.debug(f"RGP Instances: {rgp_instances}")

    return rgp_instances


def rgp_dict_to_rgp(rgp_dict: dict) -> list:
    '''
    Converts an RGP object (here object refers to the object that is written to the
    JSON file) to a list of RGP dictionary object(s) of the correct format

    Args:
        rgp_dict: RGP Dictionary Object

    Returns:
        list: list of RGP instance(s) dicts
    '''
    instances = list(rgp_dict.keys())
    assert len(instances) != 0, logger.error(f"No RGP Instances Present In The RGP Object: {rgp_dict}")

    logger.debug(f"Number of Instances in the RGP Object: {len(instances)}")

    rgp_instances = []

    for inst in instances:
        val = rgp_dict[inst]
        rgp_instances.append(val)

    logger.debug(f"RGP Instances: {rgp_instances}")

    return rgp_instances


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


def generate_unique_pairs(arr: list) -> list:
    '''
    Generate all unique pairs of elements from an array

    Args:
        arr: Array of elements

    Returns:
        list: A list of all the unique pairs generated
    '''
    return list(itertools.combinations(arr, 2))


def json_to_dict(json_file_path: str) -> dict:
    '''
    Converts a JSON object to a dictionary object

    Args:
        json_file_path: Path to JSON file

    Returns:
        dict: Dictionary containing JSON object
    '''
    logger.debug(f"File Path: {json_file_path}")

    try:
        with open(json_file_path, "r") as fh:
            dict_obj = json.load(fh)

    except FileNotFoundError:
        logger.error(f"{json_file_path}: File Not Found")

    except PermissionError:
        logger.error(f"{json_file_path}: File Access Not Permitted")

    except Exception as e:
        logger.error(f"Error Accessing JSON File: {json_file_path}")

    logger.debug(f"Dictionary Object: {dict_obj}")

    return dict_obj


def write_to_file(some_dict: dict, json_file_name: str, mode="w") -> None:
    '''
    Writes (Appends) a Dictionary Object to a JSON File

    Args:
        some_dict: A dictionary
        json_file_name: File name
        mode: Mode of writing (write/append) [Default = "w"]

    Returns:
        None
    '''
    if not isinstance(mode, str) or mode not in ["a", "w"]:
        logger.error(f"Incorrect mode of writing to file: {mode}. Must be a string that is either 'w' or 'a'!")
        sys.exit(1)

    logger.debug(f"JSON Filename: {json_file_name}")

    try:
        if mode == "a" and os.path.exists(json_file_name):
            # Read existing content
            with open(json_file_name, "r") as fh:
                try:
                    existing_data = json.load(fh)

                except json.JSONDecodeError:
                    existing_data = {}

            # Ensure the existing data is a dictionary
            if not isinstance(existing_data, dict):
                logger.error("Existing data is not a dictionary... Cannot merge...")
                return

            # Merge existing data with new data
            existing_data.update(some_dict)
            some_dict = existing_data

        json_obj = json.dumps(some_dict, indent=4)

        # Write to file
        with open(json_file_name, "w") as fh:
            fh.write(json_obj)

        logger.debug(f"Successfully wrote/merged JSON data to {json_file_name}")

    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: The file {json_file_name} was not found. {e}")

    except IOError as e:
        logger.error(f"IOError: An error occurred while writing to the file {json_file_name}. {e}")

    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError: An error occurred while converting the object to JSON. {e}")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


def get_constraint_intersection(constraints: tuple) -> int:
    '''
    Calculate the intersection of UC and SC

    Args:
        constraints: Tuple containing UC and SC

    Returns:
        int: An integer representing the intersection percentage
    '''
    uc = constraints[0]
    sc = constraints[1]

    uc_resources = [set(constraint[0]) for constraint in uc]
    sc_resources = [set(constraint[0]) for constraint in sc]

    # Count total unique resources in both lists
    total_unique_resources = set().union(*uc_resources, *sc_resources)
    logger.debug(f"Total Unique Resources: {total_unique_resources}")

    # Calculate total intersection resources
    total_intersection = 0
    for uc_res in uc_resources:
        for sc_res in sc_resources:
            total_intersection += len(uc_res.intersection(sc_res))

    degree_of_intersection = (total_intersection / len(total_unique_resources)) * 100
    logger.debug(f"Degree Of Intersection: {degree_of_intersection}")

    return int(degree_of_intersection)


def generate_timestamp_string() -> str:
    '''
    Generate a timestamp and convert it to a string

    Args:
        None

    Returns:
        str: A timestamp string
    '''
    current_timestamp = datetime.now()
    timestamp_string = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")

    return timestamp_string