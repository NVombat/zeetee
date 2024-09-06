import sys
import json
from pysat.card import *

from logger import create_logger
from helper import get_file_path

logger = create_logger(l_name="zt_converter")

default_jfp = get_file_path("testfiles", "rgp_hc.json")


def json_to_rgp(json_file_path=default_jfp) -> list:
    '''
    Converts a JSON object to a list of RGP dictionary object(s)

    Args:
        json_file_path: Path to JSON file

    Returns:
        list: list of RGP instance(s) dicts
    '''
    logger.info(f"File Path: {json_file_path}")

    try:
        with open(json_file_path, "r") as fh:
            rgp_obj = json.load(fh)

    except FileNotFoundError:
        logger.error(f"{json_file_path}: File Not Found")

    except PermissionError:
        logger.error(f"{json_file_path}: File Access Not Permitted")

    except Exception as e:
        logger.error(f"Error Accessing RGP Instances from File: {json_file_path}")

    logger.info(f"RGP Object: {rgp_obj}")

    instances = list(rgp_obj.keys())
    assert len(instances) != 0, logger.error(f"No RGP Instances Present In The RGP Object: {rgp_obj}")

    logger.info(f"Number of Instances in the RGP Object: {len(instances)}")

    rgp_instances = []

    for inst in instances:
        val = rgp_obj[inst]
        rgp_instances.append(val)

    logger.info(f"RGP Instances: {rgp_instances}")

    return rgp_instances


def extract_clauses(sat_obj: dict) -> list:
    '''
    Takes a SAT object and extracts its clauses to be
    used by the SAT Solver

    Args:
        sat_obj: SAT dictionary object

    Returns:
        list: List of all clauses in the SAT Object
    '''
    logger.info("Extracting Clauses from SAT Object...")

    final_clauses = []

    clauses = sat_obj["clauses"]
    clauses_keys = list(clauses.keys())

    clause_cnt = 0
    final_lit_cnt = 0

    for key in clauses_keys:
        logger.debug(f"Clause Key: {key}")

        clause = clauses[key]
        logger.debug(f"{key}: {clause}")

        clause_cnt = clause_cnt + len(clause)

        for cl in clause:
            final_clauses.append(cl)
            final_lit_cnt = final_lit_cnt + len(cl)

    logger.debug(f"Sum of Clauses: {clause_cnt}")
    logger.debug(f"Length of Final Clauses: {len(final_clauses)}")
    logger.debug(f"Final Literal Count: {final_lit_cnt}")

    assert clause_cnt==len(final_clauses), logger.error("Issue Extracting Correct Number of Clauses")
    logger.debug(f"SAT Object Clauses: {final_clauses}")

    return final_clauses


if __name__ == "__main__":
    logger.info("********************CONVERTER[LOCAL_TESTING]*********************")