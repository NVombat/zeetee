from .utils import get_file_path
from .logger import create_logger
from .helper import write_to_file
from .er_encoder import rgp_to_sat_er
from .mb_encoder import rgp_to_sat_mb
from . import assets_dir, files_sub_dir

logger = create_logger(l_name="zt_preprocessor")


def preprocess_instances_e1(rgp_instances: list, job_id: int) -> str:
    '''
    Takes RGP instances, converts them to SAT objects
    using encoding 1 and stores the necessary data in
    in a JSON file

    Args:
        rgp_instances: List of RGP instances
        job_id: SLURM Job ID (Unique Identifier)

    Returns:
        str: Path to SAT Objects
    '''
    encoding_type = "e1"
    sat_object_cnt_e1 = 0

    target_dir = assets_dir
    target_sub_dir = files_sub_dir

    sat_obj_filename_e1 = f"preprocessed_sat_obj_e1_N{rgp_instances[0]['n']}.json"
    sat_obj_e1_fp = get_file_path(target_dir, target_sub_dir, sat_obj_filename_e1)

    for inst in rgp_instances:
        preprocess_helper(
            instance=inst,
            filepath=sat_obj_e1_fp,
            encoding_type=encoding_type,
            index=sat_object_cnt_e1
        )

        sat_object_cnt_e1 += 1

    logger.info(f"[{encoding_type.capitalize()}] Preprocessing Complete!")

    return sat_obj_e1_fp


def preprocess_instances_e2(rgp_instances: list, job_id: int) -> str:
    '''
    Takes RGP instances, converts them to SAT objects
    using encoding 2 and stores the necessary data in
    in a JSON file

    Args:
        rgp_instances: List of RGP instances
        job_id: SLURM Job ID (Unique Identifier)

    Returns:
        str: Path to SAT Objects
    '''
    encoding_type = "e2"
    sat_object_cnt_e2 = 0

    target_dir = assets_dir
    target_sub_dir = files_sub_dir

    sat_obj_filename_e2 = f"preprocessed_sat_obj_e2_N{rgp_instances[0]['n']}.json"
    sat_obj_e2_fp = get_file_path(target_dir, target_sub_dir, sat_obj_filename_e2)

    for inst in rgp_instances:
        preprocess_helper(
            instance=inst,
            filepath=sat_obj_e2_fp,
            encoding_type=encoding_type,
            index=sat_object_cnt_e2
        )

        sat_object_cnt_e2 += 1

    logger.info(f"[{encoding_type.capitalize()}] Preprocessing Complete!")

    return sat_obj_e2_fp


def preprocess_helper(instance: dict, filepath: str, encoding_type: str, index: int) -> None:
    '''
    Converts an RGP instance to a SAT object and writes it to file

    Args:
        instance: A single RGP instance
        filepath: Path to file containing SAT objects
        encoding_type: Encoding Type E1 or E2
        index: What index to add SAT object to in the dictionary

    Returns:
        None: Writes SAT Object to file
    '''
    if not isinstance(encoding_type, str) or encoding_type not in ["e1", "e2"]:
        logger.error(f"Invalid Encoding Type Given: {encoding_type}. Please Specify A Valid Encoding Type (e1, e2)!")
        sys.exit(1)

    temp_obj = {}

    temp_obj["i"] = instance["i"]
    temp_obj["N"] = instance["n"]

    if encoding_type == "e1":
        sat_obj = rgp_to_sat_mb(instance)

    elif encoding_type == "e2":
        sat_obj = rgp_to_sat_er(instance)

    logger.debug(f"[{encoding_type.capitalize()}] SAT Object: {sat_obj}")

    temp_obj["clauses"] = sat_obj["final_clauses"]
    temp_obj["instance_data"] = sat_obj["instance_data"]

    sat_obj_temp = {}
    sat_obj_temp[index] = temp_obj

    mode = "a"
    write_to_file(sat_obj_temp, filepath, mode)

    logger.info(f"[{encoding_type.capitalize()}] Instance {index} Preprocessed! Written To File!")

    del sat_obj, temp_obj, sat_obj_temp

    return


if __name__ == "__main__":
    logger.info("********************PREPROCESSOR[LOCAL_TESTING]*********************")