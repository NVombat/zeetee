from .utils import get_file_path
from .logger import create_logger
from .er_encoder import rgp_to_sat_er
from .mb_encoder import rgp_to_sat_mb
from . import assets_dir, files_sub_dir
from .helper import extract_clauses_and_instance_data, write_to_file

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
    sat_objects_e1 = {}
    sat_object_cnt_e1 = 0

    for inst in rgp_instances:
        temp_obj = {}

        temp_obj["i"] = inst["i"]
        temp_obj["N"] = inst["n"]

        sat_obj = rgp_to_sat_mb(inst)
        logger.debug(f"SAT Object [MB-ENCODER]: {sat_obj}")

        clauses,instance_data = extract_clauses_and_instance_data(sat_obj)
        logger.debug(f"Clauses: {clauses}")
        logger.debug(f"Instance Data: {instance_data}")

        temp_obj["clauses"] = clauses
        temp_obj["instance_data"] = instance_data

        sat_objects_e1[sat_object_cnt_e1] = temp_obj
        sat_object_cnt_e1 += 1

    target_dir = assets_dir
    target_sub_dir = files_sub_dir

    sat_obj_filename_e1 = f"preprocessed_sat_obj_e1_N{inst['n']}.json"
    sat_obj_e1_fp = get_file_path(target_dir, target_sub_dir, sat_obj_filename_e1)

    write_to_file(sat_objects_e1, sat_obj_e1_fp)

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
    sat_objects_e2 = {}
    sat_object_cnt_e2 = 0

    for inst in rgp_instances:
        temp_obj = {}

        temp_obj["i"] = inst["i"]
        temp_obj["N"] = inst["n"]

        sat_obj = rgp_to_sat_er(inst)
        logger.debug(f"SAT Object [ER-ENCODER]: {sat_obj}")

        clauses,instance_data = extract_clauses_and_instance_data(sat_obj)
        logger.debug(f"Clauses: {clauses}")
        logger.debug(f"Instance Data: {instance_data}")

        temp_obj["clauses"] = clauses
        temp_obj["instance_data"] = instance_data

        sat_objects_e2[sat_object_cnt_e2] = temp_obj
        sat_object_cnt_e2 += 1

    target_dir = assets_dir
    target_sub_dir = files_sub_dir

    sat_obj_filename_e2 = f"preprocessed_sat_obj_e2_N{inst['n']}.json"
    sat_obj_e2_fp = get_file_path(target_dir, target_sub_dir, sat_obj_filename_e2)

    write_to_file(sat_objects_e2, sat_obj_e2_fp)

    return sat_obj_e2_fp


if __name__ == "__main__":
    logger.info("********************PREPROCESSOR[LOCAL_TESTING]*********************")