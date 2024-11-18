import os
import sys

from .utils import get_file_path
from .logger import create_logger
from .runner import get_experiment_config_and_run_experiment
from . import data_dir, config_sub_dir, assets_dir, files_sub_dir

logger = create_logger(l_name="zt_arc_runner")


if __name__ == "__main__":
    logger.info("********************ARC_RUNNER[RUNNING EXPERIMENT]*********************")

    cl_args = sys.argv

    if len(cl_args) != 3:
        logger.error("Usage: python arc_runner.py file_extension operation [file_extension must be a string and operation must be an integer 0, 1, 2 or 3]")
        sys.exit(1)

    logger.info(f"Command Line Args: {cl_args}")

    try:
        filename_extension = cl_args[1]
        operation = int(cl_args[2])

        if operation not in range(0,4) or not isinstance(cl_args[1], str):
            raise ValueError

    except ValueError:
        logger.error("Usage: python arc_runner.py file_extension operation [file_extension must be a string and operation must be an integer 0, 1, 2 or 3]")
        sys.exit(1)

    if filename_extension == "DNE": # Does Not Exist
        logger.error("No Extension Passed! Please Provide A Valid Extension")
        sys.exit(1)

    else:
        exp_config_filename = f"experiment_config_{filename_extension}.json"
        experiment_config_path = get_file_path(data_dir, config_sub_dir, exp_config_filename)

        if not os.path.exists(experiment_config_path):
            logger.error("Invalid Configuration Path! Please Provide A Valid Extension")
            sys.exit(1)

    logger.info(f"Experiment Configuration Path: {experiment_config_path}")

    job_id = int(os.environ.get('SLURM_JOB_ID', -1))
    logger.info(f"JOB ID: {job_id}")

    uid = str(job_id) + "_" + filename_extension
    # uid = f"{job_id}_{filename_extension}"
    logger.info(f"Unique ID: {uid}")

    if operation == 0:
        logger.info(f"OPERATION {operation}: RUN EXISTING")

        # Get the directory of the current file
        src_dir = os.path.dirname(__file__)
        # Get path to assets directory
        folder_path = os.path.abspath(os.path.join(src_dir, assets_dir, files_sub_dir))

        existing_filename = ""

        for file_name in os.listdir(folder_path):
            ext_part = file_name.split('_')[-1]
            specific_part = ext_part.split('.')[0]

            logger.info(f"File Extension: {specific_part}")

            if filename_extension == specific_part:
                existing_filename = file_name
                break

        logger.info(f"Existing File Name: {existing_filename}")

        existing_fp = get_file_path(assets_dir, files_sub_dir, existing_filename)

        # Run Existing
        get_experiment_config_and_run_experiment(
            f_path=experiment_config_path,
            job_id=uid,
            run_serially=False,
            plot_results=True,
            mail_results=True,
            run_existing=True,
            preprocess=False,
            solve_preprocessed=False,
            existing_fp=existing_fp
        )

    if operation == 1:
        logger.info(f"OPERATION {operation}: SOLVE & PREPROCESS")

        # Solve & Preprocess
        get_experiment_config_and_run_experiment(
            f_path=experiment_config_path,
            job_id=uid,
            run_serially=False,
            plot_results=True,
            mail_results=True,
            run_existing=False,
            preprocess=False,
            solve_preprocessed=False
        )

    elif operation == 2:
        logger.info(f"OPERATION {operation}: PREPROCESS")

        # Preprocess
        get_experiment_config_and_run_experiment(
            f_path=experiment_config_path,
            job_id=uid,
            run_serially=False,
            plot_results=True,
            mail_results=True,
            run_existing=False,
            preprocess=True,
            solve_preprocessed=False
        )

    elif operation == 3:
        logger.info(f"OPERATION {operation}: SOLVE PREPROCESSED")

        # Solve Preprocessed
        sat_obj_filename_e1 = f"preprocessed_sat_obj_e1_{filename_extension}.json"
        sat_objects_e1_fp = get_file_path(assets_dir, files_sub_dir, sat_obj_filename_e1)

        sat_obj_filename_e2 = f"preprocessed_sat_obj_e2_{filename_extension}.json"
        sat_objects_e2_fp = get_file_path(assets_dir, files_sub_dir, sat_obj_filename_e2)

        get_experiment_config_and_run_experiment(
            f_path=experiment_config_path,
            job_id=uid,
            run_serially=False,
            plot_results=True,
            mail_results=True,
            run_existing=False,
            preprocess=False,
            solve_preprocessed=True,
            sat_obj_fp_e1=sat_objects_e1_fp,
            sat_obj_fp_e2=sat_objects_e2_fp
        )

    logger.info("Experiment Run Complete...")