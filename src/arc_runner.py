import os
import sys

from .utils import get_file_path
from .logger import create_logger
from . import data_dir, config_sub_dir
from .runner import get_experiment_config_and_run_experiment

logger = create_logger(l_name="zt_arc_runner")

if __name__ == "__main__":
    logger.info("********************ARC_RUNNER[RUNNING EXPERIMENT]*********************")

    config_filename_extension = sys.argv[1]

    if config_filename_extension == "DNE": # Does Not Exist
        logger.error("No Extension Passed! Please Provide A Valid Extension")
        sys.exit(1)

    else:
        exp_config_filename = f"experiment_config_{config_filename_extension}.json"
        experiment_config_path = get_file_path(data_dir, config_sub_dir, exp_config_filename)

        if not os.path.exists(experiment_config_path):
            logger.error("Invalid Configuration Path. Please Provide A Valid Extension")
            sys.exit(1)

    logger.info(f"Experiment Configuration Path: {experiment_config_path}")

    job_id = int(os.environ.get('SLURM_JOB_ID', -1))
    logger.info(f"JOB ID: {job_id}")

    get_experiment_config_and_run_experiment(
        f_path=experiment_config_path,
        job_id=job_id,
        run_serially=False,
        plot_results=True,
        mail_results=True,
        run_existing=False
    )

    logger.info("Experiment Run Complete...")