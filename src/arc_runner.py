from .logger import create_logger
from . import experiment_config_path
from .runner import get_experiment_config_and_run_experiment

logger = create_logger(l_name="zt_arc_runner")

if __name__ == "__main__":
    logger.info("********************ARC_RUNNER[RUNNING EXPERIMENT]*********************")

    logger.info(f"Experiment Configuration Path: {experiment_config_path}")

    get_experiment_config_and_run_experiment(
        experiment_config_path,
        run_serially=False,
        plot_results=True,
        mail_results=True,
        run_existing=False
    )

    logger.info("Experiment Run Complete...")