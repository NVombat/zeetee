import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
from pysat.solvers import Solver

from .utils import get_file_path
from .logger import create_logger
from .er_encoder import rgp_to_sat_er
from .mb_encoder import rgp_to_sat_mb
from .mailer import send_mail_with_attachment
from .generator import generate_rgp_instances_with_config
from . import assets_dir, files_sub_dir, results_sub_dir, exp_path, experiment_config_path
from .helper import json_to_dict, json_to_rgp, extract_clauses_and_instance_data, write_to_file

logger = create_logger(l_name="zt_runner")

# Supress default logs for image processing
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


def cactus_plot(times1: list, times2: list, filename: str = "cactus_plot.svg") -> None:
    '''
    Plots a Cactus Plot based on two input arrays that
    contain the individual time taken to solve each
    instance for two different encodings.

    Args:
        times1: List of all the individual execution times of instances for encoding 1
        times2: List of all the individual execution times of instances for encoding 2
        filename: Name of the file where the plot will be saved (default: "cactus_plot.png")

    Returns:
        None: Saves and displays a cactus plot for both encodings
    '''
    # Step 1: Sort the times in ascending order for both encodings
    sorted_times1 = np.sort(times1)
    sorted_times2 = np.sort(times2)

    # Step 2: Compute cumulative time for both encodings
    cumulative_times1 = np.cumsum(sorted_times1)
    cumulative_times2 = np.cumsum(sorted_times2)

    # Step 3: Generate the X-axis (number of solved instances) for both encodings
    instances_solved1 = np.arange(1, len(times1) + 1)
    instances_solved2 = np.arange(1, len(times2) + 1)

    # Step 4: Plot the cactus plot for both encodings
    plt.figure(figsize=(10, 6))
    plt.plot(instances_solved1, cumulative_times1, marker='o', linestyle='-', color='b', label='Encoding 1')
    plt.plot(instances_solved2, cumulative_times2, marker='s', linestyle='-', color='r', label='Encoding 2')

    # Step 5: Labeling and title
    plt.xlabel("Number of Solved Instances")
    plt.ylabel("Total Execution Time (Seconds)")
    plt.title("Cactus Plot of SAT Solver Performance")
    plt.grid(True)
    plt.legend()

    target_dir = assets_dir
    image_dir = results_sub_dir
    image_file_path = get_file_path(target_dir, image_dir, filename)

    logger.debug(f"Image File Path: {image_file_path}")

    # Save the plot
    plt.savefig(image_file_path, format="svg")  # Use "png", "pdf" or "eps" if preferred, set dpi=300 if "png"
    # plt.savefig(image_file_path, format="png", dpi=300)

    # Show the plot
    # plt.show()

    # Close the plot
    plt.close()


def get_experiment_config_and_run_experiment(
    f_path: str = experiment_config_path,
    run_serially: bool = True,
    plot_results: bool = True,
    mail_results: bool = True,
    run_existing: bool = False,
    **kwargs
) -> None:
    '''
    Runs the experiment based on the run_existing Flag. If True, it runs the
    experiment on an existing experiment setup. If False, it generates all
    the instances based on the experiment configuration provided and then
    runs the experiment on the generated instances. If run_existing is True,
    the user needs to provide an additional keyword argument 'existing_fp',
    passing the path of the existing experiment setup [instances]. If run_
    serially is set to False, encoding 1 and encoding 2 are run as separate
    processes in parallel. If plot_results is set to True, a cactus plot of
    the two encodings is plot. If mail_results is set to True, a copy of the
    results is mailed to the user.

    Args:
        f_path: Path to the experiment configuration file
        run_serially: Flag to decide whether to run the experiment serially or in parallel
        plot_results: Flag to decide whether to plot the results or not
        mail_results: Flag to decide whether to email results or not
        run_existing: Flag to decide whether to run an existing experiment or not
        **kwargs['existing_fp']: To provide a file path if run_existing == True
        **kwargs['target_email_addr']: Target Email ID for mail_results

    Returns:
        None: Plots a cactus plot of both the encodings using results
              provided by the SAT Solver and emails the results to the user
    '''
    start_time = time.time()

    if run_existing:
        logger.debug("Running Existing Experiment...")

        if "existing_fp" not in kwargs:
            logger.error("run_existing SET to TRUE: existing_fp Not Provided!")
            sys.exit(1)

        if not os.path.exists(kwargs['existing_fp']):
            logger.error("Invalid Existing Path Provided. Please Provide A Valid Existing File Path")
            sys.exit(1)

        rgp_instances = json_to_rgp(kwargs['existing_fp'])

    else:
        flag, top_id = generate_rgp_instances_with_config(flag=2, experiment_config_path=f_path)

        if not flag:
            logger.error("Instance Generation Issue! Re-Generate Instances Correctly")
            sys.exit(1)

        rgp_instances = json_to_rgp(exp_path)

    if run_serially:
        logger.info("Running Experiment in Serial...")

        e1_res = run_encoding_1(rgp_instances)
        logger.debug(f"Experiment Results [E1]: {e1_res}")

        e2_res = run_encoding_2(rgp_instances)
        logger.debug(f"Experiment Results [E2]: {e2_res}")

    else:
        logger.info("Running Experiment in Parallel...")

        manager = multiprocessing.Manager()
        e1_res = manager.dict()
        e2_res = manager.dict()

        def run_e1():
            result = run_encoding_1(rgp_instances)
            e1_res.update(result)

        def run_e2():
            result = run_encoding_2(rgp_instances)
            e2_res.update(result)

        # Create and start processes for each encoding
        process1 = multiprocessing.Process(target=run_e1)
        process2 = multiprocessing.Process(target=run_e2)

        process1.start()
        process2.start()

        # Wait for both processes to complete
        process1.join()
        process2.join()

    end_time = time.time()

    execution_time = end_time - start_time
    logger.info(f"Experiment Time: {execution_time:.6f} seconds")

    if plot_results:
        cactus_plot(e1_res["instance_solving_time_e1"], e2_res["instance_solving_time_e2"])

    if mail_results:
        if "target_email_addr" not in kwargs:
            logger.debug("Mailing Results to Default Email ID")
            send_mail_with_attachment()

        else:
            logger.debug(f"Mailing Results to {kwargs['target_email_addr']}")
            send_mail_with_attachment(target_email_addr=kwargs['target_email_addr'])

    return


def run_encoding_1(rgp_instances: list) -> dict:
    '''
    Runs the experiment using ENCODING 1 [MB] and stores all
    the necessary data to be plotted and presented later.

    Args:
        rgp_instances: A list of all the instances generated

    Returns:
        dict: A dictionary containing the experiment results
    '''
    experiment_config = json_to_dict(experiment_config_path)
    timeout_limit = experiment_config["timeout"]

    experiment_data = []
    encoding_type = "e1"

    experiment_results = {
        "total_solving_time_e1": 0,
        "total_instances_solved_e1": 0,
        "total_instances_timedout_e1": 0,
        "instance_solving_time_e1": [],
        "instance_data_e1": []
    }

    num_instances_solved = 0
    total_solving_time = 0

    for inst in rgp_instances:
        # Store data for each instance and then insert into a list
        # Create a dataframe from that list
        temp_data = {}

        temp_data["instance_id"] = inst["i"]
        temp_data["encoding_type"] = encoding_type

        res = solve(1, 1, inst, timeout_limit)

        instance_data = res["instance_data"]
        experiment_results["instance_data_e1"].append(instance_data)

        temp_data["num_clauses"] = instance_data["num_clauses"]
        temp_data["num_variables"] = instance_data["num_variables"]
        temp_data["num_literals"] = instance_data ["num_literals"]

        temp_data["solving_time"] = res["tts"]

        if res["status"] != None:
            logger.debug(f"Time To Solve Instance: {res['tts']}")
            logger.debug(f"Instance Status: {res['status']}")

            experiment_results["total_solving_time_e1"] += res["tts"]
            experiment_results["total_instances_solved_e1"] += 1

            experiment_results["instance_solving_time_e1"].append(res["tts"])

            temp_data["status"] = "SLV"

            total_solving_time += res["tts"]
            num_instances_solved += 1

        else:
            logger.debug("Instance TimeOut! Instance Not Solved!")
            experiment_results["total_instances_timedout_e1"] += 1
            temp_data["status"] = "TMO"

        experiment_data.append(temp_data)

    logger.debug(f"[E1] Final Experiment Results: {experiment_results}")

    target_dir = assets_dir
    result_dir = results_sub_dir

    results_file_name = "experiment_results_e1.json"
    results_file_path = get_file_path(target_dir, result_dir, results_file_name)

    write_to_file(experiment_results, results_file_path)

    logger.debug(f"[E1] Final Experiment Data: {experiment_data}")

    df = pd.DataFrame(experiment_data)

    data_file_name_json = "experiment_data_e1.json"
    data_file_name_csv = "experiment_data_e1.csv"

    data_file_path_json = get_file_path(target_dir, result_dir, data_file_name_json)
    data_file_path_csv = get_file_path(target_dir, result_dir, data_file_name_csv)

    df.to_json(data_file_path_json, orient='records', indent=4)
    df.to_csv(data_file_path_csv, index=False)

    return experiment_results


def run_encoding_2(rgp_instances: list) -> dict:
    '''
    Runs the experiment using ENCODING 2 [ER] and stores all
    the necessary data to be plotted and presented later.

    Args:
        rgp_instances: A list of all the instances generated

    Returns:
        dict: A dictionary containing the experiment results
    '''
    experiment_config = json_to_dict(experiment_config_path)
    timeout_limit = experiment_config["timeout"]

    experiment_data = []
    encoding_type = "e2"

    experiment_results = {
        "total_solving_time_e2": 0,
        "total_instances_solved_e2": 0,
        "total_instances_timedout_e2": 0,
        "instance_solving_time_e2": [],
        "instance_data_e2": []
    }

    num_instances_solved = 0
    total_solving_time = 0

    for inst in rgp_instances:
        # Store data for each instance and then insert into a list
        # Create a dataframe from that list
        temp_data = {}

        temp_data["instance_id"] = inst["i"]
        temp_data["encoding_type"] = encoding_type

        res = solve(2, 1, inst, timeout_limit)

        instance_data = res["instance_data"]
        experiment_results["instance_data_e2"].append(instance_data)

        temp_data["num_clauses"] = instance_data["num_clauses"]
        temp_data["num_variables"] = instance_data["num_variables"]
        temp_data["num_literals"] = instance_data ["num_literals"]

        temp_data["solving_time"] = res["tts"]

        if res["status"] != None:
            logger.debug(f"Time To Solve Instance: {res['tts']}")
            logger.debug(f"Instance Status: {res['status']}")

            experiment_results["total_solving_time_e2"] += res["tts"]
            experiment_results["total_instances_solved_e2"] += 1

            experiment_results["instance_solving_time_e2"].append(res["tts"])

            temp_data["status"] = "SLV"

            total_solving_time += res["tts"]
            num_instances_solved += 1

        else:
            logger.debug("Instance TimeOut! Instance Not Solved!")
            experiment_results["total_instances_timedout_e2"] += 1
            temp_data["status"] = "TMO"

        experiment_data.append(temp_data)

    logger.debug(f"[E2] Final Experiment Results: {experiment_results}")

    target_dir = assets_dir
    result_dir = results_sub_dir

    results_file_name = "experiment_results_e2.json"
    results_file_path = get_file_path(target_dir, result_dir, results_file_name)

    write_to_file(experiment_results, results_file_path)

    logger.debug(f"[E2] Final Experiment Data: {experiment_data}")

    df = pd.DataFrame(experiment_data)

    data_file_name_json = "experiment_data_e2.json"
    data_file_name_csv = "experiment_data_e2.csv"

    data_file_path_json = get_file_path(target_dir, result_dir, data_file_name_json)
    data_file_path_csv = get_file_path(target_dir, result_dir, data_file_name_csv)

    df.to_json(data_file_path_json, orient='records', indent=4)
    df.to_csv(data_file_path_csv, index=False)

    return experiment_results


def solve(enc_type: int, solver_flag: int, rgp_instance: dict, timeout: int) -> dict:
    '''
    Takes an encoding type and solver type passed along
    with an RGP instance. It converts the input to a
    solvable form which it then solves using a SAT
    solver

    Args:
        enc_type: [1(Encoding 1 [MB]), 2(Encoding 2 [ER])]
        solver_flag: [1(Cadical195), 2(MapleChrono)]
        rgp_instance: RGP Instance
        timeout: Timeout Limit for Solver [in MilliSeconds]

    Returns:
        dict: Result of the SAT Solver
    '''
    logger.debug(f"RGP Instance: {rgp_instance}")

    try:
        if enc_type not in range(1,3) or solver_flag not in range(1,3):
            raise ValueError

    except ValueError:
        logger.error("Encoding Type must be an integer and must be either 1 or 2. Solver Flag must be an integer and must be either 1 or 2")
        sys.exit(1)

    logger.debug(f"Encoding Type is set to {enc_type}... Solver is set to {solver_flag}...")

    logger.debug("Converting Instance...")

    if enc_type == 1:
        sat_obj = rgp_to_sat_mb(rgp_instance)
        logger.debug(f"SAT Object [MB-ENCODER]: {sat_obj}")

    elif enc_type == 2:
        sat_obj = rgp_to_sat_er(rgp_instance)
        logger.debug(f"SAT Object [ER-ENCODER]: {sat_obj}")

    clauses,instance_data = extract_clauses_and_instance_data(sat_obj)
    logger.debug(f"Clauses: {clauses}")
    logger.debug(f"Instance Data: {instance_data}")

    logger.debug(f"Solving Instance...")

    solvers = ['cadical195', 'maplechrono']

    if solver_flag == 1:
        # CADICAL
        solver_name = solvers[0]

    elif solver_flag == 2:
        # MAPLECHRONO
        solver_name = solvers[1]

    logger.debug(f"Solver: {solver_name}")
    solver = Solver(name=solver_name, bootstrap_with=clauses, use_timer=True, with_proof=True)

    solver_results = call_solver_with_timeout(solver_obj=solver, timeout=timeout/1000)
    solver_results["instance_data"] = instance_data

    # satisfiable = solver.solve()
    # logger.debug(f"Satisfiable: {satisfiable}")

    # elapsed_time = solver.time()
    # logger.debug(f"TTS [TimeToSolve]: {elapsed_time} Seconds")

    # timeout_flag = False

    # if satisfiable is None:
    #     result = None
    #     timeout_flag = True
    #     logger.debug("Solver Timed Out!")

    # elif satisfiable:
    #     result = solver.get_model()

    #     if result:
    #         logger.debug(f"Solution: {result}")

    #     else:
    #         logger.debug("No model could be extracted.")

    # else:
    #     result = solver.get_proof()
    #     # result = solver.get_core()

    #     if result:
    #         logger.debug(f"No satisfiable solution exists. Proof: {result}")

    #     else:
    #         logger.debug("Proof could not be extracted.")

    # logger.debug(f"Accumulated Low Level Stats: {solver.accum_stats() or 'No stats available.'}")

    # res = {}
    # res["status"] = satisfiable
    # res["tts"] = elapsed_time
    # res["result"] = result
    # res["timed_out"] = timeout_flag
    # res["instance_data"] = instance_data

    # return res

    solver.delete()

    return solver_results


def call_solver_with_timeout(solver_obj, timeout) -> dict:
    '''
    Runs the solver with a timeout. If the solving process exceeds the timeout,
    it is terminated, and the result is set to None to indicate a timeout

    Args:
        solver_obj: The solver object that contains the SAT instance to be solved
        timeout: Maximum time (in seconds) to allow the solver to run

    Returns:
        dict: A dictionary containing the solver results
    '''
    result_dict = multiprocessing.Manager().dict()

    def solver_process(solver_obj, result_dict):
        satisfiable = solver_obj.solve()
        logger.debug(f"Satisfiable: {satisfiable}")

        elapsed_time = solver_obj.time()
        logger.debug(f"TTS [TimeToSolve]: {elapsed_time} Seconds")

        result_dict["status"] = satisfiable
        result_dict["tts"] = elapsed_time
        result_dict["result"] = solver_obj.get_model() if satisfiable else solver_obj.get_proof()
        result_dict["timed_out"] = False

    p = multiprocessing.Process(target=solver_process, args=(solver_obj, result_dict))
    p.start()

    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()

        result_dict["status"] = None
        result_dict["tts"] = timeout
        result_dict["result"] = None
        result_dict["timed_out"] = True

        logger.debug("Solver Timed Out!")

    solver_obj.delete()

    return result_dict.copy()


if __name__ == "__main__":
    logger.info("********************RUNNER[LOCAL_TESTING]*********************")

    exp_config_path = experiment_config_path
    logger.info(f"Experiment Configuration Path: {exp_config_path}")

    get_experiment_config_and_run_experiment(
        exp_config_path,
        run_serially=False,
        plot_results=True,
        mail_results=True,
        run_existing=False
    )

    target_dir = assets_dir
    target_subdir = files_sub_dir
    filename = "rgp_gen_exp.json"

    existing_fp = get_file_path(target_dir, target_subdir, filename)
    logger.info(f"Existing File Path: {existing_fp}")

    target_email = "testmail@test.com"

    # get_experiment_config_and_run_experiment(
    #     exp_config_path,
    #     run_serially=False,
    #     plot_results=True,
    #     mail_results=True,
    #     run_existing=True,
    #     existing_fp=existing_fp,
    #     target_email_addr=target_email
    # )