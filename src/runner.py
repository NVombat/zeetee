import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pysat.solvers import Solver

from . import *
from .logger import create_logger
from .er_encoder import rgp_to_sat_er
from .mb_encoder import rgp_to_sat_mb
from .generator import generate_rgp_instances
from .helper import json_to_dict, json_to_rgp, extract_clauses, rgp_dict_to_rgp, write_to_file

logger = create_logger(l_name="zt_runner")


def cactus_plot(times: list) -> None:
    '''
    Plots a Cactus Plot based on an input array
    that containts the individual time taken to
    solve each instance

    Args:
        times: List of all the individual execution times of instances

    Returns:
        None: Plots a cactus plot
    '''
    # Step 1: Sort the times in ascending order
    sorted_times = np.sort(times)

    # Step 2: Compute cumulative time
    cumulative_times = np.cumsum(sorted_times)

    # Step 3: Generate the X-axis (number of solved instances)
    instances_solved = np.arange(1, len(times) + 1)

    # Step 4: Plot the cactus plot
    plt.figure(figsize=(10, 6))
    plt.plot(instances_solved, cumulative_times, marker='o', linestyle='-', color='b')

    # Step 5: Labeling and title
    plt.xlabel("Number of Solved Instances")
    plt.ylabel("Total Execution Time (Seconds)")
    plt.title("Cactus Plot of SAT Solver Performance")
    plt.grid(True)

    # Show the plot
    plt.show()


def get_experiment_config_and_run(f_path=experiment_config_path) -> dict:
    '''
    Fetches the experiment configuration stored in a JSON
    file runs the experiment according to it. Stores all
    the necessary data to be plotted and presented later

    Args:
        f_path: Configuration File Path

    Returns:
        dict: A dictionary containing the experiment results
    '''
    experiment_config = json_to_dict(experiment_config_path)

    experiment_results = {
        "total_solving_time_e1": 0,
        "total_instances_solved_e1": 0,
        "total_instances_timedout_e1": 0,
        "instance_solving_time_e1": [],
        "total_solving_time_e2": 0,
        "total_instances_solved_e2": 0,
        "total_instances_timedout_e2": 0,
        "instance_solving_time_e2": []
    }

    num_instances = experiment_config["num_instances"]

    num_resources = experiment_config["num_resources"]
    num_constraints = experiment_config["num_constraints"]
    constraint_size = experiment_config["constraint_size"] # Can be FIXED OR RANDOM?

    timeout_limit = experiment_config["timeout"]

    for nr in num_resources:
        logger.debug(f"[EXPERIMENT_RUNNER]: Number of Resources = {nr}")
        experiment_results_nr = {}

        for nc in num_constraints:
            logger.debug(f"[EXPERIMENT_RUNNER]: Number of Constraints = {nc}")
            experiment_results_nc = {}

            for cs in constraint_size:
                logger.debug(f"[EXPERIMENT_RUNNER]: Constraint Size = {cs}")
                experiment_results_cs = {}

                num_instances_solved = 0
                total_solving_time = 0

                rgp_obj = generate_rgp_instances(flag=2, n=nr, cst_size_type="fixed", n_cst=nc, cst_size=cs, num_instance=num_instances)
                rgp_instances = rgp_dict_to_rgp(rgp_obj)

                for inst in rgp_instances:
                    res = solve(1, 1, inst, timeout_limit)
                    # Do the same for encoding 2? Use Multiprocessing?

                    if res["status"] != None:
                        logger.debug(f"Time To Solve Instance: {res['tts']}")
                        logger.debug(f"Instance Status: {res['status']}")

                        experiment_results["total_solving_time_e1"] += res["tts"]
                        experiment_results["total_instances_solved_e1"] += 1

                        experiment_results["instance_solving_time_e1"].append(res["tts"])

                        total_solving_time += res["tts"]
                        num_instances_solved += 1

                    else:
                        logger.debug("Instance TimeOut! Instance Not Solved!")
                        experiment_results["total_instances_timedout_e1"] += 1

                experiment_results_cs[cs] = (total_solving_time, num_instances_solved)
                logger.debug(f"Experiment Results [Constraint Size = {cs}]: {experiment_results_cs[cs]}")

            experiment_results_nc[nc] = experiment_results_cs
            logger.debug(f"Experiment Results [Number Of Constraints = {nc}]: {experiment_results_nc[nc]}")

        experiment_results_nr[nr] = experiment_results_nc
        logger.debug(f"Experiment Results [Number Of Resources = {nr}]: {experiment_results_nr[nr]}")

    logger.debug(f"Final Experiment Results: {experiment_results}")

    json_file_name = "experiment_results.json"
    json_file_path = get_file_path("logfiles", json_file_name)

    write_to_file(experiment_results, json_file_path)

    cactus_plot(experiment_results["instance_solving_time_e1"])

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

    logger.info(f"Encoding Type is set to {enc_type}... Solver is set to {solver_flag}...")

    logger.debug("Converting Instance...")

    if enc_type == 1:
        sat_obj = rgp_to_sat_mb(rgp_instance)
        logger.info(f"SAT Object [MB-ENCODER]: {sat_obj}")

    elif enc_type == 2:
        sat_obj = rgp_to_sat_er(rgp_instance)
        logger.info(f"SAT Object [ER-ENCODER]: {sat_obj}")

    clauses = extract_clauses(sat_obj)

    logger.debug(f"Solving Instance...")

    solvers = ['cadical195', 'maplechrono']

    if solver_flag == 1:
        # CADICAL
        solver_name = solvers[0]
        logger.debug(f"Solver: {solver_name}")

    elif solver_flag == 2:
        # MAPLECHRONO
        solver_name = solvers[1]
        logger.debug(f"Solver: {solver_name}")

    solver = Solver(name=solver_name, bootstrap_with=clauses, use_timer=True, with_proof=True)
    satisfiable = solver.solve()

    elapsed_time = solver.time()
    logger.debug(f"TTS [TimeToSolve]: {elapsed_time} Seconds")

    logger.debug(f"Satisfiable: {satisfiable}")

    timeout_flag = False

    if satisfiable is None:
        timeout_flag = True
        logger.debug("Solver Timed Out!")

    elif satisfiable:
        result = solver.get_model()

        if result:
            logger.debug(f"Solution: {result}")

        else:
            logger.debug("No model could be extracted.")

    else:
        result = solver.get_proof()

        if result:
            logger.debug(f"No satisfiable solution exists. Proof: {result}")

        else:
            logger.debug("Proof could not be extracted.")

    logger.debug(f"Accumulated Low Level Stats: {solver.accum_stats() or 'No stats available.'}")

    solver.delete()

    res = {}
    res["status"] = satisfiable
    res["tts"] = elapsed_time
    res["result"] = result
    res["timed_out"] = timeout_flag

    return res


if __name__ == "__main__":
    logger.info("********************RUNNER[LOCAL_TESTING]*********************")

    exp_config_path = experiment_config_path
    logger.debug(f"Experiment Configuration Path: {exp_config_path}")

    exp_res = get_experiment_config_and_run(exp_config_path)
    logger.debug(f"Experiment Results: {exp_res}")