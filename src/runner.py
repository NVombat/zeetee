import os
import sys
import csv
import json
import time
import signal
import logging
import numpy as np
import pandas as pd
import multiprocessing
from pysat.formula import CNF
import matplotlib.pyplot as plt
from pysat.solvers import Solver

from .utils import get_file_path
from .logger import create_logger
from .er_encoder import rgp_to_sat_er
from .mb_encoder import rgp_to_sat_mb
from .mailer import send_mail_with_attachment
from .generator import generate_rgp_instances_with_config
from .helper import json_to_dict, json_to_rgp, write_to_file
from .preprocessor import preprocess_instances_e1, preprocess_instances_e2
from . import assets_dir, files_sub_dir, results_sub_dir, data_dir, config_sub_dir

logger = create_logger(l_name="zt_runner")

# Supress default logs for image processing
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


def handle_timeout(sig, frame):
    # Timeout Handler()
    raise TimeoutError('Solver Timed Out!')


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
    f_path: str,
    job_id: int = -1,
    run_serially: bool = False,
    plot_results: bool = True,
    mail_results: bool = True,
    run_existing: bool = False,
    preprocess: bool = False,
    solve_preprocessed: bool = False,
    **kwargs
) -> None:
    '''
    Runs the experiment based on multiple flags provided by the user. The run_existing
    flag, if True, runs the experiment on an existing experiment setup. If False, it
    generates all the instances based on the experiment configuration provided and then
    runs the experiment on the generated instances. If run_existing is True, the user
    needs to provide an additional keyword argument 'existing_fp', passing the path of
    the existing experiment setup [instances]. If run_serially is set to False, encoding
    1 and encoding 2 are run as separate processes in parallel. If plot_results is set
    to True, a cactus plot of the two encodings is plot. If mail_results is set to True,
    a copy of the results is mailed to the user. If preprocess is set to True, we store
    all SAT objects in a file to be solved at a later time. If solve_preprocessed is set
    to True, the user needs to provide two additional keyword arguments 'sat_obj_fp_e1'
    and 'sat_obj_fp_e2', thus passing the paths of the stored SAT objects.

    Args:
        f_path: Path to the experiment configuration file
        job_id: SLURM Job ID (Unique Identifier)
        run_serially: Flag to decide whether to run the experiment serially or in parallel
        plot_results: Flag to decide whether to plot the results or not
        mail_results: Flag to decide whether to email results or not
        run_existing: Flag to decide whether to run an existing experiment or not
        preprocess: Flag to decide whether to preprocess and solve simultaneously or not
        solve_preprocessed: Flag to decide whether to solve stored SAT objects or not

    Keyword Args [Optional]:
        **kwargs['existing_fp']: To provide a file path if run_existing == True
        **kwargs['target_email_addr']: Target Email ID for mail_results
        **kwargs['sat_obj_fp_e1']: To provide a file path for E1 SAT Objects if solve_preprocessed == True
        **kwargs['sat_obj_fp_e2']: To provide a file path for E2 SAT Objects if solve_preprocessed == True

    Returns:
        None: Runs the experiment based on the flags provided
    '''
    start_time = time.time()

    if run_existing:
        logger.debug("Running Existing Experiment...")

        try:
            if "existing_fp" not in kwargs:
                logger.error("run_existing SET to TRUE: existing_fp Not Provided!")
                sys.exit(1)

            if not os.path.exists(kwargs['existing_fp']):
                logger.error("Invalid Existing Path Provided. Please Provide A Valid Existing File Path")
                sys.exit(1)

        except KeyError as e:
            logger.error(f"Missing required key in kwargs: {e}")
            sys.exit(1)

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            sys.exit(1)

        except PermissionError as e:
            logger.error(f"Permission error accessing file: {e}")
            sys.exit(1)

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            sys.exit(1)

        rgp_instances = json_to_rgp(kwargs['existing_fp'])

    else:
        flag, top_id = generate_rgp_instances_with_config(flag=2, experiment_config_path=f_path, job_id=job_id)

        if not flag:
            logger.error("Instance Generation Issue! Re-Generate Instances Correctly")
            sys.exit(1)

        gen_instances_filename = f"rgp_gen_exp_{job_id}.json"
        exp_path = get_file_path(assets_dir, files_sub_dir, gen_instances_filename)

        rgp_instances = json_to_rgp(exp_path)

    if solve_preprocessed:
        logger.debug("Solve_Preprocessed FLAG SET to TRUE...")

        try:
            if "sat_obj_fp_e1" not in kwargs or "sat_obj_fp_e2" not in kwargs:
                logger.error("solve_preprocessed SET to TRUE: sat_obj_fp_e1 or sat_obj_fp_e2 Not Provided!")
                sys.exit(1)

            if not os.path.exists(kwargs['sat_obj_fp_e1']) or not os.path.exists(kwargs['sat_obj_fp_e2']):
                logger.error("Invalid SAT object Path Provided. Please Provide A Valid Existing File Path To Stored SAT Objects")
                sys.exit(1)

        except KeyError as e:
            logger.error(f"Missing required key in kwargs: {e}")
            sys.exit(1)

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            sys.exit(1)

        except PermissionError as e:
            logger.error(f"Permission error accessing file: {e}")
            sys.exit(1)

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            sys.exit(1)

        experiment_config = json_to_dict(f_path)
        timeout_limit = experiment_config["timeout"]

    if preprocess:
        logger.debug("Preprocess FLAG SET to TRUE...")

        manager_preprocess = multiprocessing.Manager()
        preprocess_results = manager_preprocess.dict()

        def preprocess_e1():
            sat_obj_fp_e1 = preprocess_instances_e1(rgp_instances=rgp_instances, job_id=job_id)
            preprocess_results['e1'] = sat_obj_fp_e1

        def preprocess_e2():
            sat_obj_fp_e2 = preprocess_instances_e2(rgp_instances=rgp_instances, job_id=job_id)
            preprocess_results['e2'] = sat_obj_fp_e2

        # Create and start processes for each encoding
        process1 = multiprocessing.Process(target=preprocess_e1)
        process2 = multiprocessing.Process(target=preprocess_e2)

        process1.start()
        process2.start()

        # Wait for both processes to complete
        process1.join()
        process2.join()

        # To access the results
        # result_e1 = preprocess_results['e1']
        # result_e2 = preprocess_results['e2']

        logger.info("Preprocessing Complete!")

        return

    if run_serially:
        logger.info("Running Experiment in Serial...")

        if solve_preprocessed:
            e1_res = solve_preprocessed_e1(sat_obj_fp_e1=kwargs['sat_obj_fp_e1'], timeout=timeout_limit, job_id=job_id)
            logger.debug(f"Experiment Results [E1]: {e1_res}")

            e2_res = solve_preprocessed_e2(sat_obj_fp_e2=kwargs['sat_obj_fp_e2'], timeout=timeout_limit, job_id=job_id)
            logger.debug(f"Experiment Results [E2]: {e2_res}")

        else:
            e1_res = run_encoding_1(rgp_instances, experiment_config_path=f_path, job_id=job_id)
            logger.debug(f"Experiment Results [E1]: {e1_res}")

            e2_res = run_encoding_2(rgp_instances, experiment_config_path=f_path, job_id=job_id)
            logger.debug(f"Experiment Results [E2]: {e2_res}")

    else:
        logger.info("Running Experiment in Parallel...")

        manager = multiprocessing.Manager()
        e1_res = manager.dict()
        e2_res = manager.dict()

        if solve_preprocessed:
            def run_e1_preprocessed():
                result = solve_preprocessed_e1(sat_obj_fp_e1=kwargs['sat_obj_fp_e1'], timeout=timeout_limit, job_id=job_id)
                e1_res.update(result)

            def run_e2_preprocessed():
                result = solve_preprocessed_e2(sat_obj_fp_e2=kwargs['sat_obj_fp_e2'], timeout=timeout_limit, job_id=job_id)
                e2_res.update(result)

            # Create and start processes for each encoding
            process1 = multiprocessing.Process(target=run_e1_preprocessed)
            process2 = multiprocessing.Process(target=run_e2_preprocessed)

            process1.start()
            process2.start()

            # Wait for both processes to complete
            process1.join()
            process2.join()

        else:
            def run_e1():
                result = run_encoding_1(rgp_instances, experiment_config_path=f_path, job_id=job_id)
                e1_res.update(result)

            def run_e2():
                result = run_encoding_2(rgp_instances, experiment_config_path=f_path, job_id=job_id)
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
        image_file_name = f"cactus_plot_{job_id}.svg"
        cactus_plot(e1_res["instance_solving_time_e1"], e2_res["instance_solving_time_e2"], image_file_name)

    if mail_results:
        if "target_email_addr" not in kwargs:
            logger.debug("Mailing Results to Default Email ID")
            send_mail_with_attachment()

        else:
            logger.debug(f"Mailing Results to {kwargs['target_email_addr']}")
            send_mail_with_attachment(target_email_addr=kwargs['target_email_addr'])

    return


def run_encoding_1(rgp_instances: list, experiment_config_path: str, job_id: int) -> dict:
    '''
    Runs the experiment using ENCODING 1 [MB] and stores all
    the necessary data to be plotted and presented later.

    Args:
        rgp_instances: A list of all the instances generated
        experiment_config_path: Path to the experiment configuration file
        job_id: SLURM Job ID (Unique Identifier)

    Returns:
        dict: A dictionary containing the experiment results
    '''
    encoding_type = "e1"

    experiment_config = json_to_dict(experiment_config_path)
    timeout_limit = experiment_config["timeout"]

    return solve_and_record_results(
        rgp_instances=rgp_instances,
        encoding_type=encoding_type,
        timeout_limit=timeout_limit,
        job_id=job_id,
        use_cnf = False
    )


def run_encoding_2(rgp_instances: list, experiment_config_path: str, job_id: int) -> dict:
    '''
    Runs the experiment using ENCODING 2 [ER] and stores all
    the necessary data to be plotted and presented later.

    Args:
        rgp_instances: A list of all the instances generated
        experiment_config_path: Path to the experiment configuration file
        job_id: SLURM Job ID (Unique Identifier)

    Returns:
        dict: A dictionary containing the experiment results
    '''
    encoding_type = "e2"

    experiment_config = json_to_dict(experiment_config_path)
    timeout_limit = experiment_config["timeout"]

    return solve_and_record_results(
        rgp_instances=rgp_instances,
        encoding_type=encoding_type,
        timeout_limit=timeout_limit,
        job_id=job_id,
        use_cnf=False
    )


def solve_and_record_results(rgp_instances: list, encoding_type: str, timeout_limit: int, job_id: int, use_cnf : bool) -> dict:
    '''
    Solves all RGP instances of a specific encoding
    and returns the results in a dictionary

    Args:
        rgp_instances: A list of all the instances generated
        encoding_type: Encoding Type
        timeout_limit: Timeout Limit for Solver [in MilliSeconds]
        job_id: SLURM Job ID (Unique Identifier)
        use_cnf: Flag to decide whether to use CNF object or final_clauses

    Returns:
        dict: A dictionary containing the experiment results
    '''
    if not isinstance(encoding_type, str) or encoding_type not in ["e1", "e2"]:
        logger.error(f"Invalid Encoding Type Given: {encoding_type}. Please Specify A Valid Encoding Type (e1, e2)!")
        sys.exit(1)

    experiment_results = {
        f"total_solving_time_{encoding_type}": 0,
        f"total_instances_solved_{encoding_type}": 0,
        f"total_instances_timedout_{encoding_type}": 0,
        f"instance_solving_time_{encoding_type}": [],
        f"instance_data_{encoding_type}": []
    }

    if encoding_type == "e1":
        encoding = 1

    elif encoding_type == "e2":
        encoding=2

    slv_flag = 1

    logger.debug(f"Encoding Set To: {encoding_type} -> {encoding}... Solver Flag Set To: {slv_flag}")

    num_instances_solved = 0
    total_solving_time = 0

    # For experiment logging: To check how many instances have been gone through
    track_id = 0

    # File information
    target_dir = assets_dir
    result_dir = results_sub_dir

    data_file_name_csv = f"experiment_data_{encoding_type}_{job_id}.csv"
    data_file_path_csv = get_file_path(target_dir, result_dir, data_file_name_csv)

    for inst in rgp_instances:
        logger.info(f"[{encoding_type.capitalize()}] Instance {track_id} Started!")

        temp_data = {}

        temp_data["instance_id"] = inst["i"]
        temp_data["encoding_type"] = encoding_type

        # Storing N value in Data Frame for Plotting Instance Data Statistics
        temp_data["N"] = inst["n"]

        res = solve(encoding, slv_flag, inst, timeout_limit, use_cnf)

        logger.info(f"[{encoding_type.capitalize()}] Instance {track_id} Results Received From Solver!")

        instance_data = res["instance_data"]
        experiment_results[f"instance_data_{encoding_type}"].append(instance_data)

        temp_data["num_clauses"] = instance_data["num_clauses"]
        temp_data["num_variables"] = instance_data["num_variables"]
        temp_data["num_literals"] = instance_data ["num_literals"]

        temp_data["solving_time"] = res["tts"]

        if res["status"] != None:
            logger.debug(f"Time To Solve Instance: {res['tts']}")
            logger.debug(f"Instance Status: {res['status']}")

            experiment_results[f"total_solving_time_{encoding_type}"] += res["tts"]
            experiment_results[f"total_instances_solved_{encoding_type}"] += 1

            experiment_results[f"instance_solving_time_{encoding_type}"].append(res["tts"])

            temp_data["status"] = "SLV"

            total_solving_time += res["tts"]
            num_instances_solved += 1

        else:
            logger.debug("Instance TimeOut! Instance Not Solved!")
            experiment_results[f"total_instances_timedout_{encoding_type}"] += 1
            temp_data["status"] = "TMO"

            logger.info(f"[{encoding_type.capitalize()}] Instance {track_id} Timed Out!")

        write_temp_data_to_csv(temp_data, data_file_path_csv)

        logger.info(f"[{encoding_type.capitalize()}] Instance {track_id} Done!")
        track_id += 1

    logger.debug(f"[{encoding_type.capitalize()}] Final Experiment Results: {experiment_results}")

    results_file_name = f"experiment_results_{encoding_type}_{job_id}.json"
    results_file_path = get_file_path(target_dir, result_dir, results_file_name)

    write_to_file(experiment_results, results_file_path)

    return experiment_results


def solve(enc_type: int, solver_flag: int, rgp_instance: dict, timeout: int, use_cnf: bool) -> dict:
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
        use_cnf: Flag to decide whether to use CNF object or final_clauses

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

    logger.info(f"[E{enc_type}] Converting Instance...")

    if enc_type == 1:
        sat_obj = rgp_to_sat_mb(rgp_instance)
        logger.debug(f"SAT Object [MB-ENCODER]: {sat_obj}")

    elif enc_type == 2:
        sat_obj = rgp_to_sat_er(rgp_instance)
        logger.debug(f"SAT Object [ER-ENCODER]: {sat_obj}")

    logger.info(f"[E{enc_type}] Instance Converted To SAT Object Successfully...")

    if use_cnf:
        logger.info(f"[E{enc_type}] Using CNF Object")
        clauses = sat_obj["cnf_object"]
        logger.info(f"[E{enc_type}] CNF Size: {len(clauses.clauses)} clauses")

    else:
        logger.info(f"[E{enc_type}] Using Final_Clauses")
        clauses = sat_obj["final_clauses"]
        logger.debug(f"Clauses: {clauses}")

    instance_data = sat_obj["instance_data"]
    logger.debug(f"Instance Data: {instance_data}")

    return solve_with_timeout(
        enc_type=enc_type,
        solver_flag=solver_flag,
        clauses=clauses,
        instance_data=instance_data,
        timeout=timeout)


def solve_with_timeout(
    enc_type: str,
    solver_flag: int,
    clauses: list | CNF,
    instance_data: dict,
    timeout: int,
) -> dict:
    '''
    Solve an RGP instance, using the generated CNF
    object, within a specific Timeout Limit

    Args:
        enc_type: [1(Encoding 1 [MB]), 2(Encoding 2 [ER])]
        solver_flag: [1(Cadical195), 2(MapleChrono)]
        clauses: Clauses in the form of a list of cnf object
        instance_data: Dictionary containing Instance Data
        timeout: Timeout Limit for Solver [in MilliSeconds]

    Returns:
        dict: Result of the SAT Solver
    '''
    logger.info(f"[E{enc_type}] Solving Instance...")

    solvers = ['cadical195', 'maplechrono']

    if solver_flag == 1:
        # CADICAL
        solver_name = solvers[0]

    elif solver_flag == 2:
        # MAPLECHRONO
        solver_name = solvers[1]

    logger.debug(f"Solver: {solver_name}")
    solver = Solver(name=solver_name, bootstrap_with=clauses, use_timer=True, with_proof=True)

    logger.info(f"[E{enc_type}] Setting Timeout")

    # Register the signal function handler
    signal.signal(signal.SIGALRM, handle_timeout)
    # Define a timeout for your function
    signal.alarm(int(timeout/1000))

    timeout_flag = False

    res = {}

    try:
        logger.info(f"[E{enc_type}] Calling Solver...")

        start_time = time.time()
        satisfiable = solver.solve()
        end_time = time.time()

        logger.info(f"[E{enc_type}] Satisfiable: {satisfiable}")

        elapsed_time_using_solver = solver.time()
        logger.debug(f"Elapsed Time (BuiltIn PySAT Method .time()): {elapsed_time_using_solver:.5f} Seconds")

        elapsed_time = end_time - start_time
        logger.debug(f"Elapsed Time: {elapsed_time:.5f} Seconds")

        tts = elapsed_time

        if satisfiable:
            result = solver.get_model()

            if result:
                logger.debug(f"Solution: {result}")

            else:
                logger.debug("No model could be extracted.")

        else:
            result = solver.get_proof()
            # result = solver.get_core()

            if result:
                logger.debug(f"No satisfiable solution exists. Proof: {result}")

            else:
                logger.debug("Proof could not be extracted.")

        logger.info(f"[E{enc_type}] Solver statistics: {solver.accum_stats() or 'No stats available'}")

    except TimeoutError:
        logger.error("Solver Timed Out!")
        satisfiable = None
        timeout_flag = True
        result = None
        tts = timeout/1000

    except Exception as e:
        logger.error(f"[E{enc_type}] Solver Error: {e}")

    finally:
        # Reset the alarm (cancel the timeout)
        # signal.signal(signal.SIGALRM, signal.SIG_IGN)
        signal.alarm(0)
        solver.delete()

    res.update({
        "status": satisfiable,
        "tts": tts,
        "result": result,
        "timed_out": timeout_flag,
        "instance_data": instance_data
    })

    logger.info(f"[E{enc_type}] Instance Solved...")

    return res

    # solver_results = call_solver_with_timeout(solver_obj=solver, timeout=timeout/1000)
    # solver_results["instance_data"] = instance_data

    # return solver_results


def write_temp_data_to_csv(temp_data: dict, csv_file_path: str) -> None:
    """
    Write a single temp_data record to a CSV file.

    Args:
        temp_data: Dictionary containing instance results
        csv_file_path: Path to the CSV file

    Returns:
        None
    """
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=temp_data.keys())

        if not file_exists:
            # Write header only once
            logger.debug("Writing Headers...")
            writer.writeheader()

        writer.writerow(temp_data)

    return


def solve_preprocessed_e1(sat_obj_fp_e1: str, timeout: int, job_id: int) -> dict:
    '''
    Solves preprocessed SAT objects stored in a JSON file

    Args:
        sat_obj_fp: File path to SAT objects produced by encoding 1
        timeout: Timeout Limit for Solver [in MilliSeconds]
        job_id: SLURM Job ID (Unique Identifier)

    Returns:
        dict: A dictionary containing the experiment results
    '''
    sat_objects_e1 = json_to_dict(sat_obj_fp_e1)
    encoding_type = "e1"

    return solve_and_record_results_preprocessed(
        sat_objects=sat_objects_e1,
        encoding_type=encoding_type,
        timeout_limit=timeout,
        job_id=job_id
    )


def solve_preprocessed_e2(sat_obj_fp_e2: str, timeout: int, job_id: int):
    '''
    Solves preprocessed SAT objects stored in a JSON file

    Args:
        sat_obj_fp: File path to SAT objects produced by encoding 2
        timeout: Timeout Limit for Solver [in MilliSeconds]
        job_id: SLURM Job ID (Unique Identifier)

    Returns:
        dict: A dictionary containing the experiment results
    '''
    sat_objects_e2 = json_to_dict(sat_obj_fp_e2)
    encoding_type = "e2"

    return solve_and_record_results_preprocessed(
        sat_objects=sat_objects_e2,
        encoding_type=encoding_type,
        timeout_limit=timeout,
        job_id=job_id
    )


def solve_and_record_results_preprocessed(sat_objects: dict, encoding_type: str, timeout_limit: int, job_id: int) -> dict:
    '''
    Solves all SAT instances of a specific encoding
    and returns the results in a dictionary

    Args:
        sat_objects: A dictionary of all the SAT Objects
        encoding_type: Encoding Type
        timeout_limit: Timeout Limit for Solver [in MilliSeconds]
        job_id: SLURM Job ID (Unique Identifier)

    Returns:
        dict: A dictionary containing the experiment results
    '''
    if not isinstance(encoding_type, str) or encoding_type not in ["e1", "e2"]:
        logger.error(f"Invalid Encoding Type Given: {encoding_type}. Please Specify A Valid Encoding Type (e1, e2)!")
        sys.exit(1)

    num_sat_objects = len(sat_objects)
    logger.debug(f"Number Of SAT Objects: {num_sat_objects}")

    slv_flag = 1

    experiment_results = {
        f"total_solving_time_{encoding_type}": 0,
        f"total_instances_solved_{encoding_type}": 0,
        f"total_instances_timedout_{encoding_type}": 0,
        f"instance_solving_time_{encoding_type}": [],
        f"instance_data_{encoding_type}": []
    }

    num_instances_solved = 0
    total_solving_time = 0

    # For experiment logging: To check how many instances have been gone through
    track_id = 0

    # File Information
    target_dir = assets_dir
    result_dir = results_sub_dir

    data_file_name_csv = f"experiment_data_preprocessed_{encoding_type}_{job_id}.csv"
    data_file_path_csv = get_file_path(target_dir, result_dir, data_file_name_csv)

    for i in range (0, num_sat_objects):
        dict_index = str(i)

        temp_data = {}

        temp_data["instance_id"] = sat_objects[dict_index]["i"]
        temp_data["encoding_type"] = encoding_type

        # Storing N value in Data Frame for Plotting Instance Data Statistics
        temp_data["N"] = sat_objects[dict_index]["N"]

        instance_data = sat_objects[dict_index]["instance_data"]
        experiment_results[f"instance_data_{encoding_type}"].append(instance_data)

        temp_data["num_clauses"] = instance_data["num_clauses"]
        temp_data["num_variables"] = instance_data["num_variables"]
        temp_data["num_literals"] = instance_data ["num_literals"]

        clauses = sat_objects[dict_index]["clauses"]

        res = solve_with_timeout(
            enc_type=encoding_type,
            solver_flag=slv_flag,
            clauses=clauses,
            instance_data=instance_data,
            timeout=timeout_limit
        )

        temp_data["solving_time"] = res["tts"]

        if res["status"] != None:
            logger.debug(f"Time To Solve Instance: {res['tts']}")
            logger.debug(f"Instance Status: {res['status']}")

            experiment_results[f"total_solving_time_{encoding_type}"] += res["tts"]
            experiment_results[f"total_instances_solved_{encoding_type}"] += 1

            experiment_results[f"instance_solving_time_{encoding_type}"].append(res["tts"])

            temp_data["status"] = "SLV"

            total_solving_time += res["tts"]
            num_instances_solved += 1

        else:
            logger.debug("Instance TimeOut! Instance Not Solved!")
            experiment_results[f"total_instances_timedout_{encoding_type}"] += 1
            temp_data["status"] = "TMO"

            logger.info(f"[{encoding_type.capitalize()}] Instance {track_id} Timed Out!")

        write_temp_data_to_csv(temp_data, data_file_path_csv)

        logger.info(f"[{encoding_type.capitalize()}] Instance {track_id} Done!")
        track_id += 1

    logger.debug(f"[{encoding_type.capitalize()}] Final Experiment Results: {experiment_results}")

    results_file_name = f"experiment_results_preprocessed_{encoding_type}_{job_id}.json"
    results_file_path = get_file_path(target_dir, result_dir, results_file_name)

    write_to_file(experiment_results, results_file_path)

    return experiment_results


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

    config_filename = "experiment_config_N10.json"
    exp_config_path = get_file_path(data_dir, config_sub_dir, config_filename)
    logger.info(f"Experiment Configuration Path: {exp_config_path}")

    existing_filename = "rgp_gen_exp_-1.json"
    existing_fp = get_file_path(assets_dir, files_sub_dir, existing_filename)
    logger.info(f"Existing File Path: {existing_fp}")

    sat_obj_filename_e1 = "preprocessed_sat_obj_e1_N10.json"
    sat_objects_e1_fp = get_file_path(assets_dir, files_sub_dir, sat_obj_filename_e1)

    sat_obj_filename_e2 = "preprocessed_sat_obj_e2_N10.json"
    sat_objects_e2_fp = get_file_path(assets_dir, files_sub_dir, sat_obj_filename_e2)

    target_email = "testmail@test.com"

    get_experiment_config_and_run_experiment(
        f_path=exp_config_path,
        job_id=-1,
        run_serially=False,
        plot_results=True,
        mail_results=True,
        run_existing=False,
        preprocess=False,
        solve_preprocessed=False,
        # existing_fp=existing_fp,
        # target_email_addr=target_email,
        # sat_obj_fp_e1=sat_objects_e1_fp,
        # sat_obj_fp_e2=sat_objects_e2_fp
    )