import sys
from pysat.solvers import Solver

from . import *
from .logger import create_logger
from .er_encoder import rgp_to_sat_er
from .mb_encoder import rgp_to_sat_mb
from .helper import json_to_rgp, extract_clauses_and_instance_data

logger = create_logger(l_name="zt_solver")


def solve(enc_type: int, solver_flag: int, rgp_instance: dict) -> dict:
    '''
    Takes an encoding type and solver type passed along with
    an RGP instance. It converts the input to a solvable
    form which it then solves using a SAT solver

    Args:
        enc_type: [1(Encoding 1), 2(Encoding 2)]
        solver_flag: [1 (Glucose3), 2(Lingeling)]
        rgp_instance: RGP Instance

    Returns:
        dict: Result of the SAT Solver
    '''
    logger.debug(f"RGP Instance: {inst}")

    try:
        if enc_type not in range(1,3) or solver_flag not in range(1,3):
            raise ValueError

    except ValueError:
        logger.error("Usage: python solver.py enc_type solver [enc_type must be an integer and must be either 1 or 2; solver must be an integer and must be either 1 or 2]")
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

    res = {}

    solvers = ['glucose3', 'lingeling']

    if solver_flag == 1:
        # Glucose3
        solver_name = solvers[0]

    elif solver_flag == 2:
        # Lingeling
        solver_name = solvers[1]

    logger.debug(f"Solver: {solver_name}")
    solver = Solver(name=solver_name, bootstrap_with=clauses, use_timer=True, with_proof=True)

    satisfiable = solver.solve()
    logger.debug(f"Satisfiable: {satisfiable}")

    elapsed_time = solver.time()
    logger.debug(f"TTS [TimeToSolve]: {elapsed_time} Seconds")

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

    res["status"] = satisfiable
    res["tts"] = elapsed_time
    res["result"] = result
    res["timed_out"] = timeout_flag
    res["instance_data"] = instance_data

    solver.delete()

    return res

if __name__ == "__main__":
    logger.info("********************SOLVER[LOCAL_TESTING]*********************")

    cl_args = sys.argv

    if len(cl_args) != 3:
        logger.error("Usage: python solver.py enc_type solver [enc_type must be an integer and must be either 1 or 2; solver must be an integer and must be either 1 or 2]")
        sys.exit(1)

    logger.info(f"Command Line Args: {cl_args}")

    try:
        enc_type = int(cl_args[1])
        slv_flag = int(cl_args[2])

        if enc_type not in range(1,3) or slv_flag not in range(1,3):
            raise ValueError

    except ValueError:
        logger.error("Usage: python solver.py enc_type solver [enc_type must be an integer and must be either 1 or 2; solver must be an integer and must be either 1 or 2]")
        sys.exit(1)

    logger.info(f"Encoding Type is set to {enc_type}... Solver is set to {slv_flag}...")

    # Use Default File Path
    rgp_instances = json_to_rgp()
    # rgp_instances = json_to_rgp(pos_jfp)
    # rgp_instances = json_to_rgp(neg_jfp)
    # rgp_instances = json_to_rgp(small_jfp)

    # rgp_instances = json_to_rgp(test_path_neg)
    # rgp_instances = json_to_rgp(test_path_pos)
    # rgp_instances = json_to_rgp(test_path_ran)

    # Call the SAT Solver on each instance
    for inst in rgp_instances:
        res = solve(enc_type, slv_flag, inst)
        logger.info(f"Result for RGP Instance: {res}")