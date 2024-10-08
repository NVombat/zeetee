import sys
from pysat.solvers import Glucose3, Lingeling

from . import *
from .logger import create_logger
from .er_encoder import rgp_to_sat_er
from .mb_encoder import rgp_to_sat_mb
from .helper import json_to_rgp, extract_clauses

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

    logger.info(f"Encoding Type is set to {enc_type}... Solver is set to {solver_flag}...")

    logger.debug("Converting Instance...")

    if enc_type == 1:
        sat_obj = rgp_to_sat_mb(rgp_instance)
        logger.info(f"SAT Object [MB-ENCODER]: {sat_obj}")

    elif enc_type == 2:
        sat_obj = rgp_to_sat_er(rgp_instance)
        logger.info(f"SAT Object [ER-ENCODER]: {sat_obj}")

    res = {}

    if sat_obj["unsatisfiable"]:
        satisfiable = False

    else:
        clauses = extract_clauses(sat_obj)

        logger.debug(f"Solving Instance...")

        # Initialize SAT Solver
        if solver_flag == 1:
            # Glucose3
            logger.debug("Solver: Glucose3")
            solver = Glucose3()

        elif solver_flag == 2:
            # Lingeling
            logger.debug("Solver: Lingeling")
            solver = Lingeling()

        for clause in clauses:
            solver.add_clause(clause)

        satisfiable = solver.solve()
        logger.debug(f"Satisfiable: {satisfiable}")

        if satisfiable:
            logger.debug(f"Solution: {solver.get_model()}")
        else:
            logger.debug("No satisfiable solution exists!")

        solver.delete()

    res["status"] = satisfiable
    res["result"] = []

    return res

if __name__ == "__main__":
    logger.info("********************SOLVER[LOCAL_TESTING]*********************")

    cl_args = sys.argv

    if len(cl_args) != 3:
        logger.error("Usage: python solver.py enc_type solver [enc_type must be an integer and must be either 1 or 2; solver must be an integer and must be either 1 or 2]")
        sys.exit(1)

    logger.debug(f"Command Line Args: {cl_args}")

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
    # rgp_instances = json_to_rgp()
    # rgp_instances = json_to_rgp(pos_jfp)
    # rgp_instances = json_to_rgp(neg_jfp)
    # rgp_instances = json_to_rgp(small_jfp)

    rgp_instances = json_to_rgp(test_path_neg)
    # rgp_instances = json_to_rgp(test_path_pos)

    # Call the SAT Solver on each instance
    for inst in rgp_instances:
        res = solve(enc_type, slv_flag, inst)
        logger.debug(f"Result for RGP Instance: {res}")