import os
import sys
from pysat.formula import CNF
from pysat.card import CardEnc, EncType, NoSuchEncodingError

from . import *
from .logger import create_logger
from .helper import get_key_by_value, json_to_rgp

logger = create_logger(l_name="zt_er_encoder")


def rgp_to_sat_er(rgp_obj: dict, clause_verbosity: int = 1, save_cnf_to_file: bool = False, **kwargs) -> dict:
    '''
    Takes an RGP dictionary object and returns a SAT
    dictionary object w.r.t the second encoding which
    is a binary equivalence relation

    Args:
        rgp_obj: RGP dictionary object
        clause_verbosity: Flag to decide amount of detail in SAT Object for Clauses [1: Low, 2: High]
        save_cnf_to_file: Flag to decide if CNF Object should be saved to File
        **kwargs['output_cnf_fp']: File Path to store CNF Object

    Returns:
        dict: SAT dictionary object
    '''
    if not isinstance(clause_verbosity, int) or clause_verbosity not in range(1, 3):
        logger.error("Clause Verbosity Needs To Be An Integer = [1,2]! Please Provide a Valid Input")
        sys.exit(1)

    if save_cnf_to_file:
        if "output_cnf_fp" not in kwargs:
            logger.error("save_cnf_to_file SET to TRUE: output_cnf_fp Not Provided!")
            sys.exit(1)

        if os.path.exists(kwargs['output_cnf_fp']):
            logger.error("Existing Path Provided. Please Provide A Path That Does Not Already Exist")
            sys.exit(1)

    sat_obj = {}

    literal_mappings = {} # only for resources (xij)
    literal_mappings_uc = {} # usability constraints (yrp)
    literal_mappings_sc = {} # security constraints (yrp)

    literals = []
    clauses = {}

    # To gather instance data
    final_clauses = []
    total_num_literals = 0

    # Create a CNF object
    cnf = CNF()

    n = rgp_obj["n"]
    logger.debug(f"Number of Resources: {n}")

    lit_id = 1  # To get unique literals
    for i in range (1, n+1): # for every resource i
        literal_map = {}

        for j in range (i+1, n+1): # for every j > i
            logger.debug(f"(Resource[i], Resource[j]): ({i},{j})")
            val = (i,j) # i,j (placeholders for ith and jth resources)
            literal_map[lit_id] = val
            lit_id = lit_id+1
        literal_mappings[i] = literal_map

    sat_obj["literal_mappings"] = literal_mappings

    uc = rgp_obj["uc"]
    logger.debug(f"Usability Constraints: {uc}")

    # Ensure correctly formatted UC, create boolean variables y[r,p] and get size and "b" values of each clause
    clause_cnt_uc = 1
    clause_size_uc = {}
    b_val_uc = {}

    lit_map_uc_index = 1

    for c1 in uc:
        logger.debug(f"Constraint: {c1}")
        # create a literal for every constraint and for every resource within that constraint
        lr = len(c1[0])

        # store clause size and "b" values
        clause_size_uc[clause_cnt_uc] = lr
        b_val_uc[clause_cnt_uc] = c1[-1]
        clause_cnt_uc = clause_cnt_uc+1

        literal_map = {}
        for i in range (1, lr+1): # for every resource within a constraint
            val = (lit_map_uc_index,i) # val = [r,p]
            literal_map[lit_id] = val
            lit_id = lit_id+1

        logger.debug(f"Literal Map: {literal_map}")

        literal_mappings_uc[lit_map_uc_index] = literal_map # stores literal mappings of uc
        lit_map_uc_index = lit_map_uc_index+1

    sat_obj["literal_mappings_uc"] = literal_mappings_uc
    sat_obj["clause_size_uc"] = clause_size_uc
    sat_obj["b_val_uc"] = b_val_uc

    sc = rgp_obj["sc"]
    logger.debug(f"Security Constraints: {sc}")

    # Ensure correctly formatted SC, create boolean variables y[r,p] and get size and "b" values of each clause
    clause_cnt_sc = 1
    clause_size_sc = {}
    b_val_sc = {}

    lit_map_sc_index = 1

    for c2 in sc:
        logger.debug(f"Constraint: {c2}")
        # create a literal for every constraint and for every resource within that constraint
        lr = len(c2[0])

        # store clause size and "b" values
        clause_size_sc[clause_cnt_sc] = lr
        b_val_sc[clause_cnt_sc] = c2[-1]
        clause_cnt_sc = clause_cnt_sc+1

        literal_map = {}
        for i in range (1, lr+1): # for every resource within a constraint
            val = (lit_map_sc_index,i) # val = [r,p]
            literal_map[lit_id] = val
            lit_id = lit_id+1

        logger.debug(f"Literal Map: {literal_map}")

        literal_mappings_sc[lit_map_sc_index] = literal_map # stores literal mappings of sc
        lit_map_sc_index = lit_map_sc_index+1

    sat_obj["literal_mappings_sc"] = literal_mappings_sc
    sat_obj["clause_size_sc"] = clause_size_sc
    sat_obj["b_val_sc"] = b_val_sc

    # Get number of constraints
    num_uc = len(uc)
    sat_obj["num_uc"] = num_uc

    num_sc = len(sc)
    sat_obj["num_sc"] = num_sc

    m = num_uc + num_sc
    logger.debug(f"Total Constraints (m): {m}")

    sat_obj["total_constraints"] = m

    # Keeps track of updated literal count
    lit_cnt = lit_id-1

    outer_lits = literal_mappings.keys() # i values
    logger.debug(f"Outer Literals: {outer_lits}")

    for lit in outer_lits:
        inner_lits = literal_mappings[lit] # mapping of literals to their (i,j) equivalents
        logger.debug(f"Inner Literals: {inner_lits}")

        inner_lits_elements = list(inner_lits.keys()) # actual literals
        logger.debug(f"Inner Literal Elements: {inner_lits_elements}")

        literals += inner_lits_elements # store all literals in a list

    outer_lits_uc = literal_mappings_uc.keys()
    logger.debug(f"[UC] Outer Literals: {outer_lits_uc}")

    y_lits_uc = []
    for lit in outer_lits_uc:
        inner_lits_uc = literal_mappings_uc[lit] # mapping of literals to their (r,p) equivalents
        logger.debug(f"[UC] Inner Literals: {inner_lits_uc}")

        inner_lits_elements_uc = list(inner_lits_uc.keys()) # actual literals in uc
        logger.debug(f"[UC] Inner Literal Elements: {inner_lits_elements_uc}")

        literals += inner_lits_elements_uc # store all uc literals in literal list
        y_lits_uc += inner_lits_elements_uc # store all y vals for uc separately

    outer_lits_sc = literal_mappings_sc.keys()
    logger.debug(f"[SC] Outer Literals: {outer_lits_sc}")

    y_lits_sc = []
    for lit in outer_lits_sc:
        inner_lits_sc = literal_mappings_sc[lit] # mapping of literals to their (r,p) equivalents
        logger.debug(f"[SC] Inner Literals: {inner_lits_sc}")

        inner_lits_elements_sc = list(inner_lits_sc.keys()) # actual literals in sc
        logger.debug(f"[SC] Inner Literal Elements: {inner_lits_elements_sc}")

        literals += inner_lits_elements_sc # store all sc literals in a list
        y_lits_sc += inner_lits_elements_sc # store all y vals for uc separately

    sat_obj["y_lits_uc"] = y_lits_uc
    sat_obj["y_lits_sc"] = y_lits_sc

    # Encode regular clauses (1) -> (¬xi,j ∨ ¬xj,k ∨ xi,k) ∧ (¬xi,j ∨ xj,k ∨ ¬xi,k) ∧ (xi,j ∨ ¬xj,k ∨ ¬xi,k)
    logger.debug("********************CLAUSE(1)********************")
    transitivity_clauses = []

    for i in range (1, n+1):
        temp_row_i = literal_mappings[i]
        logger.debug(f"TEMP ROW I: {temp_row_i}")

        for j in range (i+1, n+1):
            temp_row_j = literal_mappings[j]
            logger.debug(f"TEMP ROW J: {temp_row_j}")

            for k in range (j+1, n+1):
                '''
                Aim is to get:
                - literal @ (i,j)
                - literal @ (j,k)
                - literal @ (i,k)
                '''
                xij = get_key_by_value(temp_row_i, (i,j))
                assert xij is not None, logger.error("X[i,j] should not be None")
                xjk = get_key_by_value(temp_row_j, (j,k))
                assert xjk is not None, logger.error("X[j,k] should not be None")
                xik = get_key_by_value(temp_row_i, (i,k))
                assert xik is not None, logger.error("X[i,k] should not be None")

                logger.debug(f"X[i,j], X[j,k], X[i,k] : {xij}, {xjk}, {xik}")

                tc_1 = [-xij, -xjk, xik]
                tc_2 = [-xij, xjk, -xik]
                tc_3 = [xij, -xjk, -xik]

                if clause_verbosity > 1:
                    transitivity_clauses.append(tc_1)
                    transitivity_clauses.append(tc_2)
                    transitivity_clauses.append(tc_3)

                final_clauses.append(tc_1)
                final_clauses.append(tc_2)
                final_clauses.append(tc_3)

                total_num_literals += len(tc_1)
                total_num_literals += len(tc_2)
                total_num_literals += len(tc_3)

    if clause_verbosity > 1:
        logger.debug(f"Transitivity Clauses: {transitivity_clauses}")
        clauses["transitivity_clauses"] = transitivity_clauses

    # Encode UC and SC clause (2)
    logger.debug("********************CLAUSE(2)********************")
    y_val_clauses = []

    for r in range (1, num_uc+1):
        # For every UC
        temp_row_r = literal_mappings_uc[r]
        logger.debug(f"[UC] TEMP ROW R: {temp_row_r}")

        lit = get_key_by_value(temp_row_r, (r,1))
        logger.debug(f"Literal Set To True: {lit}")

        if clause_verbosity > 1:
            y_val_clauses.append([lit])

        final_clauses.append([lit])
        total_num_literals += 1

    for r in range (1, num_sc+1):
        # For every SC
        temp_row_r = literal_mappings_sc[r]
        logger.debug(f"[SC] TEMP ROW R: {temp_row_r}")

        lit = get_key_by_value(temp_row_r, (r,1))
        logger.debug(f"Literal Set To True: {lit}")

        if clause_verbosity > 1:
            y_val_clauses.append([lit])

        final_clauses.append([lit])
        total_num_literals += 1

    if clause_verbosity > 1:
        logger.debug(f"Y Value Clauses: {y_val_clauses}")
        clauses["y_val_clauses"] = y_val_clauses

    # Encode UC and SC clause (3) - ((1∨2∨3)∨4)∧(¬1∧¬2∧¬3→4) = (1∨2∨3∨4)∧(¬1∨4)∧(¬2∨4)∧(¬3∨4)
    logger.debug("********************CLAUSE(3)********************")
    ineq_clauses = []

    for r in range (1, num_uc+1):
        temp_row_r = literal_mappings_uc[r]
        logger.debug(f"[UC] TEMP ROW R: {temp_row_r}")

        p_lim = clause_size_uc[r]
        logger.debug(f"[UC] Size of Clause R: {p_lim}")

        uc_r = uc[r-1][0] # Resources in UC
        logger.debug(f"[UC] Constraint: {uc_r}")

        for p in range (2, p_lim+1): # p > 1
            yrp = get_key_by_value(temp_row_r, (r,p))
            logger.debug(f"Y[{r},{p}]: {yrp}")

            tc = []
            tc.append(yrp)

            p_val = uc_r[p-1]
            logger.debug(f"[UC] P Value: {p_val}")

            for q in range (1, p): # q is from 1 to p-1
                q_val = uc_r[q-1]
                logger.debug(f"[UC] Q Value: {q_val}")

                temp_row_q = literal_mappings[q_val]
                logger.debug(f"[UC] TEMP ROW Q: {temp_row_q}")

                xqp = get_key_by_value(temp_row_q, (q_val,p_val))
                logger.debug(f"X[{q_val},{p_val}]: {xqp}")
                tc.append(xqp)

            logger.debug(f"[UC] Temp Clause: {tc}")

            if clause_verbosity > 1:
                ineq_clauses.append(tc)
                logger.debug(f"Ineq Clauses: {ineq_clauses}")

            final_clauses.append(tc)
            total_num_literals += len(tc)

    for r in range (1, num_sc+1):
        temp_row_r = literal_mappings_sc[r]
        logger.debug(f"[SC] TEMP ROW R: {temp_row_r}")

        p_lim = clause_size_sc[r]
        logger.debug(f"[SC] Size of Clause R: {p_lim}")

        sc_r = sc[r-1][0] # Resources in SC
        logger.debug(f"[SC] Constraint: {sc_r}")

        for p in range (2, p_lim+1): # p > 1
            yrp = get_key_by_value(temp_row_r, (r,p))
            logger.debug(f"Y[{r},{p}]: {yrp}")

            tc = []
            tc.append(yrp)

            p_val = sc_r[p-1]
            logger.debug(f"[SC] P Value: {p_val}")

            for q in range (1, p): # q is from 1 to p-1
                q_val = sc_r[q-1]
                logger.debug(f"[SC] Q Value: {q_val}")

                temp_row_q = literal_mappings[q_val]
                logger.debug(f"[SC] TEMP ROW Q: {temp_row_q}")

                xqp = get_key_by_value(temp_row_q, (q_val,p_val))
                logger.debug(f"X[{q_val},{p_val}]: {xqp}")
                tc.append(xqp)

            logger.debug(f"[SC] Temp Clause: {tc}")

            if clause_verbosity > 1:
                ineq_clauses.append(tc)
                logger.debug(f"Ineq Clauses: {ineq_clauses}")

            final_clauses.append(tc)
            total_num_literals += len(tc)

    if clause_verbosity > 1:
        logger.debug(f"Ineq Clauses: {ineq_clauses}")
        clauses["ineq_clauses"] = ineq_clauses

    # Encode UC and SC clause (4)
    logger.debug("********************CLAUSE(4)********************")
    equiv_clauses = []

    for r in range (1, num_uc+1):
        temp_row_r = literal_mappings_uc[r]
        logger.debug(f"[UC] TEMP ROW R: {temp_row_r}")

        p_lim = clause_size_uc[r]
        logger.debug(f"[UC] Size of Clause R: {p_lim}")

        uc_r = uc[r-1][0] # Resources in constraint
        logger.debug(f"[UC] Constraint: {uc_r}")

        for p in range (2, p_lim+1): # p > 1
            yrp = get_key_by_value(temp_row_r, (r,p))
            logger.debug(f"Y[{r},{p}]: {yrp}")

            p_val = uc_r[p-1]
            logger.debug(f"[UC] P Value: {p_val}")

            for q in range (1, p): # q is from 1 to p-1
                tc = []
                tc.append(-yrp)

                q_val = uc_r[q-1]
                logger.debug(f"[UC] Q Value: {q_val}")

                temp_row_q = literal_mappings[q_val]
                logger.debug(f"[UC] TEMP ROW Q: {temp_row_q}")

                xqp = get_key_by_value(temp_row_q, (q_val,p_val))
                logger.debug(f"X[{q_val},{p_val}]: {xqp}")

                tc.append(-xqp)
                logger.debug(f"[UC] Temp Clause: {tc}")

                if clause_verbosity > 1:
                    equiv_clauses.append(tc)
                    logger.debug(f"Equiv Clauses: {equiv_clauses}")

                final_clauses.append(tc)
                total_num_literals += len(tc)

    for r in range (1, num_sc+1):
        temp_row_r = literal_mappings_sc[r]
        logger.debug(f"[SC] TEMP ROW R: {temp_row_r}")

        p_lim = clause_size_sc[r]
        logger.debug(f"[SC] Size of Clause R: {p_lim}")

        sc_r = sc[r-1][0] # Resources in SC
        logger.debug(f"[SC] Constraint: {sc_r}")

        for p in range (2, p_lim+1): # p > 1
            yrp = get_key_by_value(temp_row_r, (r,p))
            logger.debug(f"Y[{r},{p}]: {yrp}")

            p_val = sc_r[p-1]
            logger.debug(f"[SC] P Value: {p_val}")

            for q in range (1, p): # q is from 1 to p-1
                tc = []
                tc.append(-yrp)

                q_val = sc_r[q-1]
                logger.debug(f"[SC] Q Value: {q_val}")

                temp_row_q = literal_mappings[q_val]
                logger.debug(f"[SC] TEMP ROW Q: {temp_row_q}")

                xqp = get_key_by_value(temp_row_q, (q_val,p_val))
                logger.debug(f"X[{q_val},{p_val}]: {xqp}")

                tc.append(-xqp)
                logger.debug(f"[SC] Temp Clause: {tc}")

                if clause_verbosity > 1:
                    equiv_clauses.append(tc)
                    logger.debug(f"Equiv Clauses: {equiv_clauses}")

                final_clauses.append(tc)
                total_num_literals += len(tc)

    if clause_verbosity > 1:
        logger.debug(f"Equiv Clauses: {equiv_clauses}")
        clauses["equiv_clauses"] = equiv_clauses

    # Encode UC clause (5) [ATMOST]
    logger.debug("********************CLAUSE(5)********************")
    clause_builder_card = []
    atmost_card_clauses = []

    # Cardinality Constraint <= b_val_uc[r]
    for r in range (1, num_uc+1):
        temp_row_r = literal_mappings_uc[r]
        logger.debug(f"[UC] TEMP ROW R: {temp_row_r}")

        # These are the cardinality literals - Do we needs an additional loop?
        inner_lits_elements_uc = list(temp_row_r.keys())
        logger.debug(f"[UC] Y Values: {inner_lits_elements_uc}")

        card_literals_uc = []

        b_val = b_val_uc[r]
        logger.debug(f"[UC] B Value: {b_val}")

        for p in range(1, clause_size_uc[r]+1):
            yrp = get_key_by_value(temp_row_r, (r,p))
            logger.debug(f"Y[{r},{p}]: {yrp}")
            card_literals_uc.append(yrp)

        logger.debug(f"[UC] Card Literals: {card_literals_uc}")

        if b_val < len(card_literals_uc):
            cnf_uc = CardEnc.atmost(lits=card_literals_uc, bound=b_val, top_id=lit_cnt, encoding=EncType.seqcounter) # default encoding
            cnf_clauses = cnf_uc.clauses
            logger.debug(f"[UC] Y[{r},{p}] Clauses: {cnf_clauses}")

            # Updating literal count
            if cnf_uc.nv != 0:
                lit_cnt = cnf_uc.nv

            logger.debug(f"UPDATED LITERAL COUNT: {lit_cnt}")

        else:
            cnf_clauses = []

        clause_builder_card.append(cnf_clauses)

    logger.debug(f"[UC] Atmost Clauses: {clause_builder_card}")

    logger.debug(f"Removing One Layer Of Lists...")
    # Remove one layer of lists
    for cl in clause_builder_card:
        # logger.debug(cl)
        for c in cl:
            # logger.debug(c)
            if clause_verbosity > 1:
                atmost_card_clauses.append(c)

            final_clauses.append(c)
            total_num_literals += len(c)

    if clause_verbosity > 1:
        logger.debug(f"[UC] Final Atmost Clauses: {atmost_card_clauses}")
        clauses["atmost_clauses"] = atmost_card_clauses

    # Encode UC and SC clause (6)
    logger.debug("********************CLAUSE(6)********************")
    clause_builder_card = []
    atleast_card_clauses = []

    for r in range (1, num_sc+1):
        temp_row_r = literal_mappings_sc[r]
        logger.debug(f"[SC] TEMP ROW R: {temp_row_r}")

        # These are the cardinality literals - Do we needs an additional loop?
        inner_lits_elements_sc = list(temp_row_r.keys())
        logger.debug(f"[SC] Y Values: {inner_lits_elements_sc}")

        card_literals_sc = []

        b_val = b_val_sc[r]
        logger.debug(f"[SC] B Value: {b_val}")

        for p in range(1, clause_size_sc[r]+1):
            yrp = get_key_by_value(temp_row_r, (r,p))
            logger.debug(f"Y[{r},{p}]: {yrp}")
            card_literals_sc.append(yrp)

        logger.debug(f"[SC] Card Literals: {card_literals_sc}")

        try:
            cnf_sc = CardEnc.atleast(lits=card_literals_sc, bound=b_val, top_id=lit_cnt, encoding=EncType.seqcounter) # default encoding
            cnf_clauses = cnf_sc.clauses
            logger.debug(f"[SC] Y[{r},{p}] Clauses: {cnf_clauses}")

            # Updating literal count
            if cnf_sc.nv != 0:
                lit_cnt = cnf_sc.nv

            logger.debug(f"UPDATED LITERAL COUNT: {lit_cnt}")

        except NoSuchEncodingError as e:
            # Handle the case where the encoding does not exist
            logger.error(f"Caught NoSuchEncodingError: {e}")

        except ValueError as e:
            # Handle ValueError if the bound or input is invalid
            logger.error(f"Caught ValueError: {e}")

        except Exception as e:
            logger.error(f"An Unexpected Error Occurred: {e}")

        clause_builder_card.append(cnf_clauses)

    logger.debug(f"[SC] Atleast Clauses: {clause_builder_card}")

    # Remove one layer of lists
    logger.debug(f"Removing One Layer Of Lists...")
    for cl in clause_builder_card:
        # logger.debug(cl)
        for c in cl:
            # logger.debug(c)
            if clause_verbosity > 1:
                atleast_card_clauses.append(c)

            final_clauses.append(c)
            total_num_literals += len(c)

    if clause_verbosity > 1:
        logger.debug(f"[SC] Final Atleast Clauses: {atleast_card_clauses}")
        clauses["atleast_clauses"] = atleast_card_clauses

    instance_data = {
        "num_clauses": len(final_clauses),
        "num_variables": lit_cnt,
        "num_literals": total_num_literals
    }

    logger.debug(f"Instance Data: {instance_data}")
    sat_obj["instance_data"] = instance_data

    cnf.extend(final_clauses)

    logger.debug(f"CNF Size: {len(cnf.clauses)} clauses, {instance_data['num_literals']} literals.")

    if not cnf.clauses:
        logger.error("CNF is empty. Check encoding logic.")
        sys.exit(1)

    if save_cnf_to_file:
        cnf.to_file(kwargs['output_cnf_fp'])
        logger.info(f"CNF Saved To File: {kwargs['output_cnf_fp']}")

    sat_obj["cnf_object"] = cnf

    sat_obj["literals"] = literals

    if clause_verbosity > 1:
        sat_obj["clauses"] = clauses

    sat_obj["final_clauses"] = final_clauses
    sat_obj["final_lit_cnt"] = lit_cnt

    return sat_obj


if __name__ == "__main__":
    logger.info("********************EQUIVALENCE-RELATION-ENCODER[LOCAL_TESTING]*********************")

    rgp_instances = json_to_rgp(default_jfp)
    # rgp_instances = json_to_rgp(small_jfp)
    # rgp_instances = json_to_rgp(neg_jfp)

    sat_instances = []

    for rgp_inst in rgp_instances:
        sat_obj = rgp_to_sat_er(rgp_inst)
        logger.info(f"SAT Object [ER-ENCODER]: {sat_obj}")

        sat_instances.append(sat_obj)

    logger.info(f"SAT Objects: {sat_instances}")

    clauses = sat_obj["final_clauses"]
    instance_data = sat_obj["instance_data"]

    logger.info(f"Clauses: {clauses}")
    logger.info(f"Instance Data: {instance_data}")