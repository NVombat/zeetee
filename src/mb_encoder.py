from pysat.card import CardEnc, EncType, NoSuchEncodingError

from . import *
from .logger import create_logger
from .helper import get_key_by_value, json_to_rgp, extract_clauses_and_instance_data

logger = create_logger(l_name="zt_mb_encoder")


def rgp_to_sat_mb(rgp_obj: dict) -> dict:
    '''
    Takes an RGP dictionary object and returns a SAT
    dictionary object w.r.t the first encoding which
    is a mapping of resources to groups

    Args:
        rgp_obj: RGP dictionary object

    Returns:
        dict: SAT dictionary object
    '''
    sat_obj = {}

    literal_mappings = {} # only for resources (xiu)
    literal_mappings_uc = {} # usability constraints (yru)
    literal_mappings_sc = {} # security constraints (yru)

    literals = []
    clauses = {}

    n = rgp_obj["n"]
    logger.debug(f"Number of Resources: {n}")

    t = rgp_obj["t"]
    logger.debug(f"Maximum Number of Groups (Upperbound): {t}")

    lit_id = 1  # To get unique literals
    for i in range (1,n+1): # for every resource
        literal_map = {}

        for u in range (1, t+1): # for every group
            logger.debug(f"Resource[i], Group[u]: {i},{u}")
            val = (i,u) # i,u (placeholders for ith resource and uth group)

            literal_map[lit_id] = val
            lit_id = lit_id+1

        literal_mappings[i] = literal_map

    sat_obj["literal_mappings"] = literal_mappings

    uc = rgp_obj["uc"]
    logger.debug(f"Usability Constraints: {uc}")

    # Ensure correctly formatted UC, create boolean variables y[r,u] and get size and "b" values of each clause
    clause_cnt_uc = 1
    clause_size_uc = {}
    b_val_uc = {}

    lit_map_uc_index = 1

    for c1 in uc:
        logger.debug(f"Constraint: {c1}")
        # create a literal for every constraint and for every group t
        lr = len(c1[0])

        # store clause size and "b" values
        clause_size_uc[clause_cnt_uc] = lr
        b_val_uc[clause_cnt_uc] = c1[-1]
        clause_cnt_uc = clause_cnt_uc+1

        literal_map = {}
        for u in range (1, t+1): # for every group
            val = (lit_map_uc_index,u) # val = [r,u]
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

    # Ensure correctly formatted SC, create boolean variables y[r,u] and get size and "b" values of each clause
    clause_cnt_sc = 1
    clause_size_sc = {}
    b_val_sc = {}

    lit_map_sc_index = 1

    for c2 in sc:
        logger.debug(f"Constraint: {c2}")
        # create a literal for every constraint and for every group t
        lr = len(c2[0])

        # store clause size and "b" values
        clause_size_sc[clause_cnt_sc] = lr
        b_val_sc[clause_cnt_sc] = c2[-1]
        clause_cnt_sc = clause_cnt_sc+1

        literal_map = {}
        for u in range (1, t+1): # for every group
            val = (lit_map_sc_index,u) # val = [r,u]
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

    outer_lits = literal_mappings.keys() # i values
    logger.debug(f"Outer Literals: {outer_lits}")

    # Keeps track of updated literal count
    lit_cnt = lit_id-1

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

    logger.debug("********************CLAUSE(1)(2)********************")
    clause_1 = [] # Every resource is mapped to atleast one group
    clause_2 = [] # Every resource is mapped to atmost one group

    for i in range (1, n+1):
        temp_row_i = literal_mappings[i]
        logger.debug(f"TEMP ROW I: {temp_row_i}")

        tc = []

        for u in range (1, t+1):
            xiu = get_key_by_value(temp_row_i, (i,u))
            logger.debug(f"X[{i},{u}]: {xiu}")
            tc.append(xiu)

        logger.debug(f"TC: {tc}")
        clause_1.append(tc)

        # A resource must in at most 1 group
        # totalizer = ITotalizer(lits=tc, ubound=1, top_id=lit_cnt)
        # cnf_atmost = CardEnc.atmost(lits=tc, bound=1, top_id=lit_cnt, encoding=EncType.pairwise)
        cnf_atmost = CardEnc.atmost(lits=tc, bound=1, top_id=lit_cnt, encoding=EncType.seqcounter) # default encoding

        cnf_atmost_clauses = cnf_atmost.clauses
        # logger.debug(f"Atmost Clauses: {cnf_atmost_clauses}")

        for clause in cnf_atmost_clauses:
            # Remove one layer of lists
            clause_2.append(clause)

        logger.debug(f"Clause 2 Temp: {clause_2}")

        # Updating literal count
        lit_cnt = cnf_atmost.nv
        logger.debug(f"Updated Literal Count: {lit_cnt}")

    logger.debug(f"Clause 1: {clause_1}")
    clauses["clause_1"] = clause_1

    logger.debug(f"Clause 2: {clause_2}")
    clauses["clause_2"] = clause_2

    logger.debug("********************CLAUSE(3)********************")
    clause_3 = [] # ¬xirp,u ∨ yr,u

    for r in range (1, num_uc+1):
        temp_row_r = literal_mappings_uc[r] # y[r,u] values
        logger.debug(f"[UC] TEMP ROW R: {temp_row_r}")

        p_lim = clause_size_uc[r]
        logger.debug(f"[UC] Size of Clause {r}: {p_lim}")

        uc_r = uc[r-1][0] # Resources in UC
        logger.debug(f"[UC] Constraint: {uc_r}")

        for p in range (1, p_lim+1):
            p_val = uc_r[p-1]
            logger.debug(f"[UC] P Value: {p_val}")

            temp_row_i = literal_mappings[p_val]
            logger.debug(f"[UC] TEMP ROW I: {temp_row_i}")

            for u in range (1, t+1):
                tc = []

                xpu = get_key_by_value(temp_row_i, (p_val,u))
                logger.debug(f"[UC] X[{p_val},{u}]: {xpu}")

                yru = get_key_by_value(temp_row_r, (r,u))
                logger.debug(f"[UC] Y[{r},{u}]: {yru}")

                tc.append(-xpu)
                tc.append(yru)

                logger.debug(f"[UC] TC: {tc}")

                clause_3.append(tc)

    for r in range (1, num_sc+1):
        temp_row_r = literal_mappings_sc[r]
        logger.debug(f"[SC] TEMP ROW R: {temp_row_r}")

        p_lim = clause_size_sc[r]
        logger.debug(f"[SC] Size of Clause {r}: {p_lim}")

        sc_r = sc[r-1][0] # Resources in SC
        logger.debug(f"[SC] Constraint: {sc_r}")

        for p in range (1, p_lim+1):
            p_val = sc_r[p-1]
            logger.debug(f"[SC] P Value: {p_val}")

            temp_row_i = literal_mappings[p_val]
            logger.debug(f"[SC] TEMP ROW I: {temp_row_i}")

            for u in range (1, t+1):
                tc = []

                xpu = get_key_by_value(temp_row_i, (p_val,u))
                logger.debug(f"[SC] X[{p_val},{u}]: {xpu}")

                yru = get_key_by_value(temp_row_r, (r,u))
                logger.debug(f"[SC] Y[{r},{u}]: {yru}")

                tc.append(-xpu)
                tc.append(yru)

                logger.debug(f"[SC] TC: {tc}")

                clause_3.append(tc)

    logger.debug(f"Clause 3: {clause_3}")
    clauses["clause_3"] = clause_3

    logger.debug("********************CLAUSE(4)********************")
    clause_4 = []

    for r in range (1, num_uc+1):
        temp_row_r = literal_mappings_uc[r] # y[r,u] values
        logger.debug(f"[UC] TEMP ROW R: {temp_row_r}")

        p_lim = clause_size_uc[r]
        logger.debug(f"[UC] Size of Clause {r}: {p_lim}")

        uc_r = uc[r-1][0] # Resources in constraint
        logger.debug(f"[UC] Constraint: {uc_r}")

        for u in range (1, t+1):
            yru = get_key_by_value(temp_row_r, (r,u))
            logger.debug(f"[UC] Y[{r},{u}]: {yru}")

            tc = []
            tc.append(-yru)

            for p in range (1, p_lim+1):
                p_val = uc_r[p-1]
                logger.debug(f"[UC] P Value: {p_val}")

                temp_row_i = literal_mappings[p_val]
                logger.debug(f"[UC] TEMP ROW I: {temp_row_i}")

                xpu = get_key_by_value(temp_row_i, (p_val,u))
                logger.debug(f"[UC] X[{p_val},{u}]: {xpu}")

                tc.append(xpu)

            logger.debug(f"[UC] TC: {tc}")
            clause_4.append(tc)

    for r in range (1, num_sc+1):
        temp_row_r = literal_mappings_sc[r]
        logger.debug(f"[SC] TEMP ROW R: {temp_row_r}")

        p_lim = clause_size_sc[r]
        logger.debug(f"[SC] Size of Clause {r}: {p_lim}")

        sc_r = sc[r-1][0] # Resources in constraint
        logger.debug(f"[SC] Constraint: {sc_r}")

        for u in range (1, t+1):
            yru = get_key_by_value(temp_row_r, (r,u))
            logger.debug(f"[SC] Y[{r},{u}]: {yru}")

            tc = []
            tc.append(-yru)

            for p in range (1, p_lim+1):
                p_val = sc_r[p-1]
                logger.debug(f"[SC] P Value: {p_val}")

                temp_row_i = literal_mappings[p_val]
                logger.debug(f"[SC] TEMP ROW I: {temp_row_i}")

                xpu = get_key_by_value(temp_row_i, (p_val,u))
                logger.debug(f"[SC] X[{p_val},{u}]: {xpu}")

                tc.append(xpu)

            logger.debug(f"[SC] TC: {tc}")
            clause_4.append(tc)

    logger.debug(f"Clause 4: {clause_4}")
    clauses["clause_4"] = clause_4

    logger.debug("********************CLAUSE(5)********************")
    '''
    CardEnc.atmost():
    When the upper bound is equal to the number of literals,
    the constraint is trivially true because all literals
    can be true without violating the constraint.
    '''
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

        for u in range(1, t+1):
            yru = get_key_by_value(temp_row_r, (r,u))
            logger.debug(f"Y[{r},{u}]: {yru}")
            card_literals_uc.append(yru)

        logger.debug(f"[UC] Card Literals: {card_literals_uc}")

        # Totalizer Atmost Constraint
        # totalizer = ITotalizer(lits=card_literals_uc, ubound=b_val, top_id=lit_cnt)
        # logger.info(f"[UC] TOTALIZER: {totalizer}")

        # totalizer_clauses = totalizer.cnf.clauses
        # logger.info(f"[UC] Y[r,u] Totalizer Clauses: {totalizer_clauses}")

        # Updating Literal Count
        # lit_cnt = totalizer.top_id

        # Regular Atmost Constraint - Only generate non-trivial clauses
        if b_val < len(card_literals_uc):
            cnf_uc = CardEnc.atmost(lits=card_literals_uc, bound=b_val, top_id=lit_cnt, encoding=EncType.seqcounter) # default encoding
            logger.info(f"[UC] CNF: {cnf_uc}")

            cnf_clauses = cnf_uc.clauses
            logger.debug(f"[UC] Y[r,u] Clauses: {cnf_clauses}")

            # Updating literal count
            if cnf_uc.nv != 0:
                lit_cnt = cnf_uc.nv

            logger.debug(f"UPDATED LITERAL COUNT: {lit_cnt}")

        else:
            cnf_clauses = []

        # clause_builder_card.append(totalizer_clauses)
        clause_builder_card.append(cnf_clauses)

    logger.debug(f"[UC] Atmost Clauses: {clause_builder_card}")

    logger.debug(f"Removing One Layer Of Lists...")
    # Remove one layer of lists
    for cl in clause_builder_card:
        # logger.debug(cl)
        for c in cl:
            # logger.debug(c)
            atmost_card_clauses.append(c)

    logger.debug(f"[UC] Final Atmost Clauses: {atmost_card_clauses}")
    clauses["atmost_clauses"] = atmost_card_clauses

    logger.debug("********************CLAUSE(6)********************")
    '''
    CardEnc.atleast():
    When the lower bound is equal to the number of literals,
    the constraint is also trivially true because all literals
    need to be true, which naturally satisfies the constraint.
    '''
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

        for u in range(1, t+1):
            yru = get_key_by_value(temp_row_r, (r,u))
            logger.debug(f"Y[{r},{u}]: {yru}")
            card_literals_sc.append(yru)

        logger.debug(f"[SC] Card Literals: {card_literals_sc}")

        try:
            cnf_sc = CardEnc.atleast(lits=card_literals_sc, bound=b_val, top_id=lit_cnt, encoding=EncType.seqcounter) # default encoding
            cnf_clauses = cnf_sc.clauses
            logger.debug(f"[SC] Y[r,u] Clauses: {cnf_clauses}")

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
            atleast_card_clauses.append(c)

    logger.debug(f"[SC] Final Atleast Clauses: {atleast_card_clauses}")
    clauses["atleast_clauses"] = atleast_card_clauses

    logger.debug("**********Additional Boolean Variables (Z Literals)**********")
    z_lits = []
    z_literal_mappings = {}

    lit_id = lit_cnt+1

    for u in range (1, t+1):
        z_literal_mappings[lit_id] = u # maps z_lits to groups
        z_lits.append(lit_id)
        lit_id = lit_id+1

    logger.debug(f"Z Literals: {z_lits}")
    sat_obj["z_lits"] = z_lits

    logger.debug(f"Z Literal Mappings: {z_literal_mappings}")
    sat_obj["z_literal_mappings"] = z_literal_mappings

    lit_cnt = lit_id-1
    logger.debug(f"UPDATED LITERAL COUNT: {lit_cnt}")

    logger.debug("********************CLAUSE(7)********************")
    clause_7 = []

    for i in range (1, n+1):
        temp_row_i = literal_mappings[i]
        logger.debug(f"TEMP ROW I: {temp_row_i}")

        for u in range (1, t+1):
            tc = []

            zu = get_key_by_value(z_literal_mappings, u)
            logger.debug(f"Z[{u}] = {zu}")

            xiu = get_key_by_value(temp_row_i, (i,u))
            logger.debug(f"X[{i},{u}]: {xiu}")

            tc.append(-xiu)
            tc.append(zu)

            logger.debug(f"TC: {tc}")

            clause_7.append(tc)

    logger.debug(f"Clause 7: {clause_7}")
    clauses["clause_7"] = clause_7

    logger.debug("********************CLAUSE(8)********************")
    clause_8 = []

    for u in range (1, t+1):
        zu = get_key_by_value(z_literal_mappings, u)
        logger.debug(f"Z[{u}] = {zu}")

        tc = []
        tc.append(-zu)

        for i in range(1, n+1):
            temp_row_i = literal_mappings[i]
            logger.debug(f"TEMP ROW I: {temp_row_i}")

            xiu = get_key_by_value(temp_row_i, (i,u))
            logger.debug(f"X[{i},{u}]: {xiu}")

            tc.append(xiu)

        logger.debug(f"TC: {tc}")
        clause_8.append(tc)

    logger.debug(f"Clause 8: {clause_8}")
    clauses["clause_8"] = clause_8

    logger.debug("********************CLAUSE(9)********************")
    clause_9 = []

    for u in range(1, t):
        tc = []

        zu = get_key_by_value(z_literal_mappings, u)
        logger.debug(f"Z[{u}] = {zu}")

        zu_1 = get_key_by_value(z_literal_mappings, u+1)
        logger.debug(f"Z[{u+1}] = {zu_1}")

        tc.append(zu)
        tc.append(-zu_1)

        logger.debug(f"TC: {tc}")
        clause_9.append(tc)

    logger.debug(f"Clause 9: {clause_9}")
    clauses["clause_9"] = clause_9

    logger.debug("**********Additional Boolean Variables (W Literals)**********")
    w_lits = []
    w_literal_mappings = {}

    lit_id = lit_cnt+1

    for i in range (1, n+1):
        literal_map = {}

        for u in range (1, t+1):
            val = (i,u)

            literal_map[lit_id] = val
            w_lits.append(lit_id)

            lit_id = lit_id+1

        w_literal_mappings[i] = literal_map

    logger.debug(f"W Literals: {w_lits}")
    sat_obj["w_lits"] = w_lits

    logger.debug(f"W Literal Mappings: {w_literal_mappings}")
    sat_obj["w_literal_mappings"] = w_literal_mappings

    lit_cnt = lit_id-1
    logger.debug(f"UPDATED LITERAL COUNT: {lit_cnt}")

    logger.debug("********************CLAUSE(10)********************")
    clause_10 = []

    for i in range (1, n+1):
        temp_row_x = literal_mappings[i]
        logger.debug(f"TEMP ROW X: {temp_row_x}")

        for j in range (i+1, n+1):
            temp_row_w = w_literal_mappings[j]
            logger.debug(f"TEMP ROW W: {temp_row_w}")

            for u in range (1, t+1):
                tc = []

                wju = get_key_by_value(temp_row_w, (j,u))
                logger.debug(f"W[{j},{u}]: {wju}")

                xiu = get_key_by_value(temp_row_x, (i,u))
                logger.debug(f"X[{i},{u}]: {xiu}")

                tc.append(-wju)
                tc.append(-xiu)
                # tc.append([-wju, -xiu])

                logger.debug(f"TC: {tc}")
                clause_10.append(tc)

    logger.debug(f"Clause 10: {clause_10}")
    clauses["clause_10"] = clause_10

    logger.debug("********************CLAUSE(11)********************")
    clause_11 = []

    for i in range (1, n+1):
        temp_row_x = literal_mappings[i]
        logger.debug(f"TEMP ROW X: {temp_row_x}")

        temp_row_w = w_literal_mappings[i]
        logger.debug(f"TEMP ROW W: {temp_row_w}")

        for u in range (1, t+1):
            tc = []

            wiu = get_key_by_value(temp_row_w, (i,u))
            logger.debug(f"W[{i},{u}]: {wiu}")

            xiu = get_key_by_value(temp_row_x, (i,u))
            logger.debug(f"X[{i},{u}]: {xiu}")

            tc.append(-wiu)
            tc.append(xiu)

            logger.debug(f"TC: {tc}")
            clause_11.append(tc)

    logger.debug(f"Clause 11: {clause_11}")
    clauses["clause_11"] = clause_11

    logger.debug("********************CLAUSE(12)********************")
    clause_12 = []

    for u in range (1, t+1):
        zu = get_key_by_value(z_literal_mappings, u)
        logger.debug(f"Z[{u}] = {zu}")

        tc = []
        tc.append(-zu)

        for i in range(1, n+1):
            temp_row_w = w_literal_mappings[i]
            logger.debug(f"TEMP ROW W: {temp_row_w}")

            wiu = get_key_by_value(temp_row_w, (i,u))
            logger.debug(f"W[{i},{u}]: {wiu}")

            tc.append(wiu)

        logger.debug(f"TC: {tc}")
        clause_12.append(tc)

    logger.debug(f"Clause 12: {clause_12}")
    clauses["clause_12"] = clause_12

    logger.debug("********************CLAUSE(13)********************")
    clause_13 = []

    for i in range (1, n+1):
        temp_row_wi = w_literal_mappings[i]
        logger.debug(f"TEMP ROW WI: {temp_row_wi}")

        for j in range (1, i+1): # j ≤ i
            temp_row_wj = w_literal_mappings[j]
            logger.debug(f"TEMP ROW WJ: {temp_row_wj}")

            for u in range (1, t+1):
                wiu = get_key_by_value(temp_row_wi, (i,u))
                logger.debug(f"W[i={i},u={u}]: {wiu}")

                for v in range (u+1, t+1): # u < v
                    tc = []

                    wjv = get_key_by_value(temp_row_wj, (j,v))
                    logger.debug(f"W[j={j},v={v}]: {wjv}")

                    tc.append(-wiu)
                    tc.append(-wjv)

                    clause_13.append(tc)

    logger.debug(f"Clause 13: {clause_13}")
    clauses["clause_13"] = clause_13

    sat_obj["literals"] = literals
    sat_obj["clauses"] = clauses
    sat_obj["final_lit_cnt"] = lit_cnt

    return sat_obj


if __name__ == "__main__":
    logger.info("********************MAPPING-BASED-ENCODER[LOCAL_TESTING]*********************")

    rgp_instances = json_to_rgp(default_jfp)
    # rgp_instances = json_to_rgp(small_jfp)
    # rgp_instances = json_to_rgp(neg_jfp)

    sat_instances = []

    for rgp_inst in rgp_instances:
        sat_obj = rgp_to_sat_mb(rgp_inst)
        logger.info(f"SAT Object [MB-ENCODER]: {sat_obj}")

        sat_instances.append(sat_obj)

    logger.info(f"SAT Objects: {sat_instances}")

    clauses,instance_data = extract_clauses_and_instance_data(sat_obj)
    logger.info(f"Clauses: {clauses}")
    logger.info(f"Instance Data: {instance_data}")