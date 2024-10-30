import os
import sys
import math
import json
import random
from itertools import combinations

from .utils import get_file_path
from .logger import create_logger
from .helper import generate_unique_pairs, write_to_file, json_to_dict, rgp_dict_to_rgp

logger = create_logger(l_name="zt_generator")


def generate_partition(n: int, k: int):
    '''
    Generate a partition of n into k non-empty groups by randomly
    selecting one set of k-1 dividers within n elements

    Args:
        n: Number of elements
        k: Number of groups

    Returns:
        list: Partition of n into k groups
    '''
    # Randomly get one set of k-1 dividers without generating all possible sets
    ran_dividers = random.sample(range(1,n), k-1)
    ran_dividers.sort()
    logger.debug(f"Randomly Selected Dividers: {ran_dividers}")

    partition = []
    start = 0

    for div in ran_dividers:
        logger.debug(f"Div: {div}")

        partition.append(div - start)
        logger.debug(f"Partition: {partition}")

        # Move to just after the divider
        start = div

    # Last group is for the remaining elements
    partition.append(n - start)
    logger.debug(f"Final Partition: {partition}")

    return partition


def partition_array_into_k_groups(elements: list, k: int) -> list:
    '''
    Partition an array into k groups where each partition is equally likely

    Args:
        elements: List of elements to partition
        k: Number of groups

    Returns:
        list: A list of k lists, where each inner list represents a group.
    '''
    n = len(elements)
    logger.debug(f"Number of Elements: {n}")

    try:
        if k > n:
            raise ValueError

    except ValueError:
        logger.error(f"Number of Groups (k={k}) cannot be greater than the Number of Elements (n={n})")
        sys.exit(1)

    # Generate a partition of n into k non-empty groups
    partition = generate_partition(n, k)

    # Distribute elements based on the selected partition
    groups = []
    start = 0

    for partition_size in partition:
        groups.append(elements[start:start + partition_size])
        start += partition_size

    for idx, group in enumerate(groups):
        logger.debug(f"Group {idx + 1}: {group}")

    return groups


def find_num_groups_involved(elements: list, partitions: list) -> int:
    '''
    Find out how many groups were involved for the given set of elements

    Args:
        elements: The list of elements
        partitions: The list of partitions of elements

    Return:
        int: The number of different groups that were holding the elements
    '''
    # Convert the list of elements to a set for O(1) membership testing
    element_set = set(elements)

    num_groups = 0

    for partition in partitions:
        # Check if any element in the partition is in the element_set
        if any(item in element_set for item in partition):
            num_groups += 1

    return num_groups


def generate_constraints(instance_config: dict, elements: list, partitions: list, flag: int) -> tuple:
    '''
    Randomly generate satisfiable [positive] or unsatisfiable [negative]
    usability and security constraints based on the flag provided

    Args:
        instance_config: Configuration details of the instance [Metadata]
        elements: List of all elements
        partitions: The list of partitions of elements
        flag: 0/1 for negative/positive constraints

    Returns:
        tuple: A tuple containing the UC and SC

    Algorithm:
        FOR SAT CONSTRAINTS:
        1. Select 's' elements from n resources
        2. Put them into a constraint array
        3. Find out how many groups were involved (k value = # of groups)
        4. [[s], 'le', k+1] -> UC
        5. [[s], 'ge', k-1] -> SC

        6. If flag == 0: FOR UNSAT CONSTRAINTS
            6.1. Select k elements (k < num_groups) each from a different group
            6.2. Generate SC of size [2],[3],...,[k-1] with every pair of the elements of the form {(1,2), ge, 2}
            6.3. Generate unsat UC of the form {(1,2,...,k), le, k-1}
    '''
    num_resources = instance_config['num_resources']
    num_groups = instance_config['num_groups']
    num_constraints = instance_config['num_constraints']
    constraint_size_type = instance_config['constraint_size_type']

    # Fixed constraint size for positive constraints
    if constraint_size_type == "fixed":
        constraint_size = instance_config['constraint_size']

    uc = []
    sc = []

    if flag == 0:
        # Generate unsatisfiable constraints
        k = random.randint(2, num_groups)
        selected_groups = random.sample(partitions, k)

        logger.debug(f"Groups From Which Elements Will Be Selected: {selected_groups}")

        neg_elements = []

        for i in range(k):
            choice = random.choice(selected_groups[i])
            neg_elements.append(choice)

        neg_elements.sort()
        logger.debug(f"Elements Selected From Groups: {neg_elements} -> [{len(neg_elements)} Elements Selected]")

        neg_uc = [neg_elements, 'le', k-1]
        logger.debug(f"Negative Usability Constraint:  {neg_uc}")
        uc.append(neg_uc)

        unique_pairs = generate_unique_pairs(neg_elements)
        num_neg_sc = len(unique_pairs)

        # Update Total Number of Constraints - Account for 1 UC and Num_Neg_SC SC
        if num_constraints < (num_neg_sc + 1):
            logger.error("Requested To Generate Fewer Constraints!")

        else:
            num_constraints = num_constraints - num_neg_sc - 1

        '''
        Flag to decide if we want more than 2 elements in an SC
        Randomly select a number -> If selection == 1, we add more than 2 elements
        Can add more elements to change probability of selecting 1
        '''
        more_than_two = [0, 1]

        for i in range(0, num_neg_sc):
            constraint = []

            # Add elements to constraint
            constraint_elements = list(unique_pairs[i])
            constraint_elements.sort()

            mtt_choice = random.choice(more_than_two)
            if mtt_choice == 1:
                # Add functionality to add more than 2 elements to a constraint
                pass

            constraint.append(constraint_elements)
            constraint.append('ge')
            constraint.append(len(constraint_elements))

            sc.append(constraint)

            logger.debug(f"Negative Security Constraint {i}:  {constraint}")

    for i in range (1, num_constraints+1):
        # [[s],op,b]
        constraint = []

        # Random constraint sizes
        if constraint_size_type == "random":
            constraint_size = random.randint(1,num_resources-1)
            logger.debug(f"Random Constraint Size: {constraint_size}")

        # Add elements to constraint
        constraint_elements = list(random.sample(elements, constraint_size))
        constraint_elements.sort()

        constraint.append(constraint_elements)

        # Add op ('le','ge') based on random choice
        options = ['le','ge']
        selection = random.choice(options)
        constraint.append(selection)
        logger.debug(f"Constraint Selection [UC/SC]: {selection}")

        # Add b value to constraint
        num_groups_involved = find_num_groups_involved(constraint_elements, partitions)
        logger.debug(f"Number of Groups Involved: {num_groups_involved}")

        if selection == 'le':
            b_val = num_groups_involved+1
            constraint.append(b_val)

            uc.append(constraint)

        else:
            b_val = num_groups_involved-1
            constraint.append(b_val)

            sc.append(constraint)

        logger.debug(f"Constraint {i}:  {constraint}")

    logger.debug(f"Final UC Generated: {uc}")
    logger.debug(f"Final SC Generated: {sc}")

    return uc,sc


def generate_rgp_instances(
    flag: int,
    n: int = 10,
    cst_size_type: str = "fixed",
    n_cst: int = 10,
    cst_size: int = 3,
    num_instance: int = 5,
    top_id: int = -1,
    exp_config: bool = False
) -> tuple:
    '''
    Generate negative [UNSAT] or positive [SAT] RGP instances based
    on a flag = 0 [negative], 1 [positive] or 2 [random selection],
    convert them to a JSON object and write them to a .json file.

    Args:
        flag: 0/1/2 for negative, positive or random instances [random selection of pos/neg instances]
        n: Number of resources in each instance
        cst_size_type: Constraint Size Type [Random/Fixed]
        n_cst: Number of constraints
        cst_size: Size of each constraint
        num_instance: Number of instances to be generated
        top_id: The IDs we assign to instances [instances will be generated with ID=top_id+1 and so on...]
        exp_config: Flag to decide how we need to store the instances [If we are using experiment config or not]

    Returns:
        tuple: Containing "num_instances" instances in a dictionary and the top_id
    '''
    if not isinstance(flag, int) or flag not in range(0,3):
        logger.error(f"Flag {flag} must be an integer that is either 0, 1 or 2")
        sys.exit(1)

    rgp_instances = {}

    # Instance Meta Data
    instance_config = {
        "num_instances": num_instance,
        "num_groups": 1,
        "num_resources": n,
        "num_constraints": n_cst,
        "constraint_size_type": cst_size_type,
        "constraint_size": cst_size
    }

    begin_id = top_id+1
    end_id = begin_id + num_instance

    for i in range(begin_id, end_id):
        inst = {}
        inst["i"] = i

        inst["n"] = n
        elements = list(range(1, n+1))

        sq_n = int(math.sqrt(n))
        t = random.randint(2,sq_n)
        inst["t"] = t
        logger.debug(f"Number of Groups: {t}")

        instance_config["num_groups"] = t

        partitioned_groups = partition_array_into_k_groups(elements, t)
        inst["partitions"] = partitioned_groups

        if flag == 0: # Negative
            inst["flag"] = flag
            uc,sc = generate_constraints(instance_config, elements, partitioned_groups, flag)

        elif flag == 1: # Positive
            inst["flag"] = flag
            uc,sc = generate_constraints(instance_config, elements, partitioned_groups, flag)

        elif flag == 2: # Randomly Select Positive or Negative Flag
            options = [0,1]
            ran_flag = random.choice(options)

            inst["flag"] = ran_flag
            logger.debug(f"Random Chosen Flag: {ran_flag}")

            uc,sc = generate_constraints(instance_config, elements, partitioned_groups, ran_flag)

        inst["uc"] = uc
        inst["sc"] = sc

        logger.debug(f"Instance {i}: {inst}")

        key = str(i)
        rgp_instances[key] = inst

    logger.debug(f"Generated RGP Instances: {rgp_instances}")

    if not exp_config:
        target_dir_name = "data"
        sub_dir_name = "testfiles"

        json_file_name = "rgp_gen_test_" + str(flag) + ".json"
        json_file_path = get_file_path(target_dir_name, sub_dir_name, json_file_name)
        mode = "w"

    else:
        target_dir_name = "assets"
        sub_dir_name = "files"

        json_file_name = "rgp_gen_exp.json"
        json_file_path = get_file_path(target_dir_name, sub_dir_name, json_file_name)
        mode = "a"

    write_to_file(rgp_instances, json_file_path, mode)

    top_id = end_id-1

    return (rgp_instances, top_id)


def generate_rgp_instances_with_config(flag: int, experiment_config_path: str) -> tuple:
    '''
    Generate negative [UNSAT] or positive [SAT] RGP instances based
    on a flag = 0 [negative], 1 [positive] or 2 [random selection],
    and the experiment configurations specificied. Convert them to
    a JSON object and write them to a .json file.

    Args:
        flag: 0/1/2 for negative, positive or random instances [random selection of pos/neg instances]
        experiment_config_path: Path to experiment configuration file

    Returns:
        tuple: Contains a boolean flag (T/F) to indicate if the generation
               of instances was successful and the updated top_id
    '''
    experiment_config = json_to_dict(experiment_config_path)

    num_instances = experiment_config["num_instances"]

    num_resources = experiment_config["num_resources"]
    num_constraints = experiment_config["num_constraints"]
    constraint_size = experiment_config["constraint_size"]

    total_num_instances = num_instances * len(num_resources) * len(num_constraints) * len(constraint_size)
    logger.debug(f"Total Number Of Instances: {total_num_instances}")

    top_id = -1

    for nr in num_resources:
        for nc in num_constraints:
            for cs in constraint_size:
                logger.debug(f"Top ID: {top_id}")
                rgp_obj, new_top_id = generate_rgp_instances(flag=flag, n=nr, cst_size_type="fixed", n_cst=nc, cst_size=cs, num_instance=num_instances, top_id=top_id, exp_config=True)

                rgp_instances = rgp_dict_to_rgp(rgp_obj)
                top_id = new_top_id
                logger.debug(f"Updated Top ID: {top_id}")

    if total_num_instances != (top_id+1):
        logger.error("Error Generating Instances Correctly!")
        return (False, top_id)

    else:
        return (True, top_id)


if __name__ == "__main__":
    logger.info("********************GENERATOR[LOCAL_TESTING]*********************")

    flag, top_id = generate_rgp_instances(flag=0, n=10)
    # flag, top_id = generate_rgp_instances(flag=1, n=10)
    # flag, top_id = generate_rgp_instances(flag=2, n=10)