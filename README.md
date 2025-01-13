# ZeeTee - A Tool For Balancing Usability & Security in a Zero Trust Environment

## Project Workflow

![image](./docs/workflow.jpeg)

## The Resource Grouping Problem (RGP)

![image](./docs/resource_grouping_problem.jpeg)

## Mapping-Based Encoding

![image](./docs/mapping_based_encoding.jpeg)

### Eliminating Combination Symmetries

![image](./docs/eliminating_comb_symmetries.jpeg)

### Eliminating Permutation Symmetries

![image](./docs/eliminating_perm_symmetries.jpeg)

## Equivalence-Based Encoding

![image](./docs/equivalence_based_encoding.jpeg)

## 📌 Prerequisites

### 💻 System requirement :

1. Any system with basic configuration.
2. Operating System : Any (Windows / Linux / Mac).
3. Access to an High Performance Computing (HPC) Setup (Preferable for running experiments)

### 💿 Software requirement :

1. Python installed (If not download it [here](https://www.python.org/downloads/)).
2. Poetry Installed (If not download it [here](https://python-poetry.org/docs/)).
2. Any text editor of your choice.

## 🔧 Installation & Setup

### 🏗️ Clone The Repository :

```
$ git clone https://github.com/NVombat/zeetee.git
```

### ➕ Install Dependencies :

```
$ poetry install

OR

$ pip install -r requirements.txt
```

If you do not wish to use poetry, you can convert the poetry.lock file to a requirements.txt file

### 📝 Setup .env File :

Refer to .env.example and create project secrets
Setup Email API keys and Target Email ID

### 🔐 Setup Configuration Files :

Given below is a sample configuration file. While naming configuration files they should be named as such:

```
$ FILENAME -> experiment_config_N10.json
$ FILENAME -> experiment_config_N700.json
```

The number following N is the value of N (num_resources) in the experiment configuration file.
All the configuration files used can be found in the /data/configfiles directory

You can refer to our Configuration Files when setting up your own.

![image](./docs/exp_config.jpeg)

Configuration files are .json files with the following keys:

```
$ 'num_instances' [int] - Number of RGP instances to be generated
$ 'num_resources' [list] - Number of RGP resources to be generated
$ 'constraint_percentages' [list] - Number of constraints expressed as a ratio to num_resources
$ 'constraint_size' [list] - Size of the set S in a constraint (S, op b)
$ 'timeout' [int] - Timeout limit for the solver (in milliseconds)
```

### 🔐 Setup SLURM Files [For HPC] :

SLURM Documentation can be found [here](https://slurm.schedmd.com/documentation.html).

If an HPC setup is being used, then a SLURM file will have to be setup.
All the SLURM files used can be found in the /data/slurm directory.

We have used 4 different types of slurm files based on what they do:

```
$ TYPE 1 -> pre_process_N10.slurm
$ TYPE 2 -> run_existing_N10.slurm
$ TYPE 3 -> run_experiment_N10.slurm
$ TYPE 4 -> solve_preprocessed_N10.slurm
```

You can refer to our SLURM Files when setting up your own.

### 🚀 Run Project :

The main entry point is the run.sh file. We first convert this file to an executable.

```
$ chmod u+x run.sh
```

arc_runner.py and runner.py can be used to further fine tune parameters for running the project.

The project can be run in 4 different ways:

```
$ ./run.sh arc_runner N10 0
```
Run Exisiting: In this case the user must provide the path to a file containing RGP instances that have already been generated. That file will be used to then generate SAT instances which will be solved by a SAT solver.

```
$ ./run.sh arc_runner N10 1
```
Preprocess & Solve: In this case the user must provide the path to a file containing the experiment configuration. This file is used to generate RGP instances which will th

```
$ ./run.sh arc_runner N10 2
```
Preprocess: In this case the user must provide the path to a file containing the experiment configuration and set the 'preprocess' flag to TRUE. This will enable the RGP instances to be converted to SAT instances which will be stored in json files. They will need to be solved separately

```
$ ./run.sh arc_runner N10 3
```
Solve Preprocessed: In this case the user will need to provide the paths to the preprocessed RGP instances (Stored SAT instances) and set the 'solve_preprocessed' flag to TRUE. This will enable the SAT instances to be solved directly from the JSON files where they were being stored

In addition to this the user is given a choice to:

```
$ Run the code in a serial or parallel manner
$ Plot the results in the form of a Cactus Plot
$ Mail the results to the user
```

## 📄 Publication Details

[Nikhill Vombatkere; Philip W.L. Fong – “A Family of Zero-Trust Access Control Models and Automated Policy Formulation” – Submitted on 16 Dec 2024 to the 15th ACM Conference on Data and Application Security and Privacy (CODASPY) 2025](INSERT_DOI_HERE)