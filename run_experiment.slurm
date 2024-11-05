#!/bin/bash

# ---------------------------------------------------------------------
#SBATCH --job-name=zeetee
#SBATCH --mail-user=nikhill.vombatkere@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=500gb
#SBATCH --time=30:00:00
#SBATCH --partition=bigmem

# ---------------------------------------------------------------------
module load python/3.12.5

# ---------------------------------------------------------------------
chmod u+x run.sh

./run.sh arc_runner