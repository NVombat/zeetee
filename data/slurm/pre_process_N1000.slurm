#!/bin/bash

# ---------------------------------------------------------------------
#SBATCH --job-name=zeetee_preprocess_%j
#SBATCH --output=../logfiles/zt_output_preprocess_%j.log
#SBATCH --error=../logfiles/zt_error_preprocess_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256gb
#SBATCH --time=1-00:00:00
#SBATCH --partition=bigmem
#SBATCH --exclude=mmg1,mmg2
#SBATCH --mail-user=nikhill.vombatkere@ucalgary.ca
#SBATCH --mail-type=begin,end,fail

# ---------------------------------------------------------------------
echo "PRE-PROCESS EXPERIMENT - OPERATION FLAG set to 2..."

# ---------------------------------------------------------------------
echo "Loading python-3.12.5"
module load python/3.12.5
echo "Loaded python-3.12.5"

cd ../..

# ---------------------------------------------------------------------
# Ensure Poetry is available in the PATH
export PATH="$HOME/.local/bin:$PATH"

# Check if Poetry is installed, and install if not
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "Poetry is already installed."
fi

# Install dependencies (only if not already installed in the Poetry environment)
poetry install

# ---------------------------------------------------------------------
chmod u+x run.sh

./run.sh arc_runner N1000 2

echo "Experiment Pre-Processing Complete!"