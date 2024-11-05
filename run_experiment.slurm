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

./run.sh arc_runner