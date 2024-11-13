#!/bin/bash

# Get path to this script
script_dir="$(dirname "$(realpath "$0")")"

# Directory where log files are located
log_dir="$script_dir/data/logfiles"

# Directory where job specific log files are located
slurm_log_dir="$script_dir/data/slurm"

# Directory where experiment assets are located
asset_dir="$script_dir/src/assets"

# Find and delete files in the log directory
find "$log_dir" -type f -exec rm -f {} \;

# Find and delete only .log files in the script_dir
find "$slurm_log_dir" -type f -name "*.log" -exec rm -f {} \;

# Find and delete files in the asset directory
find "$asset_dir" -type f -exec rm -f {} \;