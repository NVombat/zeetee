#!/bin/bash

# Main Command : python -m src.filename [optional args]

# Get path to this script
script_dir="$(dirname "$(realpath "$0")")"

# Directory where Python files are located relative to the script
src_dir="$script_dir/src"
echo "Source directory: $src_dir"

# Check if the filename argument is provided
if [ -z "$1" ]; then
    echo "Usage: ./run.sh filename [optional args]"
    exit 1
fi

# Get the filename provided by the user
filename="$1"

echo "Running File: $filename"

# If the filename contains .py, check if the exact file exists
if [[ "$filename" == *.py ]]; then
    # Check if the file exists with the .py extension
    if [ ! -f "$src_dir/$filename" ]; then
        echo "Error: File $filename does not exist in $src_dir"
        exit 1
    fi
    # Strip .py for later use
    filename="${filename%.py}"
else
    # Add .py extension and check if the file exists
    if [ ! -f "$src_dir/$filename.py" ]; then
        echo "Error: File $filename.py does not exist in $src_dir"
        exit 1
    fi
fi

# Shift arguments so that $2 becomes the first optional arg
shift

# Files that require additional arguments
if [[ "$filename" == "solver" ]]; then
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "Usage for $filename.py: ./run.sh $filename enc_type solver"
        echo "  enc_type must be an integer (1 or 2)"
        echo "  solver must be an integer (1 or 2)"
        exit 1
    fi

    enc_type="$1"
    solver="$2"

    # Ensure enc_type and solver are either 1 or 2
    if ! [[ "$enc_type" =~ ^[12]$ ]] || ! [[ "$solver" =~ ^[12]$ ]]; then
        echo "Error: enc_type and solver must both be 1 or 2."
        exit 1
    fi

    # Run solver.py with the provided arguments
    echo "Running poetry run python -m src.$filename $enc_type $solver"
    poetry run python -m "src.$filename" "$enc_type" "$solver"
else
    # Run other Python files without additional arguments
    echo "Running poetry run python -m src.$filename"
    poetry run python -m "src.$filename"
fi