#!/bin/bash

# Function to display help information
function display_help() {
    echo "
    Description:
        This script will install and compile all required dependencies and packages, including maplesat-ks, cadical, networkx, z3-solver, and march_cu from cube and conquer

    Usage:
        ./dependency-setup.sh 
    "
    exit
}

# Check if help is requested
[ "$1" = "-h" -o "$1" = "--help" ] && display_help

echo "Prerequisite: pip and make installed"

required_version="2.5"
installed_version=$(pip3 show networkx | grep Version | awk '{print $2}')

# Check if networkx needs to be updated
if [[ "$(printf '%s\n' "$installed_version" "$required_version" | sort -V | head -n1)" != "$required_version" ]]; then
    echo "Need to install networkx version newer than $required_version"
    pip3 install --upgrade networkx
    echo "Networkx has been successfully updated."
else
    echo "Networkx version $installed_version is already installed and newer than $required_version"
fi

# Check if coloredlogs exists
if ! python3 -c "import coloredlogs" &> /dev/null; then
    echo "Installing coloredlogs..."
    pip3 install coloredlogs
else
    echo "coloredlogs is already installed."
fi

#tqdm
if ! python3 -c "import tqdm" &> /dev/null; then
    echo "Installing tqdm..."
    pip3 install tqdm
else
    echo "tqdm is already installed."
fi

#wandb
if ! python3 -c "import wandb" &> /dev/null; then
    echo "Installing wandb..."
    pip3 install wandb
else
    echo "wandb is already installed."
fi


# Check if march_cu is present
if [ -f gen_cubes/march_cu/march_cu ]
then
    echo "March installed and binary file compiled"
else
    cd gen_cubes/march_cu
    make
    cd -
fi

# Check if cadical-ks is present
if [ -f cadical-ks/build/cadical-ks ]
then
    echo "Cadical-ks installed and binary file compiled"
else
    cd cadical-ks
    ./configure
    make
    cd -
fi

# Install maplesat-ks
if [ -d maplesat-ks ] && [ -f maplesat-ks/simp/maplesat_static ]
then
    echo "Maplesat-ks installed and binary file compiled"
else
    cd maplesat-ks
    make
    cd -
fi 

echo "All dependencies properly installed"