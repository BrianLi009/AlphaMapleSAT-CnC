#!/bin/bash

# Function to display help information
display_help() {
    echo "
Description:
    This script will install and compile all required dependencies and packages, 
    including maplesat-ks, cadical, networkx, z3-solver, and march_cu from 
    cube and conquer.

Usage:
    ./dependency-setup.sh
"
    exit 0
}

# Check if help is requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    display_help
fi

echo "Prerequisite: pip and make installed"
echo

required_version="2.5"
# Handling cases where networkx might not be installed
installed_version=$(pip3 show networkx | grep Version | awk '{print $2}' || echo "0.0")

# Check if networkx needs to be updated
if [[ "$(printf '%s\n' "$installed_version" "$required_version" | sort -V | head -n1)" != "$required_version" ]]; then
    echo "Networkx version $installed_version is older than required version $required_version. Installing/upgrading..."
    pip3 install --upgrade networkx
    echo "Networkx has been successfully updated."
else
    echo "Networkx version $installed_version is already installed and meets requirements."
fi
echo

# Check if march_cu is present
if [ -f "gen_cubes/march_cu/march_cu" ]; then
    echo "march_cu is already installed and compiled."
else
    echo "Installing march_cu..."
    (cd gen_cubes/march_cu && make)
    echo "march_cu has been installed."
fi
echo

# Check if cadical-ks is present
if [ -f "cadical-ks/build/cadical-ks" ]; then
    echo "cadical-ks is already installed and compiled."
else
    echo "Installing cadical-ks..."
    (cd cadical-ks && ./configure && make)
    echo "cadical-ks has been installed."
fi
echo

# Install maplesat-ks
if [ -d "maplesat-ks" ] && [ -f "maplesat-ks/simp/maplesat_static" ]; then
    echo "Maplesat-ks is already installed and compiled."
else
    echo "Installing maplesat-ks..."
    (cd maplesat-ks && make)
    echo "maplesat-ks has been installed."
fi
echo

# Setup for AlphaMapleSAT
echo "Setting up AlphaMapleSAT..."

# Clone AlphaMapleSAT repository if it doesn't exist
if [ ! -d "AlphaMapleSAT" ]; then
    echo "Cloning AlphaMapleSAT repository..."
    git clone https://github.com/piyush-J/AlphaMapleSAT.git
else
    echo "AlphaMapleSAT repository already exists."
fi

# Install AlphaMapleSAT requirements
if [ -f "AlphaMapleSAT/alphamaplesat/requirements.txt" ]; then
    echo "Installing AlphaMapleSAT requirements..."
    pip3 install -r AlphaMapleSAT/alphamaplesat/requirements.txt
    echo "AlphaMapleSAT setup complete."
else
    echo "Error: AlphaMapleSAT/alphamaplesat/requirements.txt not found."
    exit 1
fi
echo

echo "All dependencies properly installed"