#!/bin/bash

# Reorder parameters
n=$1      #order
solver=$2  #solver type
t=$3      #timeout in seconds
f=$4      #instance file name (if no mode provided)

# Check if 5th parameter exists - if so, f is 5th param and mode is 4th
if [ $# -eq 5 ]; then
    mode=$4    #solver mode
    f=$5      #instance file name
fi

[ "$1" = "-h" -o "$1" = "--help" -o "$#" -lt 4 ] && echo "
Description:
    Script for solving and generating drat proof for instance

Usage:
    ./solve.sh n solver t [mode] f

Arguments:
    <n>: the order of the instance/number of vertices in the graph
    <solver>: -cadical or -maplesat
    <t>: timeout in seconds
    [mode]: -cas (optional)
    <f>: file name of the CNF instance to be solved
" && exit

# Select solver based on arguments
if [ "$solver" = "-cadical" ]; then
    if [ "$mode" = "-cas" ]; then
        timeout $t ./cadical-ks/build/cadical-ks $f --order $n --proofsize 7168 | tee $f.log
    else
        timeout $t ./cadical-ks/build/cadical-ks $f --proofsize 7168 | tee $f.log
    fi
elif [ "$solver" = "-maplesat" ]; then
    if [ "$mode" = "-cas" ]; then
        timeout $t ./maplesat-ks/simp/maplesat_static $f -order=$n -no-pre -minclause -exhaustive=$f.exhaust -max-proof-size=7168 | tee $f.log
    else
        timeout $t ./maplesat-ks/simp/maplesat_static $f -no-pre -max-proof-size=7168 | tee $f.log
    fi
else
    echo "Invalid solver option. Use -cadical or -maplesat"
    exit 1
fi
