#!/bin/bash

# Reorder parameters
n=$1      #order
solver=$2  #solver type
t=$3      #timeout in seconds
mode=$4    #solver mode (optional)
f=$5      #instance file name (moved to last)

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
        timeout $t ./cadical-ks/build/cadical-ks $f --order $n | tee $f.log
    else
        timeout $t ./cadical-ks/build/cadical-ks $f | tee $f.log
    fi
elif [ "$solver" = "-maplesat" ]; then
    if [ "$mode" = "-cas" ]; then
        timeout $t ./maplesat-ks/simp/maplesat_static $f -order=$n -no-pre -minclause -exhaustive=$f.exhaust | tee $f.log
    else
        timeout $t ./maplesat-ks/simp/maplesat_static $f -no-pre -minclause | tee $f.log
    fi
else
    echo "Invalid solver option. Use -cadical or -maplesat"
    exit 1
fi
