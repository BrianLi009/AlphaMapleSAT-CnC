#!/bin/bash

n=$1 #order
f=$2 #instance file name
solver=$3 #solver type
t=$4 #timeout in seconds
mode=$5 #solver mode (optional)

[ "$1" = "-h" -o "$1" = "--help" -o "$#" -lt 4 ] && echo "
Description:
    Script for solving and generating drat proof for instance

Usage:
    ./solve.sh n f solver t [mode]

Arguments:
    <n>: the order of the instance/number of vertices in the graph
    <f>: file name of the CNF instance to be solved
    <solver>: -cadical or -maplesat
    <t>: timeout in seconds
    [mode]: -cas (optional)
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
        timeout $t ./maplesat-ks/simp/maplesat_static $f -order=$n -no-pre -minclause | tee $f.log
    else
        timeout $t ./maplesat-ks/simp/maplesat_static $f -no-pre -minclause | tee $f.log
    fi
else
    echo "Invalid solver option. Use -cadical or -maplesat"
    exit 1
fi