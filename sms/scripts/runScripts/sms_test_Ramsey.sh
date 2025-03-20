#!/bin/bash

a=$1
b=$2
n=$3
echo "k $k"
echo "n $n"

TMPFile="R$a$b$n.enc"
echo "use tmpfile $TMPFile"

python3 sat_graphs.py --dontsolve $n --omega_upp $(($a-1)) --alpha_upp $(($b-1)) -i2o > $TMPFile && time ../src/graphgen -v $n  --allModels --cnf $TMPFile
