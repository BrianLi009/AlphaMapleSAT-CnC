#!/bin/bash

k=$1
n=$((4*$k-1))
echo "k $k"
echo "n $n"

TMPFile="GS$k.enc"
echo "use tmpfile $TMPFile"

python3 sat_graphs.py --dontsolve $n -GS $k -i2o > $TMPFile && time ../src/graphgen -v $n  --allModels --cnf $TMPFile
