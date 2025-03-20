#!/bin/bash

n=$1
echo "n $n"

TMPFile="EG$n.enc"
echo "use tmpfile $TMPFile"

python3 sat_graphs.py --dontsolve $n --ErdosGyarfas -i2o > $TMPFile && time ../src/graphgen -v $n  --allModels --cnf $TMPFile
