#!/bin/bash

n=$((2*$1)) 
echo "n $n"

TMPFile="foo.enc"
echo "use tmpfile $TMPFile"

cd ../encodings
python3 sat_graphs.py $n --connectivity 1 --omega_upp 2 --delta_low 3 --Delta_upp 3 -i2o --dontsolve > $TMPFile && time ../../src/graphgen -v $n  --allModels --cnf $TMPFile -planar 1 --useCadical --checkSolutionInProp

