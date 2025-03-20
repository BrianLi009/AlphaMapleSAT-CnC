#!/bin/bash

n=$((2*$1)) 
echo "n $n"

TMPFile="foo.enc"
echo "use tmpfile $TMPFile"

cd ../encodings
python3 sat_graphs.py --dontsolve $n --Delta_upp 3 --delta_low 3 --connectivity 2 -i2o > $TMPFile && time ../../src/graphgen -v $n  --allModels --cnf $TMPFile -planar 1 --useCadical --checkSolutionInProp

