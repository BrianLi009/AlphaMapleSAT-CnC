#!/bin/bash

n=$1 
echo "n $n"

TMPFile="foo.enc"
echo "use tmpfile $TMPFile"

cd ../encodings
python3 sat_graphs.py $n --evendegrees -i2o --dontsolve > $TMPFile && time ../../src/graphgen -v $n  --allModels --cnf $TMPFile -planar 1 --useCadical --checkSolutionInProp

