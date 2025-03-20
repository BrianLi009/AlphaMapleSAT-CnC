#!/bin/bash

n=$1
echo "n $n"

TMPFile="foo.enc"
echo "use tmpfile $TMPFile"

cd ../encodings
python3 sat_graphs.py --dontsolve $n --planar_universal -i2o > $TMPFile && time ../../src/graphgen -v $n  --allModels --cnf $TMPFile --useCadical --checkSolutionInProp
