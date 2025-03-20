#!/bin/bash

m=$1 # number of edges
echo "m $m"
n=$((2*$m/3)) # number of vertices fulfills 3|V| <= 2|E| because mindeg >= 3. 
echo "n $n"

TMPFile="foo.enc"
echo "use tmpfile $TMPFile"

cd ../encodings
python3 sat_graphs.py --dontsolve $n --connectivity 1 --delta_low 3 --num_edges_upp $m --num_edges_low $m -i2o > $TMPFile && time ../../src/graphgen -v $n  --allModels --cnf $TMPFile -planar 1 --useCadical --checkSolutionInProp

