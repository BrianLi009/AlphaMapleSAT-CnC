#!/bin/bash

pythonscript="./scripts/encodings/sat_efl.py"
TMPFile="./tmp$$.enc"
graphgencpp="./build/src/graphgen"

n=$1
m=$2
c=$3 # max closed neighborhood
chi=$(($c + 1))

time python3 $pythonscript -n $n -m $m --primary "hypergraph"  --chi3 $chi --maxClosedNeighborhood $c --deactivateCovering  > $TMPFile # --chi3 $chi  --chi3 $chi --codishSym
time $graphgencpp -b $n $m --nonhec --dimacs $TMPFile --cutoff 200000 --frequency 20 \
     --printStats  --useCadical --allModels -chi $chi --coloringAlgo 2  

rm $TMPFile