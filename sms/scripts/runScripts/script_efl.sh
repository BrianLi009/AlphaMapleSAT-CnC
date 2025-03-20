#!/bin/bash

pythonscript="./scripts/encodings/sat_efl.py"
TMPFile="./tmp$$.enc"
graphgencpp="./build/src/graphgen"

n=$1
m=$2
chi=$(($n + 1))


time python3 $pythonscript -n $n -m $m --primary "hypergraph"  --chi3 $chi > $TMPFile
time  $graphgencpp -b $n $m --nonhec --dimacs $TMPFile --cutoff 200000 --frequency 20 \
     --printStats  --useCadical --allModels -chi $chi --coloringAlgo 2  

rm $TMPFile