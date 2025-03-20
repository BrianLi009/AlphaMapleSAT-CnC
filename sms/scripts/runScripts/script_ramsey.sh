#!/bin/bash

pythonscript="./scripts/sat_ramsey.py"
TMPFile="./asdf"
graphgencpp="./build/src/graphgen"

n=18
x=4
y=4

time python3 $pythonscript -n $n  --ramsey $x $y  > $TMPFile # -
time  $graphgencpp -v $n --cnf $TMPFile --cutoff 20000 --frequency 5 \
    --printStats  --useCadical  --allModels # --symClauses sym.cnf