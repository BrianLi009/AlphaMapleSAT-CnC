#!/bin/bash


n=12

pythonscript="./scripts/sat_diameter2critical.py"
TMPFile="./asdf"

graphgencpp="./build/src/graphgen"

CMD1="python3 $pythonscript -n $n  "
CMD2="$graphgencpp -v $n --cnf $TMPFile --cutoff 20000 --frequency 5 --printStats --allModels " # --symClauses sym.cnf

echo $CMD1
echo $CMD2

time $CMD1 > $TMPFile
time $CMD2
