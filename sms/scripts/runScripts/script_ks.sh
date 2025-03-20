#!/bin/bash

TMPFile="./enc.tmp"
graphgencpp="./build/src/graphgen"
pythonscript=./scripts/encodings/kochen_specker.py

n=$1
echo "Generate encoding"
python3 $pythonscript -n $n  -x $n > $TMPFile
echo "Start solving"
$graphgencpp -v $n --cnf $TMPFile --cutoff 200000 --frequency 30 --printStats --checkSolutionInProp --allModels ${@:2}