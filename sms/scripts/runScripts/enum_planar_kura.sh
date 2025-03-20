#!/bin/bash

n=$1
echo "n $n"

cd ../encodings
time ../../src/graphgen -v $n  --allModels -planar 1 --useCadical --checkSolutionInProp

