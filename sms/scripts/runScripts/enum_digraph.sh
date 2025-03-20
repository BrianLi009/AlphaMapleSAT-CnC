#!/bin/bash

n=$1
echo "n $n"

TMPFile="foo.enc"
echo "use tmpfile $TMPFile"

cd ../encodings
python3 sat_graphs.py --dontsolve $n -i2o > $TMPFile && time ../../src_directed/graphgendirected -v $n  --allModels --cnf $TMPFile 
# 1, 3, 16, 218, 9608, 1540944 https://oeis.org/A000273