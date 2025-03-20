#!/bin/bash

pythonscript="./scripts/encodings/sat_efl.py"
TMPFile="./tmp$$.enc"
graphgencpp="./build/src/graphgen"
prooffile="./proof$$.drat"
clausefile="./clause$$.dimacs"
clausesWithHeader="./clauses2$$.dimacs"

n=$1
m=$2
chi=$(($n + 1))

time python3 $pythonscript -n $n -m $m --primary "hypergraph"  --chi3 $chi > $TMPFile # --chi3 $chi  --chi3 $chi --codishSym
time  $graphgencpp -b $n $m --nonhec --dimacs $TMPFile --cutoff 200000 --frequency 20 \
     --printStats  --useCadical --allModels -chi $chi --coloringAlgo 2  --printAddedClauses $clausefile
./scripts/other/createDimacs $clausefile > $clausesWithHeader
echo "Start producing proof"
./cadical-internal-theory-propagation/build/cadical $clausesWithHeader $prooffile --unsat -q
echo "Check proof"
drat-trim $clausesWithHeader $prooffile

rm $TMPFile
rm $prooffile
rm $clausefile