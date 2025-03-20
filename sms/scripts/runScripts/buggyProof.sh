#!/bin/bash

# small proof where clauses are delted which don't exist in DRAT proof.

pythonscript="./scripts/encodings/kochen_specker.py"
TMPFile="./asdf"
graphgencpp="./build/src/graphgen"
prooffile="./proof.drat"
clausefile="./clausesDimacs"
clausesWithHeader="./clauses.dimacs"

n=12
# n=19

time python3 $pythonscript -n $n -x $n > $TMPFile # --chi3 $chi  --chi3 $chi --codishSym
time  $graphgencpp -v $n --cnf $TMPFile --cutoff 200000 --frequency 20 \
     --printStats  --useCadical --allModels --checkSolutionInProp  --printAddedClauses $clausefile --proof $prooffile

./scripts/other/createDimacs $clausefile > $clausesWithHeader # Add "p cnf ..." header to the clause file
drat-trim $clausesWithHeader $prooffile # check the proof