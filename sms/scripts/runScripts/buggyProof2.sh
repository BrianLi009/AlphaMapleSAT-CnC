#!/bin/bash

# relatively small proof where DRAT-trim fails. IPASIR interface is used for excluding models.

pythonscript="./scripts/encodings/sat_erdos_faber_lovasz.py"
TMPFile="./asdf"
graphgencpp="./build/src/graphgen"
prooffile="./proof.drat"
clausefile="./clausesDimacs"
clausesWithHeader="./clauses.dimacs"
symClauseFile="./symClauses"

n=15
l=10
chi=$(($l + 1))


time python3 $pythonscript -n $n -l $l --primary "hypergraph"   > $TMPFile # --chi3 $chi  --chi3 $chi --codishSym
time  $graphgencpp -b $l $n --nonhec --cnf $TMPFile --cutoff 200000 --frequency 20 \
     --printStats  --useCadical --allModels \
     --printAddedClauses $clausefile --proof $prooffile # -chi $chi --printPartial 100 # --symClauses sym.cnf

./scripts/other/createDimacs $clausefile > $clausesWithHeader

drat-trim $clausesWithHeader $prooffile