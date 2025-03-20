#!/bin/bash

pythonscript="./scripts/encodings/sat_ck_free.py"
TMPFile="./enc.tmp"
graphgencpp="./build/src/graphgen"

n=$1
m=$2
c=$3

planEnc=$4

if [[ "$planEnc" == "kur" ]]
then
    time python3 $pythonscript -n $n  -m $m  -c $c > $TMPFile # -
    time  $graphgencpp -v $n --cnf $TMPFile --cutoff 20000 --frequency 5 \
        --printStats  --useCadical  --planarityFrequency 5 # --symClauses sym.cnf
elif [[ "$planEnc" == "ord" ]]
then
    time python3 $pythonscript -n $n  -m $m  -c $c --planar_order > $TMPFile # -
    time  $graphgencpp -v $n --cnf $TMPFile --cutoff 20000 --frequency 5 \
        --printStats  --useCadical  # --symClauses sym.cnf

else
    echo "encoding type not given or encoding $planEnc not supported"
fi
