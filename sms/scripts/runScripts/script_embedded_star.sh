#!/bin/bash

minDegree=3
maxDegree=5
n=16
m=29

# -v 29 --nEdges 45 --minDegree 2 --maxDegree 4

pythonscript="./scripts/sat_girth_extremal.py"
TMPFile="./asdf"

graphgencpp="./build/src/graphgen"

# time python $pythonscript -n $n -m $m --minDegree $minDegree --maxDegree $maxDegree --girth 5   --embeddedStar > $TMPFile
# time  $graphgencpp -v $n --cnf $TMPFile --cutoff 20000 --frequency 5 --printStats  --embeddedStar $minDegree $maxDegree --timeout 10 # --symClauses sym.cnf

# time python $pythonscript -n $n -m $m --minDegree $minDegree --maxDegree $maxDegree --girth 7 --compactGirthConstraints    > $TMPFile
# time $graphgencpp -v $n --cnf $TMPFile --cutoff 20000 --frequency 5 --printStats   # --symClauses sym.cnf


# time python $pythonscript -n $n -m $m --minDegree $minDegree --maxDegree $maxDegree --girth 7 --compactGirthConstraints   --embeddedStar2 --initialVertexOrderings qwer > $TMPFile
# time $graphgencpp -v $n --cnf $TMPFile --cutoff 20000 --frequency 5 --printStats   --vertexOrderingsFile qwer # --symClauses sym.cnf

time python $pythonscript -n $n -m $m --minDegree $minDegree --maxDegree $maxDegree --codish > $TMPFile
time $graphgencpp -v $n --cnf $TMPFile --cutoff 20000 --frequency 5 --printStats --turnoffSMS