pythonscript="./scripts/encodings/sat_earth-moon.py"
TMPFile="./enc$$.tmp"
graphgencpp="./build/src/graphgendirected"

n=$1
chi=$2
planEnc=$3

if [[ "$planEnc" == "kur" ]]
then
    time python3 $pythonscript --n $n --chi $chi --triangulation --explizitlyForbidSubgraph --explicitColoring > $TMPFile # -
    time   $graphgencpp -v $n   --cutoff 20000 --frequency 5 --printStats  --useCadical --allModels --cnf $TMPFile --thickness2 5 --checkSolutionInProp  # --allModels  --printFullMatrix #  #   --symClauses sym.cnf
    rm $TMPFile
elif [[ "$planEnc" == "ord" ]]
then
    time python3 $pythonscript --n $n --chi $chi --triangulation --explizitlyForbidSubgraph --explicitColoring --planar_order > $TMPFile # -
    time   $graphgencpp -v $n   --cutoff 20000 --frequency 5 --printStats  --useCadical --allModels --cnf $TMPFile --checkSolutionInProp  # --allModels  --printFullMatrix #  #   --symClauses sym.cnf
    rm $TMPFile
else
    echo "encoding type not given or encoding $planEnc not supported"
fi

