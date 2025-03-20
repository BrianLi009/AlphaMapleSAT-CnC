#!/bin/bash

resultFolder=./sms-cpp-impl-planar

rm -r $resultFolder
mkdir -p $resultFolder/scripts/encodings
mkdir -p $resultFolder/scripts/runScripts
mkdir -p $resultFolder/scripts/other

cp -r ./cadical-internal-theory-propagation $resultFolder/cadical-internal-theory-propagation
cp -r ./src $resultFolder/src
cp ./README.md $resultFolder/README.md
cp ./buildScript.sh $resultFolder
cp ./CMakeLists.txt $resultFolder
cp ./scripts/encodings/sat_graphs.py $resultFolder/scripts/encodings/
cp ./scripts/encodings/counterImplementations.py $resultFolder/scripts/encodings/
cp ./scripts/encodings/sat_ck_free.py $resultFolder/scripts/encodings/
cp ./scripts/encodings/sat_planar.py $resultFolder/scripts/encodings/
cp ./scripts/encodings/sat_earth-moon.py $resultFolder/scripts/encodings/
cp ./scripts/runScripts/enum_* $resultFolder/scripts/runScripts/
cp ./scripts/runScripts/script_earthmoon.sh $resultFolder/scripts/runScripts/
cp ./scripts/runScripts/script_ck_free.sh $resultFolder/scripts/runScripts/
cp ./scripts/other/createDimacsHeader.cpp $resultFolder/scripts/other/