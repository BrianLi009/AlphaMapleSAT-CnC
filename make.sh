#!/bin/bash
cd cadical-ks/build/
mpic++ -W -O -DNDEBUG -I../build ../src/dist-solve.cpp -L. -lcadical -o dist-solve
mv dist-solve ../../
cd ../../
