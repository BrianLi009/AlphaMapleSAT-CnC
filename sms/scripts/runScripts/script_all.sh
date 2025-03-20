#!/bin/bash
graphgencpp="./build/src/graphgen"

$graphgencpp -v $1 --frequency 5 --printStats --allModels ${@:2}