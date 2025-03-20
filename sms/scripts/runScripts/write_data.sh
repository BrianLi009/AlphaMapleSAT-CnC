#/bin/bash

local_solfile=$1
local_timefile=$2
solfile=$3
timefile=$4

cat "$local_solfile" >> "$solfile"
cat "$local_timefile" >> "$timefile"
