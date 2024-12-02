#!/bin/bash

# Ensure parameters are specified on the command-line
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 filename order conflicts [-cas]"
    echo "Need filename, order, and the number of conflicts for which to simplify"
    exit 1
fi

f=$1 # Filename
o=$2 # Order
m=$3 # Number of conflicts
mode=$4 # Optional -cas parameter

# Create necessary directories
mkdir -p log

f_dir=$f

# Simplify for m conflicts
echo "simplifying for $m conflicts"

if [ "$mode" = "-cas" ]; then
    ./cadical-ks/build/cadical-ks "$f_dir" --order $o -o "$f_dir".simp1 -e "$f_dir".ext -n -c $m | tee "$f_dir".simplog
else
    ./cadical-ks/build/cadical-ks "$f_dir" -o "$f_dir".simp1 -e "$f_dir".ext -n -c $m | tee "$f_dir".simplog
fi

# Output final simplified instance
./gen_cubes/concat-edge.sh $o "$f_dir".simp1 "$f_dir".ext > "$f_dir".simp
rm -f "$f_dir".simp1
