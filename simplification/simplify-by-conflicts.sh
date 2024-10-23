#!/bin/bash

# Option parsing
while getopts ":s" opt; do
  case $opt in
    s) s=true ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
  esac
done

shift $((OPTIND -1))

# Ensure parameters are specified on the command-line
if [ -z "$3" ]; then
  echo "Need filename, order, and the number of conflicts for which to simplify"
  exit
fi

f=$1 # Filename
o=$2 # Order
m=$3 # Number of conflicts
e=$((o*(o-1)/2)) # Number of edge variables

# Create necessary directories
mkdir -p log

f_dir=$f
f_base=$(basename "$f")

# Simplify m seconds
echo "simplifying for $m conflicts"

./cadical-ks/build/cadical-ks "$f_dir" --order $o -o "$f_dir".simp1 -e "$f_dir".ext -n -c $m | tee "$f_dir".simplog

# Output final simplified instance
./gen_cubes/concat-edge.sh $o "$f_dir".simp1 "$f_dir".ext > "$f_dir".simp
rm -f "$f_dir".simp1
