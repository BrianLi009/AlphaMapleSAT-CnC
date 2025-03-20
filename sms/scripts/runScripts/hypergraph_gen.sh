#!/bin/bash

m=${1-7}
k=${2-3}


if [[ $k -lt 2 ]]; then
	echo "$k must be at least $2"
fi

n=$((2*k-1))
h=$(python3 -c "from math import comb; print(comb($n,$k))")
u=$m #TODO figure out the correct upper bound

if [[ $m == $h ]]; then
	u=$n
fi


# update the value of cores here
cores=2
seq $n $u | parallel --jobs $cores\
   	cnf=hyper_n{}_k${k}_m${m}.cnf "&&"\
   	./hypergraph_gen.py -n {} -k $k -m $m ">" '$cnf' "&&"\
   	./hypergraphgen --vertices {} --uniform $k --cnf '$cnf' --printStats "&&"\
	echo
