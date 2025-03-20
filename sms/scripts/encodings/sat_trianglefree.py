#!/usr/bin/python

from itertools import *
from sys import *
from time import perf_counter

from numpy import product
# from pysat.card import *

import counterImplementations
DEFAULT_COUNTER = "sequential"

import argparse
parser = argparse.ArgumentParser()

# -----------------------------basic------------------------------------------
parser.add_argument('--vertices', '-n', type=int, required=True, help="Number of vertices")
parser.add_argument('--codish', '-c', action="store_true", help="Apply codish like symmetry breaking")

t_begin = perf_counter()


args = parser.parse_args()

print("c\targs:",args, file=stderr)
vargs = vars(args)
print("c\tactive args:",{x:vargs[x] for x in vargs if vargs[x] != None and vargs[x] != False}, file=stderr)

n = args.vertices
V = range(n)


all_variables = []
all_variables += [('edge',(u,v)) for u,v in combinations(V,2)]
all_variables_index = {}

_num_vars = 0
for v in all_variables:
    _num_vars += 1
    all_variables_index[v] = _num_vars

# print(all_variables_index)

class IDPool:
	def __init__(self, start_from = 0) -> None:
		self.start_from = start_from
	start_from = 1

	def id(self):
		x = self.start_from
		self.start_from += 1
		return x

vpool = IDPool(start_from=_num_vars+1)  # for auxiliary pysat variables

def var(L):	return all_variables_index[L]
def var_edge(u,v): return var(('edge',(min(u,v), max(u,v))))


constraints = []


# triangle free
for v1, v2, v3 in combinations(V, 3):
    constraints.append([-var_edge(v1,v2), -var_edge(v2,v3), -var_edge(v1,v3)])

if args.codish:
    for v,u in combinations(V, 2):
            allPreviousEqual = vpool.id()
            constraints.append([+allPreviousEqual])

            for w in range(n - 1):
                if w in [u,v]:
                    continue
                constraints.append([-allPreviousEqual, -var_edge(v,w), +var_edge(u,w)]) # if all previous are equal then no edge to first vertex or an edge to second
                allPreviousEqualNew = vpool.id()
                constraints.append([-allPreviousEqual, -var_edge(v,w), +allPreviousEqualNew])
                constraints.append([-allPreviousEqual, +var_edge(u,w), +allPreviousEqualNew])
                allPreviousEqual = allPreviousEqualNew


       
print("c\tTotal number of constraints:", len(constraints), file=stderr)
print("c\tTotal number of variables:", vpool.id(), file=stderr)



print('c\tstart writting encoding to file', file=stderr)

print("\n".join(" ".join(map(str, C)) for C in constraints)) # performs best
print('c\tencoding finished', file=stderr)


t_tot = perf_counter()
print(f"Encoding time: {perf_counter()-t_begin:.4g}", file=stderr)
