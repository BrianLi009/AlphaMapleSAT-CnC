#!/usr/bin/python

from itertools import combinations, permutations
from sat_planar import *
from sys import *
from counterImplementations import *
DEFAULT_COUNTER = "sequential"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--vertices', '-n', type=int, required=True)
parser.add_argument('--minEdges', '-m', type=int, required=True, help="Minimum number of edges")
parser.add_argument('--cycle', '-c', type=int, required=True, help="Length of forbidden cycle")
parser.add_argument('--planar_order', help="order based planarity encoding (standard)", action="store_true")
parser.add_argument('--planar_order_heavy', help="order based planarity encoding (heavy)", action="store_true")

args = parser.parse_args()

print("c\targs:",args)

n = args.vertices

V = range(n)

all_variables = []
all_variables += [('edge',(u,v)) for u,v in combinations(V,2)]
all_variables_index = {}

_num_vars = 0
for v in all_variables:
    _num_vars += 1
    all_variables_index[v] = _num_vars
def var(L):	return all_variables_index[L]
def var_edge(u,v): return var(('edge',(min(u,v), max(u,v))))

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

constraints = []
# ----------------------------- start encoding ----------------------------------------------------
def CNF_OR(ins, out):
    return [[-out] + ins] + [[out, -x] for x in ins]

def CNF_AND(ins, out):
    return [[out] + [-x for x in ins]] + [[-out, x] for x in ins]


for v in V:
    constraints.append([var_edge(v,u) for u in V if u != v]) # at least on edge

counterFunction([var_edge(v1, v2) for v1, v2 in combinations(V,2)], args.minEdges, vpool, constraints, atLeast=args.minEdges)

# forbid cycles with specific length
for cycle in permutations(V,args.cycle):
    if cycle[0] != min(cycle):
        continue
    if cycle[1] > cycle[args.cycle - 1]:
        continue
    constraints.append([-var_edge(cycle[i], cycle[i + 1]) for i in range(args.cycle - 1)] + [-var_edge(cycle[0], cycle[-1])]) # forbid cycle

if args.planar_order:
    planar_encoding(V, var_edge, vpool, constraints)

if args.planar_order_heavy:
    planar_encoding_heavy(V, var_edge, vpool, constraints)

print("c\tTotal number of constraints:", len(constraints))
print("c\tTotal number of variables:", vpool.id())

print('c\tbegin of CNF')
for c in constraints:
    print (' '.join(str(x) for x in c))
print('c\tend of CNF')










