#!/usr/bin/python

from counterImplementations import *

import argparse
from itertools import combinations
# from pysat.card import *

DEFAULT_COUNTER = "sequential"

parser = argparse.ArgumentParser()
parser.add_argument('--vertices', '-n', type=int, required=True)
parser.add_argument('--ramsey', type=int, nargs="+", required=True)

args = parser.parse_args()

n = args.vertices
V = list(range(n))

edge_vars = [('edge',(u,v)) for u,v in combinations(V,2)]
edge_vars_index = {}

_num_vars = 0
for v in edge_vars:
    _num_vars += 1
    edge_vars_index[v] = _num_vars

# print(all_variables_index)

class IDPool:
	def __init__(self, start_from = 0) -> None:
		self.start_from = start_from
	start_from = 1

	def id(self):
		x = self.start_from
		self.start_from += 1
		return x

def var(L):	return edge_vars_index[L]
def var_edge(u,v): return var(('edge',(min(u,v), max(u,v))))

# we will need auxiliary variables for each triangle
# these will be indexed through the combinatorial number system
# and shifted by the number of edge variables

vpool = IDPool(start_from=_num_vars+1)  # for auxiliary pysat variables
constraints = []
# ----------------------------- start encoding ----------------------------------------------------

for S in combinations(V, args.ramsey[0]):
    constraints.append([-var_edge(i,j) for i,j in combinations(S,2)])

for S in combinations(V, args.ramsey[1]):
    constraints.append([var_edge(i,j) for i,j in combinations(S,2)])



print("c\tTotal number of constraints:", len(constraints))
print("c\tTotal number of variables:", vpool.id())

print('c\tbegin of CNF')
for c in constraints:
    print(' '.join(str(x) for x in c))
print('c\tend of CNF')
