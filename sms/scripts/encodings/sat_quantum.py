#!/usr/bin/python

from itertools import combinations, permutations
import itertools
from operator import indexOf
from sys import *

from numpy import product
# from pysat.card import *

import counterImplementations
import ast
DEFAULT_COUNTER = "sequential"

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--vertices', '-n', type=int, required=True)
parser.add_argument('--minDegree', '-d', type=int)
parser.add_argument('--maxDegree', '-D', type=int)
parser.add_argument('--graphFile', type=str, help="File containing graphs as adjacency list in python format")
parser.add_argument('--nGraph', type=int, help="Which line should be selected in the graph file")
parser.add_argument('--numberOfEdges', type=int, help="Exact number of edges")
parser.add_argument("--basicCritical", action="store_true", help="discard some cases which are obviously not critical")

parser.add_argument('--partition', '-p', type=int, nargs="+", help="paritioning of vertices")
parser.add_argument('--degrees', type=int, nargs="+", help="degree of vertices of the partition")

parser.add_argument('--subgraphMinDegree', '-ds', type=int)


# python ./scripts/sat_colorExpansion.py -n 25 -s 15 -i 9 -d 4 -D 5 --graphFile critical_graphs.txt --nGraph 1

args = parser.parse_args()

print("c\targs:",args)
vargs = vars(args)
print("c\tactive args:",{x:vargs[x] for x in vargs if vargs[x] != None and vargs[x] != False})

n = args.vertices

V = range(n)

edgeListSubgraph = []
if args.graphFile:
    with open(args.graphFile) as fp:
        for i, line in enumerate(fp):
            if i == args.nGraph:
                edgeListSubgraph = ast.literal_eval(line.strip())
                break


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

#for i,j in combinations(V,2):
#    print(var_equal(i,j))

# for i,j in combinations(V,2):
#    print("c\t edge ", i,j, "to", var_edge(i,j))

#-------------------------create encoding-----------------------

# degree constraints
d = [] # d[i] is variable indicating whether i has maximum degree
dAll = [] # all Degrees
if not args.degrees: # only if not predefined degrees
    for i in V:
        counterVariables = counterImplementations.counterFunction([var_edge(i,j) for j in V if j != i], countUpto=args.maxDegree, vPool=vpool,
            clauses=constraints, atMost=args.maxDegree, atLeast=args.minDegree, type=DEFAULT_COUNTER)
        d.append(counterVariables[args.maxDegree - 1]) # only true if a vertex has maximum degree
        dAll.append(counterVariables)

commonNeighbor = { (i,j,k): vpool.id() for i,j in combinations(V,2) for k in set(V)-{i,j}}

for i,j in combinations(V,2):
    for k in set(V)-{i,j}:
        L = (i,j,k) 
        constraints.append([-commonNeighbor[L], +var_edge(i,k)])
        constraints.append([-commonNeighbor[L], +var_edge(j,k)])
        constraints.append([+commonNeighbor[L], -var_edge(i,k), -var_edge(j,k)])


# Neighborhoods are disjoint and one not a subset of the other
if args.basicCritical or args.critical:
    for i,j in permutations(V,2):
        # There must be a vertex adjecent to i which is not adjacent to j
        adjacentOnlyToI = []
        for k in V:
            if k == i or k == j: continue
            kIsAdjacentOnlyToI = vpool.id()
            constraints.append([+var_edge(i,k), -kIsAdjacentOnlyToI])
            constraints.append([-var_edge(j,k), -kIsAdjacentOnlyToI])
            adjacentOnlyToI.append(kIsAdjacentOnlyToI)
        constraints.append([+var_edge(i,j)] + adjacentOnlyToI)


if args.kColorable:
    colors = [[vpool.id() for _ in V] for _ in range(args.kColorable)]
    constraints.append([+colors[0][0]]) # vertex 0 has color 1
    # maybe fix all 3 colors buy "first triangle containing vertex 0"
    
    # at least one color
    for u in V:
        constraints.append([colors[r][u] for r in range(args.kColorable)])
    # adjacent once cannot have the same color
    for u1,u2 in combinations(V,2):
        for r in range(args.kColorable):
            constraints.append([-var_edge(i,j), -var_edge(u1,u2), -colors[r][u1], -colors[r][u2]])

if args.numberOfEdges:
    counterImplementations.counterFunction([var_edge(i,j) for i,j in combinations(V,2)], args.numberOfEdges, 
        vpool, constraints, atMost=args.numberOfEdges, atLeast=args.numberOfEdges, type=DEFAULT_COUNTER)


if args.partition:
	pos = 0
	for i in range(len(args.partition)):
		posNew = pos + args.partition[i]

		# Each vertex has exactly degree  args.degrees[i]
		for v in range(pos, posNew):
			degree = args.degrees[i]
			counterImplementations.counterFunction([var_edge(v,u) for u in V if u != v], countUpto=degree, vPool=vpool, clauses=constraints, atLeast=degree, atMost=degree, type=DEFAULT_COUNTER)
		pos = posNew

   
print("c\tTotal number of constraints:", len(constraints))
print("c\tTotal number of variables:", vpool.id())


print('c\tbegin of CNF')
for c in constraints:
    for x in c:
        if type(x) != int:
            exit("Error")
for c in constraints:
    print (' '.join(str(x) for x in c))
print('c\tend of CNF')

