#!/usr/bin/python

import argparse
import counterImplementations

from ast import arguments
from itertools import combinations, permutations
from sys import *
# from pysat.card import *
import math

DEFAULT_COUNTER = "totalizer"

parser = argparse.ArgumentParser()
parser.add_argument('--vertices', '-n', type=int, required=True)
parser.add_argument('--codish', "-c", action="store_true",
                    help="apply codish like symmetry breaking")
parser.add_argument('--partition', '-p', type=int,
                    nargs="+", help="paritioning of vertices")
parser.add_argument('--degrees', '-d', type=int, nargs="+",
                    help="degree of vertices of the partition")
parser.add_argument('--different', action="store_true",
                    help="Vertices can not have the same neighborhood or one subset of the other")
parser.add_argument('--maxDegree', '-D', type=int)
parser.add_argument('--edges', '-m', type=int)
parser.add_argument('--triangular', action="store_true",
                    help="Each vertex with degree > ceil(n/2) must be in a triangle")
parser.add_argument('--minDegreeVertex', type=int, help="Fist vertex must have the given degree")

args = parser.parse_args()

print("c\targs:", args)

n = args.vertices

V = range(n)
lastVertex = n - 1


all_variables = []
all_variables += [('edge', (u, v)) for u, v in combinations(V, 2)]
all_variables_index = {}

_num_vars = 0
for v in all_variables:
    _num_vars += 1
    all_variables_index[v] = _num_vars


def var(L): return all_variables_index[L]
def var_edge(u, v): return var(('edge', (min(u, v), max(u, v))))

# print(all_variables_index)


class IDPool:
    def __init__(self, start_from=0) -> None:
        self.start_from = start_from
    start_from = 1

    def id(self):
        x = self.start_from
        self.start_from += 1
        return x


vpool = IDPool(start_from=_num_vars+1)  # for auxiliary pysat variables


constraints = []

# ----------------------------- start encoding ----------------------------------------------------

falseLiteral = vpool.id()
trueLiteral = -falseLiteral
constraints.append([-falseLiteral])

commonNeighbor = {(i, j, k): vpool.id()
                  for i, j in combinations(V, 2) for k in set(V)-{i, j}}

for i, j in combinations(V, 2):
    for k in set(V)-{i, j}:
        L = (i, j, k)
        constraints.append([-commonNeighbor[L], +var_edge(i, k)])
        constraints.append([-commonNeighbor[L], +var_edge(j, k)])
        constraints.append(
            [+commonNeighbor[L], -var_edge(i, k), -var_edge(j, k)])

noCommonNeighbor = {(i, j): vpool.id() for i, j in combinations(V, 2)}
for i, j in combinations(V, 2):
    for k in set(V)-{i, j}:
        # if the have a common neighbor, noCommonNeighbor is false
        constraints.append(
            [-commonNeighbor[(i, j, k)], -noCommonNeighbor[(i, j)]])
    
    if args.triangular:
        # if no common neighbor is false the must have a common neighbor
        constraints.append([+noCommonNeighbor[(i, j)]] + [+commonNeighbor[(i, j, k)] for k in set(V)-{i, j}])


for i, j in combinations(V, 2):
    constraints.append([+var_edge(i, j)] + [+commonNeighbor[(i, j, k)]
                       for k in set(V)-{i, j}])  # adjacent or common neighbor

for i, j in combinations(V, 2):
    # ensure that critical i.e. if edge ij is present removing will lead to diamter > 2
    clause = [-var_edge(i, j), +noCommonNeighbor[(i, j)]]
    for k in set(V)-{i, j}:
        for v1, v2 in [(i, j), (j, i)]:
            # v2 and k have diameter > after removing ij
            # v1 adjacent to k and v1 is the only common neighbor from v2 and k. And k not adjacent to v2
            diameterIncreasing = vpool.id()
            constraints.append([+var_edge(v1, k), -diameterIncreasing])
            constraints.append([-var_edge(v2, k), -diameterIncreasing])
            for l in set(V)-{i, j, k}:
                constraints.append(
                    [-commonNeighbor[(min(v2, k), max(v2, k), l)], -diameterIncreasing])
            clause.append(diameterIncreasing)
    constraints.append(clause)

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


m = math.floor(n**2 / 4)

'''
from pysat.card import *

cnf =  CardEnc.atleast(lits=[var_edge(i,j) for i,j in combinations(V,2)],bound=m, top_id=vpool.id(), encoding=EncType.totalizer)

for c in cnf.clauses:
	constraints.append(c)
# vpool.start_from = max(vpool.start_from, map(lambda c: max(map(abs,c)) + 1 , cnf.clauses)) '''


# seqCounter([var_edge(i,j) for i,j in combinations(V,2)], m, vpool, atLeast = m)

if args.partition:
    pos = 0
    for i in range(len(args.partition)):
        posNew = pos + args.partition[i]

        # Each vertex has exactly degree  args.degrees[i]
        for v in range(pos, posNew):
            degree = args.degrees[i]
            counterImplementations.counterFunction([var_edge(v, u) for u in V if u != v], countUpto=degree,
                                                   vPool=vpool, clauses=constraints, atLeast=degree, atMost=degree, type=DEFAULT_COUNTER)
        pos = posNew
# else:
#	counterVariables = counterImplementations.counterFunction([var_edge(i,j) for i,j in combinations(V,2)], countUpto=m, vPool=vpool, clauses=constraints, atLeast=m, type=DEFAULT_COUNTER)

if args.different:
    for i, j in permutations(V, 2):
        # There must be a vertex adjecent to i which is not adjacent to j
        adjacentOnlyToI = []
        for k in V:
            if k == i or k == j:
                continue
            kIsAdjacentOnlyToI = vpool.id()
            constraints.append([+var_edge(i, k), -kIsAdjacentOnlyToI])
            constraints.append([-var_edge(j, k), -kIsAdjacentOnlyToI])
            adjacentOnlyToI.append(kIsAdjacentOnlyToI)
        constraints.append([+var_edge(i, j)] + adjacentOnlyToI)

d = []  # d[i] is variable indicating whether i has maximum degree
dAll = []  # all Degrees
if args.maxDegree:  # only if not predefined degrees
    for i in V:
        counterVariables = counterImplementations.counterFunction([var_edge(i, j) for j in V if j != i], countUpto=args.maxDegree, vPool=vpool,
                                                                  clauses=constraints, atMost=args.maxDegree, atLeast=args.minDegree, type=DEFAULT_COUNTER)
        # only true if a vertex has maximum degree
        d.append(counterVariables[args.maxDegree - 1])
        dAll.append(counterVariables)


if args.triangular:

    verticesWithHighDegree = []
    pos = 0
    for i in range(len(args.partition)):
        posNew = pos + args.partition[i]
        # Each vertex has exactly degree  args.degrees[i]
        degree = args.degrees[i]
        if degree > math.ceil(n / 2):
            verticesWithHighDegree = verticesWithHighDegree + \
                list( [ (v,degree) for v in range(pos, posNew) ])
        pos = posNew

    '''
    # indicate whether the three edges form a triangle
    var_triangle = {c: vpool.id() for c in combinations(V, 3) if c[0] in verticesWithHighDegree or c[1] in verticesWithHighDegree or c[2] in verticesWithHighDegree}
    for i, j, k in combinations(V, 3):
        if i in verticesWithHighDegree or j in verticesWithHighDegree or k in verticesWithHighDegree:
            # no edge implies no triangle
            constraints.append([+var_edge(i, j), -var_triangle[(i, j, k)]])
            constraints.append([+var_edge(i, k), -var_triangle[(i, j, k)]])
            constraints.append([+var_edge(j, k), -var_triangle[(i, j, k)]])

    # Each vertex with degree greater then ceil(n/2) most be in a triangle
    pos = 0
    for v in verticesWithHighDegree:
        potentialTriangles = [tuple(sorted([v, i, j]))
                              for i, j in combinations(V, 2) if i != v and j != v]
        constraints.append([+var_triangle[c] for c in potentialTriangles])

    '''

    for v,degree in verticesWithHighDegree:
        specialEdges = [vpool.id() if u != v else None for u in V]
        minDelete = degree - math.ceil(n / 2)
        counterImplementations.counterFunction([specialEdges[u] for u in V if u != v], minDelete, vpool, constraints, atLeast=minDelete, type=DEFAULT_COUNTER)

        for u in V:
            if u == v: continue
            constraints.append([+var_edge(v,u), -specialEdges[u]]) # no edge not special
            constraints.append([-noCommonNeighbor[(min(v,u),max(v,u))], -specialEdges[u]]) # if the have no common neighbor it's fine
            # if there is a vertex k not adjacent to v such that u is the only common neighbor then u is not special
            for k in V:
                if k in [u,v]: continue
                constraints.append([+var_edge(v,k), -commonNeighbor[(min(v,k),max(v,k), u)]] + [+commonNeighbor[(min(v,k),max(v,k), y)] for y in V if y not in [k,u,v]] + 
                    [-specialEdges[u]])


if args.minDegreeVertex:
    for i in range(1, len(V) - args.minDegreeVertex):
        constraints.append([-var_edge(0,i)])

if args.edges:
    counterVariables = counterImplementations.counterFunction([var_edge(i, j) for i,j in combinations(V,2)], countUpto=args.edges, vPool=vpool,
                                                                  clauses=constraints, atLeast=args.edges, type="sequential")
        

print("c\tTotal number of constraints:", len(constraints))
print("c\tTotal number of variables:", vpool.id())

print('c\tbegin of CNF')
for c in constraints:
    print(' '.join(str(x) for x in c))
print('c\tend of CNF')
