#!/usr/bin/python

from ast import arguments
from itertools import combinations, permutations
from operator import indexOf
from sys import *
# from pysat.card import *

import counterImplementations
DEFAULT_COUNTER = "sequential"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--partionsColoring', '-p', nargs="+", type=int, required=True, help="The sizes of the independent sets")
parser.add_argument('--exchangeAble', "-e", action="store_true", help="Test swapping pairs of vertices which can be swapped according to the independent sets")
parser.add_argument('--fixed2layer', "-f", action="store_true", help="If first vertex of a color is adjacent to only one vertex of another color, then this vertex must be adjacent to a vertex of fixed color")
parser.add_argument('--increaseFirstParition', "-i", action="store_true", help="Testing whether paritions can be increased based on connected components.")
parser.add_argument('--connectedComponents', "-c", action="store_true", help="Calulate connected components within encoding and try to increase partitions.")
parser.add_argument('--connectedComponentsColoring', "-cc", action="store_true", help="Calulate connected components within encoding and try to increase partitions.")
parser.add_argument('--noMaxDegree', action="store_true", help="No maximum degree given")
parser.add_argument('--girth', type=int, help="Minimal girth")
parser.add_argument('--noDistinctNeighborhood', action="store_true", help="Do not ensure distinct neighborhood")
parser.add_argument('--maxDegree', type=int, help="Maximal degree")
parser.add_argument('--mtf', action="store_true", help="MTF preserving maximum degree")
parser.add_argument('--numberOfEdges', type=int, help="Exact number of edges")

args = parser.parse_args()

print("c\targs:",args)
vargs = vars(args)
print("c\tactive args:",{x:vargs[x] for x in vargs if vargs[x] != None and vargs[x] != False})

n = sum(args.partionsColoring) + 1
nPartitions = len(args.partionsColoring)
N = nPartitions

# Create list for each partition
partitions = []
current = 0
for i in args.partionsColoring:
    partitions.append(list(range(current, current + i)))
    current += i
print("c\tpartitions:", partitions)

V = range(n)
lastVertex = n - 1


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

#-------------------------create encoding-----------------------

if args.numberOfEdges:
    counterImplementations.counterFunction([var_edge(i,j) for i,j in combinations(V,2)], args.numberOfEdges, 
        vpool, constraints, atMost=args.numberOfEdges, atLeast=args.numberOfEdges, type=DEFAULT_COUNTER)

commonNeighbor = { (i,j,k): vpool.id() for i,j in combinations(V,2) for k in set(V)-{i,j}}
neighborFromExactlyOneVertex = { (i,j,k): vpool.id() for i,j in combinations(V,2) for k in set(V)-{i,j}}

for i,j in combinations(V,2):
    for k in set(V)-{i,j}:
        L = (i,j,k) 
        constraints.append([-commonNeighbor[L], +var_edge(i,k)])
        constraints.append([-commonNeighbor[L], +var_edge(j,k)])
        constraints.append([+commonNeighbor[L], -var_edge(i,k), -var_edge(j,k)])

        constraints.append([+var_edge(i,k), +var_edge(j,k), -neighborFromExactlyOneVertex[L]])
        constraints.append([-var_edge(i,k), -var_edge(j,k), -neighborFromExactlyOneVertex[L]])


if not args.maxDegree:
	args.maxDegree = N + 1

#degree constraints
deg = [] # Variables is true if degree is N and false if N + 1 for each vertex
for i in V:
		d = counterImplementations.counterFunction([var_edge(i,j) for j in V if j != i], args.maxDegree,
			vpool, clauses=constraints, atLeast = N, atMost= None if args.noMaxDegree else args.maxDegree, type="totalizer") # at least N
		# print("c\t", len(d))
		deg.append(-d[(args.maxDegree - 1)]) # true if not maximal degree

if args.mtf:
	if not args.girth:
		for i,j in combinations(V,2):
			degreeLits = [] if args.noMaxDegree else [-deg[i], -deg[j]] # only if it doesn't violate the degree constraints, i.e., not already maximal degree
			constraints.append(degreeLits + [var_edge(i,j)] + [+commonNeighbor[(i,j,k)] for k in set(V)-{i,j}]) # Edge must be present if degree constraints are not violated and does not introduce a triangle
	else:
		#distance at most girth
		for i,j in combinations(V,2):
			degreeLits = [] if args.noMaxDegree else [-deg[i], -deg[j]]

			reached = [var_edge(i,v) if v != i else 0 for v in V] # base case
			
			for d in range(2, args.girth):
				reachedNew = [vpool.id() if v != i else 0 for v in V]  # if not reached within distance then variable false

				for k in V:
					if k == i:
						continue

					reachedViaL = { l:vpool.id() for l in V if l != i and l != k}


					clause = []
					for l in V:
						if l == i or l == k:
							continue
						constraints.append([+var_edge(k,l), -reachedViaL[l]]) # no edge implies not reached via l
						constraints.append([+reached[l], -reachedViaL[l]]) # l not reached in previous step 
					constraints.append([+reachedViaL[l] for l in V if l != i and l != k] + [+reached[k], -reachedNew[k]]) #not reached in previous step and not reached via any vertex implies not reached 
					reached = reachedNew	
				constraints.append(degreeLits + [var_edge(i,j), +reachedNew[j]]) # either edge or j was reached



if args.girth:
	for i in range(3,args.girth): # forbid cycles with length i
		for cycle in permutations(V,i):
			if cycle[0] != min(cycle) or cycle[1] < cycle[-1]:
				continue
			constraints.append([-var_edge(cycle[j],cycle[j + 1]) for j in range(i - 1)] + [-var_edge(cycle[0], cycle[i - 1])])

# distinct neighborhood (or adjacent)
if not args.noDistinctNeighborhood:
	for i,j in combinations(V,2):
		constraints.append([+var_edge(i,j)] + [+neighborFromExactlyOneVertex[(i,j,k)] for k in set(V)-{i,j}])

# triangular free
for i,j,k in combinations(V,3):
    constraints.append([-var_edge(i,j), -var_edge(j,k), -var_edge(i,k)])

# vertices in partition are not adjacent
for p in partitions:
	for i,j in combinations(p,2):
		constraints.append([-var_edge(i,j)])

# last vertex adjacent to first of each color
for p in partitions:
	constraints.append([+var_edge(min(p),lastVertex)])

# last vertex adjacent to first of each color
for p in partitions:
	for v in p:
		if v == min(p):
			continue
		constraints.append([-var_edge(v,lastVertex)])

# Each vertex most be adjacent to a vertex from a lower paritition
for p1,p2 in combinations(partitions,2):
	for v in p2:
		constraints.append([+var_edge(u,v) for u in p1])

# For first vertices also other direction
for p1,p2 in combinations(partitions,2):
	constraints.append([+var_edge(u,min(p1)) for u in p2])

# introducing fixed variables (adjacent to each other color)
fixedColor = [ vpool.id() for v in V if v != lastVertex]
for p in partitions:
	for v in p:
		for p2 in partitions:
			if p == p2: # same color
				continue
			constraints.append([+var_edge(v,u) for u in p2] + [-fixedColor[v]]) # if not adjacent to any of p2 then not fixed
			
# Each first vertex of each color must be adjacent to a vertex of fixed color
for p in partitions:
	f = min(p) # first vertex in partition
	for p2 in partitions:
		if p == p2: # same color
			continue

		clause = []
		for v in p2:
			adjacentAndFixed = vpool.id()
			constraints.append([-adjacentAndFixed, var_edge(f,v)])
			constraints.append([-adjacentAndFixed, fixedColor[v]])
			clause.append(adjacentAndFixed)
		constraints.append(clause)


# If same size then also in the other direction
for p1,p2 in combinations(partitions,2):
	if len(p1) == len(p2):
		for v in p1:
			constraints.append([+var_edge(u,v) for u in p2])

# if first vertex is adjacent to only one vertex of a color, then this vertex must be adjacent to a vertex of fixed color
if args.fixed2layer:
	for p in partitions:
		f = min(p)
		for p2 in partitions:
			if p == p2:
				continue
			for v in p2:
				for p3 in partitions:
					if p == p3 or p2 == p3:
						continue
					clause = [-var_edge(f,v)] + [+var_edge(f,u) for u in p2 if u != v] # not adjacent or not only neighbor

					for u in p3:
						adjacentAndFixed = vpool.id()
						constraints.append([-adjacentAndFixed, var_edge(v,u)])
						constraints.append([-adjacentAndFixed, fixedColor[u]])
						clause.append(adjacentAndFixed)
					constraints.append(clause)

# forbid two swaps for vertices of same color except first (Codish like)
for p in partitions:
	for v,u in combinations(sorted(set(p)-{min(p)}), 2):
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


# try swapping variables vertices which are exchangeAble
if args.exchangeAble:
	for p1, p2 in combinations(partitions,2):
		for v in set(p1)-{min(p1)}:
			for u in set(p2)-{min(p2)}:
				exchangeAble = vpool.id()
				constraints.append([+var_edge(w,u) for w in p1 if w != v] + [+var_edge(w,v) for w in p2 if w != u] + [exchangeAble]) # exchangeable if not adjacent to any other

				allPreviousEqual = vpool.id()
				constraints.append([+allPreviousEqual])
				for w in range(n - 1):
					if w == v or w == u:
						continue
					# add -exchangeAble to all clauses so satisfied if not exchangeAble
					constraints.append([-exchangeAble, -allPreviousEqual, -var_edge(v,w), +var_edge(u,w)]) # if all previous are equal then no edge to first vertex or an edge to second
					allPreviousEqualNew = vpool.id()
					constraints.append([-exchangeAble, -allPreviousEqual, -var_edge(v,w), +allPreviousEqualNew])
					constraints.append([-exchangeAble, -allPreviousEqual, +var_edge(u,w), +allPreviousEqualNew])
					allPreviousEqual = allPreviousEqualNew

# for each connected componenet of the induced subgraph of two colors, the number of vertices with the smaller color is not allowed to be smaller
# used exponential encoding
# also each first vertex of a color must be in the same connected component
if args.increaseFirstParition:
	for p1, p2 in combinations(partitions,2):
		f1 = min(p1)
		f2 = min(p2)
		for s1 in range(1,len(p1) + 1):
			for s2 in range(1,len(p2) + 1):
				for c1 in combinations(p1, s1):
					for c2 in combinations(p2, s2):
						if s1 < s2 or f1 in c1 and f2 not in c2:
							# first component smaller or 
							# first in componenet only for the lower color
							constraints.append([+var_edge(v,u) for v in c1 for u in set(p2)-set(c2)] + 
								[+var_edge(v,u) for v in c2 for u in set(p1)-set(c1)])
							

#version with connected components
if args.connectedComponents:
	for p1, p2 in combinations(partitions,2):
		for v in p1:
			# connected component containing v
			sizeOfSubgraph = len(p1) + len(p2)
			reached = {(x,k) : vpool.id() for x in p1 + p2 for k in range(sizeOfSubgraph)}

			for x in p2:
				reached[(x,0)] = var_edge(x,v)

			for x in p1 + p2:
				# starting case
				if x == v:
					constraints.append([+reached[(x,0)]])
				else:
					if x in p1: # others already the edges
						constraints.append([-reached[(x,0)]])
			
				#inductively
				for k in range(1,sizeOfSubgraph):
					constraints.append([-reached[(x,k - 1)], reached[(x,k)]])
					for y in p1 if x in p2 else p2: #choose other partition to avoid some variables (same partition can not be adjacent)
						constraints.append([-reached[(y,k - 1)], -var_edge(x,y), +reached[(x,k)]])
					# cases when not reached
					clause = [reached[(x,k - 1)]]
					for y in p1 if x in p2 else p2: #choose other partition to avoid some variables (same partition can not be adjacent)
						notReachedOrNotAdjacent = vpool.id()
						constraints.append([+var_edge(x,y), +notReachedOrNotAdjacent])
						constraints.append([+reached[(y,k - 1)], +notReachedOrNotAdjacent])
						constraints.append([-var_edge(x,y), -reached[(y,k - 1)], -notReachedOrNotAdjacent])
						clause.append(-notReachedOrNotAdjacent)
					clause.append(-reached[(x,k)])
					constraints.append(clause)

			# reached[(x,sizeOfSubgraph - 1)] contains whether x is part of the connected compoenent
			lastStep = 	sizeOfSubgraph - 1

			# add clause that both first vertices of each color must be in the same partition
			if v == min(p1):
				constraints.append([+reached[(min(p2),lastStep)]])

			# Compare counts after swapping
			count1 = counterImplementations.counterFunction([reached[(x,lastStep)] for x in p1], len(p1), vpool, clauses=constraints, type=DEFAULT_COUNTER) # at least N

			list2 = [reached[(x,lastStep)] for x in p2] # add addable variables to list
			i  = indexOf(partitions, p1)
			for p3 in partitions[i + 1:]: # only consider later partitions then p1
				if p3 == p2:
					continue
				for x in p3:
					# x can be added to p1 swapping connected component
					addAble = vpool.id() 
					list2.append(addAble)
					addAbleClause = []
					for y in p1:
						notAdjacacentOrInComponent = vpool.id()
						constraints.append([+var_edge(x,y), notAdjacacentOrInComponent])
						constraints.append([-reached[(y,lastStep)], notAdjacacentOrInComponent])
						addAbleClause.append(-notAdjacacentOrInComponent)
					for y in p2:
						notAdjacacentOrNotInComponent = vpool.id()
						constraints.append([+var_edge(x,y), notAdjacacentOrNotInComponent])
						constraints.append([+reached[(y,lastStep)], notAdjacacentOrNotInComponent])
						addAbleClause.append(-notAdjacacentOrNotInComponent)
					addAbleClause.append(addAble) # if all previous do not hold than vertex can be added

					constraints.append(addAbleClause)
			count2 = counterImplementations.counterFunction(list2, len(p1) + 1, vpool, clauses=constraints, type=DEFAULT_COUNTER)
			constraints.append([-count2[len(p1)]]) # not allowed to be more

			for i in range(len(p1)):
				constraints.append([+count1[i], -count2[i]]) # count2 <= count1 

			if not args.connectedComponentsColoring:
				continue

			# Check if swapping connected component leads to an easy N coloring	
			fixedColorV = [vpool.id() for _ in range(n - 1)] # false if not a fixed color after swapping
			for p3 in partitions:
				for v in p3:
					if p1 == p3 or p2 == p3:
						for p4 in partitions:
							if p3 == p4: # same color
								continue
							constraints.append([+var_edge(v,u) for u in p4] + [-fixedColorV[v]]) # if not adjacent to any of p2 then not fixed
					else:
						for p4 in partitions:
							if p3 == p4: # same color
								continue
							if p4 in [p1, p2]:
								clause = [-fixedColorV[v]]
								for u in p1 if p4 == p1 else p2:
									adjacentAndUnchangedColor = vpool.id()
									constraints.append([+var_edge(v,u), -adjacentAndUnchangedColor])
									constraints.append([-reached[(u,lastStep)], -adjacentAndUnchangedColor])
									clause.append(adjacentAndUnchangedColor)
								for u in p2 if p4 == p1 else p1:
									adjacentAndChangedColor = vpool.id()
									constraints.append([+var_edge(v,u), -adjacentAndChangedColor])
									constraints.append([+reached[(u,lastStep)], -adjacentAndChangedColor])
									clause.append(adjacentAndChangedColor)
								constraints.append(clause)
							else: # same as not swapped
								constraints.append([+var_edge(v,u) for u in p4] + [-fixedColorV[v]])

			# begin checking coloring:  each f_k must be still adjacent to a fixed vertex of each other color
			for p3 in partitions:
				f = min(p3) # first vertex in partition
				if p3 in [p1,p2]:
					for p4 in partitions:
						if p3 == p4: # same color
							continue

						clause = []
						for v in p4:
							adjacentAndFixed = vpool.id()
							constraints.append([-adjacentAndFixed, var_edge(f,v)])
							constraints.append([-adjacentAndFixed, fixedColorV[v]])
							clause.append(adjacentAndFixed)
						constraints.append(clause)
				else:
					for p4 in partitions:
						if p4 == p3:
							continue
						if p4 in [p1,p2]:
							clause = []
							for v in p1 if p4 == p1 else p2 :
								adjacentAndFixed = vpool.id()
								constraints.append([-adjacentAndFixed, var_edge(f,v)])
								constraints.append([-adjacentAndFixed, fixedColorV[v]])
								constraints.append([-adjacentAndFixed, -reached[(v,lastStep)]])
								clause.append(adjacentAndFixed)
							for v in p2 if p4 == p1 else p1 :
								adjacentAndFixed = vpool.id()
								constraints.append([-adjacentAndFixed, var_edge(f,v)])
								constraints.append([-adjacentAndFixed, fixedColorV[v]])
								constraints.append([-adjacentAndFixed, reached[(v,lastStep)]])
								clause.append(adjacentAndFixed)
							constraints.append(clause)
						else:
							for v in p4:
								adjacentAndFixed = vpool.id()
								constraints.append([-adjacentAndFixed, var_edge(f,v)])
								constraints.append([-adjacentAndFixed, fixedColorV[v]])
								clause.append(adjacentAndFixed)
							constraints.append(clause)

print("c\tTotal number of constraints:", len(constraints))
print("c\tTotal number of variables:", vpool.id())


print('c\tbegin of CNF')
for c in constraints:
    print (' '.join(str(x) for x in c))
print('c\tend of CNF')

