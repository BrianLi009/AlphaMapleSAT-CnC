#!/usr/bin/python

from itertools import combinations, permutations
from counterImplementations import *
from sys import *
# from pysat.card import *


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--vertices', '-n', type=int, required=True)
parser.add_argument('--edges', '-m', type=int, required=True)
parser.add_argument('--minDegree', "-d", type=int, required=True)
parser.add_argument('--maxDegree', "-D", type=int, required=True)
parser.add_argument('--embeddedStar', action="store_true", help="Version with embedded star (strictly speaking tree with depth 2)")
parser.add_argument('--embeddedStar2', action="store_true", help="Version with embedded star with depth 3 instead of 2")
parser.add_argument('--codish', "-c", action="store_true", help="apply codish like symmetry breaking")
parser.add_argument('--girth', type=int, default=5, help="Minimum girth of the graph")
parser.add_argument('--compactGirthConstraints', action="store_true", help="Don't enumerate all cycles but use additional variables")
parser.add_argument('--atLeastOneMaximumDegree', action="store_true", help="At least one vertex has maximum degree")
parser.add_argument('--initialVertexOrderings', type=str, help="The file where the vertex orderings should be stored")

args = parser.parse_args()

print("c\targs:",args)

n = args.vertices

V = range(n)
lastVertex = n - 1


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

if not args.compactGirthConstraints:
	for i in range(3, args.girth): # forbid cycles with length i
		for cycle in permutations(V,i):
			if cycle[0] != min(cycle) or cycle[1] < cycle[-1]:
				continue
			constraints.append([-var_edge(cycle[j],cycle[j + 1]) for j in range(i - 1)] + [-var_edge(cycle[0], cycle[i - 1])])
else:
	# check distance of i,j without edge i,j.
	for i,j in combinations(V,2):
		reached = [var_edge(i,k) if k not in [i,j] and k > i else None for k in V]

		for runde in range(args.girth - 4): # if girth is 4 than no triangles so not in the loop
			reachedNew = [vpool.id() for _ in V]
			for k in V:
				if k in [i,j] or k < i: continue
				constraints.append([-reached[k], +reachedNew[k]]) # already reached

				# check if reached over l
				for l in V:
					if l in [i,j,k] or l < i: continue
					constraints.append([-var_edge(k,l), -reached[l], +reachedNew[k]]) # l reached in previous step and edge implies reached
			reached = reachedNew

		for k in V:
			if k in [i,j] or k < i: continue
			# not reached, not adjacent to j, or edge not present
			constraints.append([-var_edge(i,j), -var_edge(j,k), -reached[k]])

counterFunction([var_edge(i,j) for i,j in combinations(V,2)], args.edges, vpool, constraints, atLeast = args.edges, type="sequential")
maxDegreeVariables = []
for i in V:
	degree = counterFunction([var_edge(i,j) for j in V if j != i],  args.maxDegree, 
			  vpool, constraints, atLeast=args.minDegree, atMost=args.maxDegree, type="sequential")
	maxDegreeVariables.append(degree[args.maxDegree - 1])

# at least one with maximum degree
if args.atLeastOneMaximumDegree:
	constraints.append(maxDegreeVariables)

# fix embedded star
if args.embeddedStar:
	for i in range(lastVertex - args.maxDegree, lastVertex):
		constraints.append([var_edge(i,lastVertex)])

	remainingDegree = args.minDegree - 1
	start = lastVertex - args.maxDegree - remainingDegree * args.maxDegree
	for i in range(lastVertex - args.maxDegree, lastVertex):
		lower = start + (i - (lastVertex - args.maxDegree)) * remainingDegree
		# print("c\t", lower, lower + remainingDegree)
		for j in range(lower, lower + remainingDegree):
			constraints.append([var_edge(i,j)])
			if args.codish:
				for v,u in combinations(range(lower, lower + remainingDegree), 2):
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

	if args.initialVertexOrderings:
		initialVertexOrderingsFile = open(args.initialVertexOrderings, "w")

		# print initial partition
		print(" ".join(map(str,[n - (remainingDegree *  args.maxDegree + args.maxDegree + 1)] + [remainingDegree]* args.maxDegree + [1] * args.maxDegree + [1])) , 
		 	file=initialVertexOrderingsFile)
		
		# print different vertex orderings to start with
		print(" ".join(map(str, V)), file=initialVertexOrderingsFile) # standard ordering 

		# swap every subtree starting with distance 1
		for i,j in combinations(range(args.maxDegree), 2):
			v1 = lastVertex - args.maxDegree  + i
			v2 = lastVertex - args.maxDegree + j
			vertexOrdering = []
			for v in V:
				if v == v1:
					vertexOrdering.append(v2)
				elif v == v2:
					vertexOrdering.append(v1)
				elif v in range(start + remainingDegree * i, start + remainingDegree * (i + 1)):
					vertexOrdering.append(v + (j - i) * remainingDegree)
				elif v in range(start + remainingDegree * j, start + remainingDegree * (j + 1)):
					vertexOrdering.append(v + -(j - i) * remainingDegree)
				else:
					vertexOrdering.append(v)
			print(" ".join(map(str, vertexOrdering)), file=initialVertexOrderingsFile) # standard ordering 



if args.codish and not args.embeddedStar and not args.embeddedStar2:
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

print("c\tTotal number of constraints:", len(constraints))
print("c\tTotal number of variables:", vpool.id())

print('c\tbegin of CNF')
for c in constraints:
    print (' '.join(str(x) for x in c))
print('c\tend of CNF')






# not used anymore but maybe relevant for the future:
'''
if args.embeddedStar2:
	
	remainingDegree = args.minDegree - 1
	Delta = args.maxDegree
	layers = [[lastVertex]]
	for i in range(3):
		firstPrev = layers[-1][0] # the first from previous layer
		first = firstPrev - Delta * (remainingDegree ** i)
		layers.append(list(range(first, firstPrev)))
	# print(layer)
	def vertex2predecessor(v):
		l = None
		for i in range(4):
			if v in layers[i]: l = i
		if l == 1:
			return lastVertex
		prev = (v - layers[l][0]) // remainingDegree
		return prev + layers[l - 1][0]

	childs = [[] for _ in V] # all
	parents = [None for _ in V]
	for v in range(layers[-1][0], lastVertex):
		parents[v] = vertex2predecessor(v)
		# print(v, parents[v])
		childs[parents[v]].append(v)

	firstVertexEmbeddedStar = layers[-1][0]
	for v in range(firstVertexEmbeddedStar, lastVertex):
		constraints.append([+var_edge(v, parents[v])])

	# print(childs, parents)

	if args.codish:
		for v,u in combinations(layers[-1], 2):
			if vertex2predecessor(v) != vertex2predecessor(u): continue
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

	if args.initialVertexOrderings:
		initialVertexOrderingsFile = open(args.initialVertexOrderings, "w")

		#first print initial partition
		initialPartition = [firstVertexEmbeddedStar] + [remainingDegree] * remainingDegree * args.maxDegree + [1] * (n - layers[4 - 2][0])
		assert(sum(initialPartition) == n)
		print( " ".join(map(str, initialPartition)), file=initialVertexOrderingsFile)

		# f is some function; DFS 
		def transveral(predecessors, currentVertex, f):
			for v in predecessors[currentVertex]:
				transveral(predecessors, v, f)
			f(currentVertex)


		def getVertexOrdering(tree):
			layersReorderd = [[] for _ in range(4)]
			def f(v):
				l = None
				for i in range(4):
					if v in layers[i]: l = i
				layersReorderd[l].append(v)
			transveral(tree, lastVertex, f)
			layersReorderd =  layersReorderd[::-1] # reverse layers
			for v in range(0, layers[-1][0]):
				print(v,end=" ", file=initialVertexOrderingsFile)
			for l in layersReorderd:
				for v in l:
					print(v,end=" ", file=initialVertexOrderingsFile)
			print(file=initialVertexOrderingsFile)

		getVertexOrdering(childs)
		for i,j in list(combinations(range(args.maxDegree),2)): # choose two to swap
			childs[lastVertex][i], childs[lastVertex][j] =  childs[lastVertex][j], childs[lastVertex][i] # swap this two and hence the whole subtree
			if remainingDegree > 1:
				for x in [i,j]:
					v = childs[lastVertex][x]
					for i2,j2 in combinations(range(remainingDegree), 2):
						childs[v][i2], childs[v][j2] =  childs[v][j2], childs[v][i2]
						getVertexOrdering(childs)
						childs[v][i2], childs[v][j2] =  childs[v][j2], childs[v][i2] # Reverse
			else:
				getVertexOrdering(childs)
			childs[lastVertex][i], childs[lastVertex][j] =  childs[lastVertex][j], childs[lastVertex][i] # swap this two and hence the whole subtree

		# same as before but only vertices from third layer swapped
		for x in range(args.maxDegree): # choose two to swap
				v = childs[lastVertex][x]
				for i2,j2 in combinations(range(remainingDegree), 2):
					childs[v][i2], childs[v][j2] =  childs[v][j2], childs[v][i2]
					getVertexOrdering(childs)
					childs[v][i2], childs[v][j2] =  childs[v][j2], childs[v][i2] # Reverse '''