from itertools import combinations, permutations
from math import ceil
from sys import *
from counterImplementations import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("n",type=int,help="number of vertices")

# maybe Smith's conjecture: k>=7-connected => any 2 longest circuits have at least k common vertices

parser.add_argument("--transitive", "-t", action='store_true', help="ensure transitivity of directed edges")
parser.add_argument("--acyclic", action='store_true', help="ensure that no directed cycle")
parser.add_argument("--secondNeighborhoodConjecture", "-s2", action='store_true', help="test second neihgborhood conjecture for given number of vertices")
parser.add_argument("--CaccettaHaggkvistConjecture", "-chc", type=int, help="test Caccetta-Haggkvist where k is the outdegree and  ceil(n/k) the smallest cycle")

parser.add_argument("--DEBUG","-D",type=int,default=1,help="debug level")
args = parser.parse_args()

if args.DEBUG:
	print("c\targs:",args)
vargs = vars(args)
print("c\tactive args:",{x:vargs[x] for x in vargs if vargs[x] != None and vargs[x] != False})

n = args.n
V = range(n)

DEBUG = args.DEBUG

all_variables = []
all_variables += [('edge',(u,v)) for u,v in permutations(V,2)]

all_variables_index = {}

_num_vars = 0
for v in all_variables:
    _num_vars += 1
    all_variables_index[v] = _num_vars


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
def var_edge(u,v): return var(('edge',(u,v)))

constraints = []

if DEBUG: print("c\tTotal number of constraints:"+str(len(constraints)))

# return the counter variables for the last variable: l[i] is true if there are at least i + 1 elements true
def seqCounter(variables, countUpto, vPool):
	n = len(variables)
	counterVariables = [[vPool.id() for _ in range(countUpto)] for _ in range(n)] #Create new variables
	# print("c\t" + str(counterVariables))
	# first element
	counterVariables[0][0] = variables[0]
	for i in range(1,countUpto):
		constraints.append([-counterVariables[0][i]]) # at most one at the beginning

	# adapt counter for each step
	for i in range(n - 1):
		constraints.append([-variables[i + 1], counterVariables[i + 1][0]]) # if there is an element than there is at least on element
		for j in range(countUpto):
			constraints.append([-counterVariables[i][j], counterVariables[i + 1][j]]);               # at least as many
			constraints.append([counterVariables[i][j], variables[i + 1], -counterVariables[i + 1][j]]); # the same if element is not present

			if j < countUpto - 1:
				constraints.append([-counterVariables[i][j], -variables[i + 1], counterVariables[i + 1][j + 1]]); # one more element
				constraints.append([counterVariables[i][j], -counterVariables[i + 1][j + 1]]);                # at most one more

	return [counterVariables[n - 1][j] for j in range(countUpto)]


if args.transitive:
	for u,v,w in permutations(V,3):
		constraints.append([-var_edge(u,v),-var_edge(v,w), +var_edge(u,w)])

if args.secondNeighborhoodConjecture:
	# oriented graph (digraph without edges in both directions)
	for i,j in combinations(V,2):
		constraints.append([-var_edge(i,j), -var_edge(j,i)])

	for i in V:
		outgoingEdges = [var_edge(i,j) for j in V if i != j]
		# outdegree at least 7 by some previous results
		neighborCount = seqCounter(outgoingEdges, max(7, n // 2 + 1), vpool)
		constraints.append([+neighborCount[7 - 1]]) # at least 7 must be true

		secondNeighbors = []
		for j in V :
			if i == j: continue
			v = vpool.id() # true if j is a second neighbor
			secondNeighbors.append(v)
			for k in set(V) - {i,j}:
				constraints.append([+var_edge(i,j), -var_edge(i,k), -var_edge(k,j), +v]) # if i not adjacent to j and ik and kj edges than j is a second neighbor
		
		secondNeighborsCount = seqCounter(secondNeighbors, max(7, n // 2 + 1), vpool)
		# not allowed to have more second neighbors
		for x in range(1,max(7, n // 2 + 1)):
			constraints.append([+neighborCount[x], -secondNeighborsCount[x - 1]]) # at least one less



if args.CaccettaHaggkvistConjecture:
	outdegree = args.CaccettaHaggkvistConjecture
	k = ceil(n / outdegree)
	print("c\tno cycle with length at most",k, "allowed")

	# discard very small cycles
	for i,j in combinations(V,2):
		constraints.append([-var_edge(i,j), -var_edge(j,i)])

	for i in V:
		outgoingEdges = [var_edge(i,j) for j in V if i != j]
		# outdegree at exactly the specified value
		neighborCount = seqCounter(outgoingEdges, outdegree + 1, vpool)
		constraints.append([+neighborCount[outdegree - 1]])
		constraints.append([-neighborCount[outdegree]])
	if True:
		# girth at least k + 1
		for i,j in permutations(V,2): # in both directions
			# starting from i check if v reached in s steps
			reached = {}
			for v in set(V)-{i,j}:
				reached[v] = var_edge(i,v)

			for s in range(2,k - 1): #TODO check last index (should be okay for example for k = 3 we do not want to enter the loop)
				reachedNew = {}
				for v in set(V)-{i,j}:
					reachedNew[v] = vpool.id()
					constraints.append([-reached[v], reachedNew[v]])
					for u in set(V)-{i,j,v}:
						constraints.append([-reached[u], -var_edge(u,v), reachedNew[v]])
				reached = reachedNew
			#  no reached vertex adjacent to j or ij not and egde
			for v in set(V)-{i,j}:
				constraints.append([-reached[v], -var_edge(v,j), -var_edge(j,i)])

if args.acyclic:
	transitiveClosure = [[vpool.id() for _ in V] for _ in V] # [i][j] true if j can be reached from j
	for v1, v2 in permutations(V,2):
		constraints.append([-var_edge(v1,v2), transitiveClosure[v1][v2]])
	for v1, v2, v3 in permutations(V,3):
		constraints.append([-transitiveClosure[v1][v2],-transitiveClosure[v2][v3], transitiveClosure[v1][v3]])
	for v1, v2 in combinations(V,2):
		constraints.append([-transitiveClosure[v1][v2], -transitiveClosure[v2][v1]])



print('c\tbegin of CNF')
for c in constraints:
	print (' '.join(str(x) for x in c))
print('c\tend of CNF')



