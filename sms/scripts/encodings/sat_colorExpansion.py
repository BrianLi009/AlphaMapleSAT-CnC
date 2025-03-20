#!/usr/bin/python

from itertools import combinations, permutations
import itertools
from operator import indexOf
from sys import *
from time import perf_counter
from sat_indset_heuristic import *

from numpy import product
# from pysat.card import *

import counterImplementations
import ast
DEFAULT_COUNTER = "sequential"

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--config1', action="store_true", help="Basic expansion with reduceable and special.")
parser.add_argument('--config2', action="store_true", help="Expansion from a subgraph of an mtf_d with reduceable and special.")

# -----------------------------basic------------------------------------------
parser.add_argument('--vertices', '-n', type=int, required=True, help="Number of vertices")
parser.add_argument('--verticesSubgraph', '-s', type=int, required=True, help="Size of the subgraph")
parser.add_argument('--verticesIndependentSet', '-i', type=int, required=True, help="Size of the fixed maximal independent set")
parser.add_argument('--subgraphIsNotInduced', action="store_true", help="Consider SUBgraph of the given graph for the first VS vertices")
parser.add_argument('--subgraphIsNotInduced2', action="store_true", help="Consider SUPERgraph of the given graph for the first VS vertices")
parser.add_argument('--minDegree', '-d', type=int, required=True)
parser.add_argument('--maxDegree', '-D', type=int, required=True)
parser.add_argument('--graphFile', type=str, help="File containing graphs as adjacency list in python format ")
parser.add_argument('--graphFile2', type=str, help="File containing graphs as adjacency list in python format ")
parser.add_argument('--nGraph', type=int, help="Which line should be selected in the graph file starting with line 0")
parser.add_argument('--critical', type=int, help="Ensure that resulting graphs are critical")
parser.add_argument('--criticalAndReduceAble', type=int, help="Ensure that resulting graphs are critical and check if the colors from V-v does not result in a degree reducer")
parser.add_argument('--subgraphMinDegree', '-ds', type=int, help="The minimum degree of the subgraph")
parser.add_argument("--basicCritical", action="store_true", help="discard some cases which are obviously not critical, i.e., N(v) subset N(u) for two distinct vertices")
parser.add_argument('--basicCriticalSubgraph', action="store_true", help="Same as above but only for VS")

# -------------------------mtf related; by default mtf_D--------------------------------
parser.add_argument('--realMTF', action="store_true")
parser.add_argument('--mtfSpecial', action="store_true", help="All vertex exept one containing a minimum degree vertex must have a common neighbor or be adjacent; also maximum degree vertices must have radius 2")
parser.add_argument('--mtfSpecial2', action="store_true", help="mark special vertices which don't have to be mtf")
parser.add_argument('--noMtfD', action="store_true", help="don't ensure any version of mtf")

# -------------------------predefined partition of vertices and there degrees
parser.add_argument('--partition', '-p', type=int, nargs="+", help="paritioning of vertices")
parser.add_argument('--degrees', type=int, nargs="+", help="degree of vertices of the partition")

#------------------------------------------- checking if we" can remove more vertices-------------------------------------
parser.add_argument('--removeAble', action="store_true", help="Check if more then |VR| + |VI| vertices can be removed using unique mtf expansion; assumes that subgraphMinDegree is set")
parser.add_argument('--special', type=int, help="Integer gives the minimal sizes considered for the indpendent sets. In combination with MDE: Try to reduce graph starting from independent set of subgraph and VR")
parser.add_argument('--special_2', type=int, help="Same as before but try to increase independent set")
parser.add_argument('--special2', type=int, help="Integer gives the minimal sizes considered for the indpendent sets. Starting from independent sets of subgraph, extend them with special encoding and check removeable once")
parser.add_argument('--special4', type=int, help="Add first possible to independent set and then delete small degree vertices")


#--------------------------------refine edges between VR and VS---------------------------------------
parser.add_argument('--MDE', action="store_true", help="Maximum degree expansion, i.e., remaining part only adjacent to vertices in independent set and only one such vertex")
parser.add_argument('--EMDE', action="store_true", help="Vertex in VR at most one neighbor in VS")
parser.add_argument('--EMDE2', action="store_true", help="Vertex in VI + VR must be a degree reducer")
parser.add_argument('--EMDE3', action="store_true", help="Vertex in VR must be a degree reducer and VI is not given")

#---------------------------other-------------------------------------
parser.add_argument('--increaseSpecial', action="store_true")
parser.add_argument('--increaseSpecial2', action="store_true")
parser.add_argument('--largestIndependentSet', action="store_true", help="Ensure that we select the largest possible independent set")
parser.add_argument('--explicitColorClauses', action="store_true")
parser.add_argument('--maxIndependentSet', type=int, help="largest size of an independent set")
parser.add_argument('--numberOfEdges', type=int, help="Exact number of edges")
parser.add_argument("--numberOfMaxdegreeVertices", type=int, help="Upperbound on the number of vertices with maximum degree")
parser.add_argument("--minDegreeVertex", action="store_true", help="The first vertex has minium Degree")
parser.add_argument("--edgeCritical", type=int, help="only edge critical graphs")
parser.add_argument("--equalityVersion", action="store_true")
parser.add_argument("--maxDegreeSubgraph", type=int, help="ensure that at least one vertex of the subgraph has maximum degree")
parser.add_argument('--adjacentOrAtmost3commonNeighbors', action="store_true")
parser.add_argument('--adjacentOrOtherMoreGeneral', type=int, help="integer gives the minimum degree after the reduction")
parser.add_argument('--special3', type=int, help="Integer gives the minimal sizes considered for the indpendent sets. Starting from independent sets of subgraph just case distinction of expanding")
parser.add_argument('--degreeToRemainingPart', action="store_true")
parser.add_argument('--partialColorExpansion', action="store_true")
parser.add_argument('--partialColorExpansionNColors', type=int, default=200)
parser.add_argument('--partialColorExpansionImbalanced', type=bool, default=True)

parser.add_argument('--exactMaxDegree', action="store_true", help="There is a vertex with maximum degree")
parser.add_argument('--nonDecreasingIndpendentSetInReducer', action="store_true")
parser.add_argument('--fixed', type=int)
parser.add_argument('--breakAutomorphismsSubgraph', type=str, help="Check if we can explote automorphisms on VS; initial vertex ordering are written to the given file")
parser.add_argument('--groetsch', action="store_true", help="At least on Grötzsch graph must be kept as subgraph of VS")


t_begin = perf_counter()

# Some known unsatisfiable case
def exitWithUnsatEncoding():
    print(1)
    print(-1)
    exit()

args = parser.parse_args()


print("c\targs:",args)
vargs = vars(args)
print("c\tactive args:",{x:vargs[x] for x in vargs if vargs[x] != None and vargs[x] != False})

verticesRest = args.vertices - args.verticesSubgraph - args.verticesIndependentSet
n = args.vertices

V = range(n)
VS = list(range(args.verticesSubgraph)) # vertices of subgraph
VI = list(range(args.verticesSubgraph, args.verticesSubgraph + args.verticesIndependentSet)) # vertices of independent set
VR = list(range(args.verticesSubgraph + args.verticesIndependentSet, n)) # vertices of rest

DEFAULT_CHROMATIC_NUMBER = 5
if args.config1 or args.config2:
    args.basicCritical = True
    args.partialColorExpansion = True
    args.removeAble = True
    args.exactMaxDegree = True
    
    args.subgraphMinDegree = DEFAULT_CHROMATIC_NUMBER - 2

if args.config1:
    if len(VI) == args.maxDegree and len(VR) == 1:
        args.MDE = True
        args.special = len(VR) + len(VI) - 2
    else:
        args.special4 = len(VR) + len(VI) - 2

if args.config2:
    args.subgraphIsNotInduced = True
    if len(VI) == args.maxDegree and len(VR) == 1:
        args.MDE = True
        args.special = len(VR) + len(VI) - 2
    else:
        args.special4 = len(VR) + len(VI) - 2

    
# check some basic configurations
if len(VI) == args.maxDegree and len(VR) == 1 and not args.MDE:
    raise ValueError("MDE should be used if possible")
if (len(VI) != args.maxDegree or len(VR) != 1) and args.MDE:
    raise ValueError("MDE is not applicable with this configuration " + str(len(VI)) + ", " + str(len(VR)))
if args.special and not args.MDE:
    raise ValueError("special only in combination with MDE")


print("c\t", V, VS, VI, VR)

edgeListSubgraph = []
maximalIndependentSets = []
if args.graphFile:
    with open(args.graphFile) as fp:
        for i, line in enumerate(fp):
            if i == args.nGraph:
                if ";" in line:
                    graph_data, indset_data = line.strip().split(";")
                    edgeListSubgraph = ast.literal_eval(graph_data)
                    maximalIndependentSets = ast.literal_eval(indset_data)
                else:
                    edgeListSubgraph = ast.literal_eval(line.strip())
                break

if args.graphFile2:
    with open(args.graphFile2) as fp:
        for i, line in enumerate(fp):
            if i == args.nGraph:
                if ";" in line:
                    graph_data, indset_data = line.strip().split(";")
                    edgeListSubgraph = ast.literal_eval(graph_data)
                    maximalIndependentSets = ast.literal_eval(indset_data)
                else:
                    edgeListSubgraph = ast.literal_eval(line.strip())
                break

if args.breakAutomorphismsSubgraph:
    from sage.all import Graph
    G = Graph([(v,u) for v,u in edgeListSubgraph if v in VS and u in VS])

    def abstractPermutation2permutation(d):
        return [d[v] for v in VS] + VI + VR
    l = G.automorphism_group().list()
    l = map(lambda x: x.dict(), l)
    l = map(lambda x: abstractPermutation2permutation(x), l)

    with open(args.breakAutomorphismsSubgraph, "w") as fp:
        for _ in VS:
            print("1 ", end="", file=fp)
        print(len(VI), len(VR), file=fp)

        #print permutations
        for perm in l:
            print(" ".join(map(str, perm)), file=fp)


all_variables = []
all_variables += [('edge',(u,v)) for u,v in combinations(V,2)]
if args.equalityVersion:
    all_variables += [('equal',(v,v + 1)) for v in range(n - 1)]
    # all_variables += [('equal_extended',(u,v)) for u,v in combinations(V,2)]
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
def var_equal(u,v): return var(('equal',(u,v)))


constraints = []

#for i,j in combinations(V,2):
#    print(var_equal(i,j))

# for i,j in combinations(V,2):
#    print("c\t edge ", i,j, "to", var_edge(i,j))

#-------------------------create encoding-----------------------

if args.groetsch:
    # at least one induced grötsch graph
    from sage.all import Graph
    G = Graph([(v,u) for v,u in edgeListSubgraph if v in VS and u in VS])
    GroetzschGraph = Graph([(0,5),(0,6),(0,10),(1,7),(1,9),(1,10),(2,8),(2,9),(2,10),(3,6),(3,7),(3,10),(4,5),(4,8),(4,10),(5,7),(5,9),(6,8),(6,9),(7,8)])
    occurences = list(G.subgraph_search_iterator(GroetzschGraph))
    print(len(occurences), occurences, file=stderr)

    # remove duplicates thanks due to automorphism of Grötschgraph
    occurencesUnique = []
    for s in occurences:
        isPresent = False
        for s2 in occurencesUnique:
            if s.sorted() == s2.sorted():
                isPresent = True
                break
        if not isPresent:
            occurencesUnique.append(s)
    print(len(occurencesUnique), occurencesUnique)


# triangle free
for v1, v2, v3 in combinations(V, 3):
    constraints.append([-var_edge(v1,v2), -var_edge(v2,v3), -var_edge(v1,v3)])

# fix induced subgraph
if args.graphFile:
    print("c\t fix induced subgraph")
    for i,j in combinations(VS, 2):
        if (i,j) in edgeListSubgraph:
            if not args.subgraphIsNotInduced:
                constraints.append([+var_edge(i,j)])
        else:
            if not args.subgraphIsNotInduced2:
                constraints.append([-var_edge(i,j)])

if args.graphFile2:
    print("c\t fix induced subgraph2")
    print("c\t", edgeListSubgraph)
    for i,j in combinations(range(11), 2):
        if (i,j) in edgeListSubgraph or (j,i) in edgeListSubgraph:
                print((i,j),file=stderr)
                constraints.append([+var_edge(i,j)])
        else:
                constraints.append([-var_edge(i,j)])

if args.fixed:
    for i,j in combinations(range(args.fixed), 2):
        if (i,j) in edgeListSubgraph or (j,i) in edgeListSubgraph:
                constraints.append([+var_edge(i,j)])
        else:
                constraints.append([-var_edge(i,j)])

# independent set
for i,j in combinations(VI, 2):
    constraints.append([-var_edge(i,j)])

# maximal independent set
if len(VI) != 0:
    for i in VS + VR:
        constraints.append([+var_edge(i,j) for j in VI]) # adjacent to some vertex in VI



# degree constraints
d = [] # d[i] is variable indicating whether i has maximum degree
dAll = [] # all Degrees
if not args.degrees: # only if not predefined degrees
    for i in V:
        counterVariables = counterImplementations.counterFunction([var_edge(i,j) for j in V if j != i], countUpto=args.maxDegree, vPool=vpool,
            clauses=constraints, atMost=args.maxDegree, atLeast=args.minDegree, type=DEFAULT_COUNTER)
        d.append(counterVariables[args.maxDegree - 1]) # only true if a vertex has maximum degree
        dAll.append(counterVariables)

if args.exactMaxDegree:
    constraints.append(d)

# guess degree to other part
degreeToRemainingVertices = None
if args.degreeToRemainingPart:
    degreeToRemainingVertices = []
    minDegree = 4 # so more edges don't make sense
    degreeToRemainingVertices = [[vpool.id() for _ in range(minDegree - 1)] for _ in V] # if degreeToRemainingVertices[v][c] is true then at least c + 2 edges (At least one by default)

    # TODO Some up true variables: at most (8*7 - 25)
    countUpTo = 8*7 - 25 # Assumes all neighbors degree 8 so can be slightly improved
    counterVariables = [[ vpool.id() for _ in range(countUpTo)] for _ in range(25)]

    allVariablesDegreeToRemainingPart = []
    for step in range(25):
        allVariablesDegreeToRemainingPart = allVariablesDegreeToRemainingPart + degreeToRemainingVertices[step]

    counterImplementations.counterFunction(allVariablesDegreeToRemainingPart, countUpTo, vpool, constraints, atMost=countUpTo, type=DEFAULT_COUNTER)

    # TODO At most 8 with degree 1 to rest
    counterImplementations.counterFunction([-degreeToRemainingVertices[v][0] for v in V], 8, vpool, constraints, atMost=8, type=DEFAULT_COUNTER) # not more than 8 with not at least 2


    # TODO make all variables smaller variables true.
    for v in V:
        for i,j in combinations(range(minDegree - 1), 2):
            constraints.append([-degreeToRemainingVertices[v][j], +degreeToRemainingVertices[v][i]])

    for v in V:
        for i in range(minDegree - 1):
            constraints.append([-degreeToRemainingVertices[v][i], -dAll[v][8 - i - 2]]) # if degreeToRemainingVertices[i] and dAll[v][8 - i] to then i + 2 + 8 - i - 2 + 1 edges so 9

   


if args.equalityVersion:
    # equality gives intervalls
    #for i,j,k in combinations(V,3):
    #    constraints.append([-var_equal(i,k), var_equal(i,j)])
    #    constraints.append([-var_equal(i,k), var_equal(j,k)])
    # if equal than same degree
    for i in range(n - 1):
        j = i + 1
        for x in range(args.maxDegree):
            constraints.append([-var_equal(i,j), -dAll[i][x], +dAll[j][x]])
            constraints.append([-var_equal(i,j), +dAll[i][x], -dAll[j][x]])
    # if not equal than different degree
    for i in range(n - 1):
        j = i + 1
        clause = [+var_equal(i,j)]
        for x in range(args.maxDegree):
            q = vpool.id() # indicates that dAll[i][x] is false and dAll[j][x] is true. So x is the reason why this is indeed smaller
            constraints.append([-q, -dAll[i][x]])
            constraints.append([-q, +dAll[j][x]])

            constraints.append([+dAll[i][x], -dAll[j][x], q])
            clause.append(q)
        constraints.append(clause)
  
commonNeighbor = { (i,j,k): vpool.id() for i,j in combinations(V,2) for k in set(V)-{i,j}}

for i,j in combinations(V,2):
    for k in set(V)-{i,j}:
        L = (i,j,k) 
        constraints.append([-commonNeighbor[L], +var_edge(i,k)])
        constraints.append([-commonNeighbor[L], +var_edge(j,k)])
        constraints.append([+commonNeighbor[L], -var_edge(i,k), -var_edge(j,k)])

if args.mtfSpecial:
    for i,j in combinations(V,2):
        # if non of them has min degree then adjacent or common neighbor
        constraints.append([-dAll[i][args.minDegree], -dAll[j][args.minDegree], var_edge(i,j)] + [commonNeighbor[(i,j,k)] for k in set(V)-{i,j}]) # either one of them maximum degree or adjacent or common neighbor
        # if degree is maximal then radius 2
        constraints.append([-dAll[i][args.maxDegree - 1], var_edge(i,j)] + [commonNeighbor[(i,j,k)] for k in set(V)-{i,j}]) # either one of them maximum degree or adjacent or common neighbor
        constraints.append([-dAll[j][args.maxDegree - 1], var_edge(i,j)] + [commonNeighbor[(i,j,k)] for k in set(V)-{i,j}]) # either one of them maximum degree or adjacent or common neighbor
elif args.noMtfD:
    pass
elif args.mtfSpecial2:
    special = [vpool.id() for _ in V]
    for v in V:
        constraints.append([-dAll[i][5 - 1] , -special[v]]) # if at least 5  then not mindegree so not special

    mtfSpecialGraph = [[None for _ in V] for _ in V]
    for i,j in combinations(V,2):
        mtfSpecialGraph[i][j] = mtfSpecialGraph[j][i] = vpool.id()
        constraints.append([-var_edge(i,j), +mtfSpecialGraph[i][j]]) # proper supergraph
    for i,j,k in combinations(V,3):
        constraints.append([-mtfSpecialGraph[i][j], -mtfSpecialGraph[j][k], -mtfSpecialGraph[i][k]]) # triangle free
    for i in V:
        asdf = counterImplementations.counterFunction([mtfSpecialGraph[j][i] for j in V if j != i], 8, vpool, constraints, type=DEFAULT_COUNTER)
        # if not degree eight then not special
        constraints.append([+asdf[8 - 1], -special[i]])
    
    for i,j in combinations(V,2):
        # if both ar not special than add edge
        constraints.append([+special[i], +special[j], var_edge(i,j)] + [commonNeighbor[(i,j,k)] for k in set(V)-{i,j}]) # either one of them maximum degree or adjacent or common neighbor
        # if degree is maximal then radius 2
        constraints.append([-dAll[i][args.maxDegree - 1], var_edge(i,j)] + [commonNeighbor[(i,j,k)] for k in set(V)-{i,j}]) # either one of them maximum degree or adjacent or common neighbor
        constraints.append([-dAll[j][args.maxDegree - 1], var_edge(i,j)] + [commonNeighbor[(i,j,k)] for k in set(V)-{i,j}]) # either one of them maximum degree or adjacent or common neighbor

    if args.removeAble:
        for i in V: # check you many can be removed if vertex i and neighborhood and with small degree are removed
            var_removeable = []
            var_removeable = extend_set_of_removeable_vertices_wrapper(V, var_edge, vpool, constraints, 3, removeAtMost=8, start_from_indset=[i])
        
            for j in V:
                if j == i: continue
                constraints.append([-mtfSpecialGraph[i][j], +var_removeable[j]])
else:
    #mtf_d
    for i,j in combinations(V,2):
        constraints.append([d[i], d[j], var_edge(i,j)] + [commonNeighbor[(i,j,k)] for k in set(V)-{i,j}]) # either one of them maximum degree or adjacent or common neighbor
   

if args.realMTF:
    for i,j in combinations(V,2):
        constraints.append([var_edge(i,j)] + [commonNeighbor[(i,j,k)] for k in set(V)-{i,j}]) # either one of them maximum degree or adjacent or common neighbor

if args.adjacentOrAtmost3commonNeighbors:
    for i,j in combinations(V,2):
        asdf = counterImplementations.counterFunction([commonNeighbor[(i,j,k)] for k in set(V)-{i,j}], 3, vpool, constraints, atMost=3) # if they are adjacent then the have 0 common neighbor

    for i in V:
        for j,k in combinations(V,2):
            if j == i or k == i:
                continue
            # l is the unique neighbor in N[i] of j and k
            for l in set(V)-{i,j,k}:
                constraints.append([-commonNeighbor[(min(i,j),max(i,j),l)], -commonNeighbor[(min(i,k),max(i,k),l)]] + 
                    [+commonNeighbor[(min(i,j),max(i,j),m)] for m in set(V)-{i,j,k,l}] + 
                    [+commonNeighbor[(min(i,k),max(i,k),m)] for m in set(V)-{i,j,k,l}]) # Fails if both have l as common neighbor with i but nothing else

# more general version
if args.adjacentOrOtherMoreGeneral:
    constraints.append(d) # at least one vertex with maximum degree
    minDegreeReduction = args.adjacentOrOtherMoreGeneral # minimum degree after reducing a vertex and its neighborhood
    for i,j in permutations(V,2):
        upperBoundOfCommonNeighborsIfDegree7 = args.maxDegree - minDegreeReduction # at least degree upperBoundOfCommonNeighborsIfDegree7 must remain after deleting neighborhood of maximum degree vertex
        asdf = counterImplementations.counterFunction([commonNeighbor[(min(i,j),max(i,j),k)] for k in set(V)-{i,j}], upperBoundOfCommonNeighborsIfDegree7 + 1, vpool, constraints) # if they are adjacent then the have 0 common neighbor
        constraints.append([-d[i], -asdf[upperBoundOfCommonNeighborsIfDegree7]]) # Upperbound on max
        for c in range(args.minDegree, args.maxDegree):
            upperboundForLowDegreeVertex = c - minDegreeReduction
            constraints.append([-d[i], +dAll[j][c + 1 - 1], -asdf[upperboundForLowDegreeVertex]]) # if i has max degree and j has not at least c + 1 than

if args.increaseSpecial:
    for i in V:
        for j,k in combinations(V,2):
            if j == i or k == i:
                continue
            # l is the unique neighbor in N[i] of j and k
            for l in set(V)-{i,j,k}:
                constraints.append([-d[i], -commonNeighbor[(min(i,j),max(i,j),l)], -commonNeighbor[(min(i,k),max(i,k),l)]] + 
                    [+commonNeighbor[(min(i,j),max(i,j),m)] for m in set(V)-{i,j,k,l}] + 
                    [+commonNeighbor[(min(i,k),max(i,k),m)] for m in set(V)-{i,j,k,l}]) # Fails if both have l as common neighbor with i but nothing else

# for each maximum degree vertex there must be another one with maximum degree such that at most one common neighbor otherwise degree can be reduced by 2
# increase by 
if args.increaseSpecial2:
    for i in V:
        vertexSelection = [ vpool.id() for _ in V] 
        constraints.append([-d[i]] + vertexSelection)
        constraints.append([-vertexSelection[i]]) # not the vertex itself
        for j in V:
            if i == j: continue
            constraints.append([-vertexSelection[j], d[j]]) # if selected then maximumDegree
             # if selected then at most one common neighbor
            for k,m in combinations(V,2):
                if k in [i,j] or m in [i,j]:
                    continue
                constraints.append([-vertexSelection[j], -commonNeighbor[(min(i,j),max(i,j),k)], -commonNeighbor[(min(i,j),max(i,j),m)]])
                   
#print(f"{perf_counter()-t_begin:.3f} seconds were spend until now", file=stderr)
#remo_t = perf_counter()
if args.removeAble:
    assert(args.subgraphMinDegree)
    removeAtMost = len(VI) + len(VR)
    minDegreeSubgraph = args.subgraphMinDegree

    # expand graph to unique mtf; not necessary if already an mtf
    if not args.realMTF:
    
        var_mtf_edge = [[None for _ in V] for _ in V]
        for a,b in combinations(V,2):
            var_mtf_edge[b][a] = var_mtf_edge[a][b] = vpool.id()
            constraints.append([-var_edge(a,b), var_mtf_edge[a][b]])
            
        # triangular free
        for i,j,k in combinations(V,3):
            constraints.append([-var_mtf_edge[i][j], -var_mtf_edge[j][k], -var_mtf_edge[i][k]])

        commonNeighbor1 = { (i,j,k): vpool.id() for i,j in combinations(V,2) for k in set(V)-{i,j} if k < max(i,j)} # ik and jk are edges in the mtf
        commonNeighbor2 = { (i,j,k): vpool.id() for i,j in permutations(V,2) for k in set(V)-{i,j} if k < max(i,j)} # ik is an edge in the original and jk in the mtf

        # order of edges in colex ordering (i,k) < (i,j) if k < j; (j,k) < (i,j) if i < k; (i,k < (i,j) and (j,k) < (i,j) if i < min(i,j) )

        for i,j in combinations(V,2):
            for k in set(V)-{i,j}:
                if k >= min(i,j): continue
                L = (i,j,k) 
                constraints.append([-commonNeighbor1[L], +var_mtf_edge[i][k]])
                constraints.append([-commonNeighbor1[L], +var_mtf_edge[j][k]])
                constraints.append([+commonNeighbor1[L], -var_mtf_edge[i][k], -var_mtf_edge[j][k]])
        
        for i,j in permutations(V,2):
            for k in set(V)-{i,j}:
                if k >= i: continue
                L = (i,j,k) 
                constraints.append([-commonNeighbor2[L], +var_edge(i,k)])
                constraints.append([-commonNeighbor2[L], +var_mtf_edge[j][k]])
                constraints.append([+commonNeighbor2[L], -var_edge(i,k), -var_mtf_edge[j][k]])

        for i,j in combinations(V,2):
            # if edge is not present then adding this edge would result in a triangular with edges in the original graph and smaller edges in the mtf
            constraints.append([+var_mtf_edge[i][j]] + [+commonNeighbor[(i,j,k)] for k in set(V)-{i,j}] + 
                [+commonNeighbor2[(i,j,k)] for k in set(V)-{i,j} if k < i ] + # only applicable if (j,k) < (i,j) 
                [+commonNeighbor2[(j,i,k)] for k in set(V)-{i,j} if k < j] +
                [+commonNeighbor1[(i,j,k)] for k in set(V)-{i,j} if k < min(i,j)])

    
    for i in V: # check you many can be removed if vertex i and neighborhood and with small degree are removed
        

        var_removeable = []
        if not args.nonDecreasingIndpendentSetInReducer:
            var_removeable = extend_set_of_removeable_vertices_wrapper(V, var_edge, vpool, constraints, minDegreeSubgraph, removeAtMost=removeAtMost, start_from_indset=[i])
        else:
            # better reducer if size of independent set doesn't decrease
            var_removeable = extend_set_of_removeable_vertices_wrapper(V, var_edge, vpool, constraints, minDegreeSubgraph, start_from_indset=[i]) # without at most
            
            countRemoveAble = counterImplementations.counterFunction([var_removeable for j in V if j != i], len(VI) + len(VR) + 1, vpool, constraints, type=DEFAULT_COUNTER)
            if not args.realMTF:
                # if already more in independent set than better
                degMTF = counterImplementations.counterFunction([var_mtf_edge[i][j] for j in V if j != i], len(VI), vpool, constraints, atMost=len(VI), type=DEFAULT_COUNTER)
                constraints.append([-var_mtf_edge[i][j], +var_removeable[j]])
                constraints.append([-degMTF[len(VI) -1], -countRemoveAble[len(VI) + len(VR)]])
            else:
                constraints.append([-var_edge(i,j), +var_removeable[j]])
                if args.maxDegree == len(VI):
                    constraints.append([-d[i], -countRemoveAble[len(VI) + len(VR) - 1]])

        for j in V:
            if j == i: continue
            if not args.realMTF:
                constraints.append([-var_mtf_edge[i][j], +var_removeable[j]])
            else:
                constraints.append([-var_edge(i,j), +var_removeable[j]])
#remo_t_end = perf_counter()
#print(f"Removable encoded in {remo_t_end-remo_t:.3f} seconds",file=stderr)




# Neighborhoods are disjoint and one not a subset of the other
if args.basicCritical or args.critical:
    for i,j in permutations(V,2):
        if i in VS and j in VS: # since subgraph is vertex critical this doesn't have to be checked
            continue
        # There must be a vertex adjecent to i which is not adjacent to j
        adjacentOnlyToI = []
        for k in V:
            if k == i or k == j: continue
            kIsAdjacentOnlyToI = vpool.id()
            constraints.append([+var_edge(i,k), -kIsAdjacentOnlyToI])
            constraints.append([-var_edge(j,k), -kIsAdjacentOnlyToI])
            adjacentOnlyToI.append(kIsAdjacentOnlyToI)
        constraints.append([+var_edge(i,j)] + adjacentOnlyToI)

if args.critical:
    nColors = args.critical - 1
    for v in V:
        # check if G-v is args.critical - 1 colorable
        colors = [[vpool.id() for _ in V] for _ in range(nColors)]
        # at least one color
        for u in V:
            if u != v:
                constraints.append([colors[r][u] for r in range(nColors)])
        # adjacent once cannot have the same color
        for u1,u2 in combinations(V,2):
            if u1 == v or u2 == v:
                continue
            for r in range(nColors):
                constraints.append([-var_edge(u1,u2), -colors[r][u1], -colors[r][u2]])

if args.criticalAndReduceAble:
    nColors = args.criticalAndReduceAble - 1
    for v in [0,5,15,20,24]:
        # check if G-v is args.critical - 1 colorable
        colors = [[vpool.id() for _ in V] for _ in range(nColors)]
        adjacentAndSecondVertexColorC = [[[vpool.id() for _ in V] for _ in V] for _ in range(nColors)] # adjacentAndSecondVertexColorC[c][i][j] iff ij \in E and j color c
        adjacentToVertex_v_andSingleColor_c = [[vpool.id() for _ in V] for _ in range(nColors)]
        # at least one color
        for u in V:
            if u != v:
                constraints.append([colors[r][u] for r in range(nColors)])
        
        for i,j in permutations(V,2):
            for c in range(nColors):
                constraints.append([+colors[c][j],  -adjacentAndSecondVertexColorC[c][i][j]]) # if not color c then not adjacent and color c
                constraints.append([+var_edge(i,j),  -adjacentAndSecondVertexColorC[c][i][j]]) # not adjacent then not adjaent and color c
                constraints.append([-var_edge(i,j), -colors[c][j], +adjacentAndSecondVertexColorC[c][i][j]])

        for i in V:
            if i == v: continue
            for c in range(nColors):
                constraints.append([+var_edge(v,i), -adjacentToVertex_v_andSingleColor_c[c][i]])
                for c2 in range(nColors):
                    if c2 == c:
                        continue
                    constraints.append([-colors[c2][i], -adjacentToVertex_v_andSingleColor_c[c][i]])

        
        # adjacent once cannot have the same color
        for u1,u2 in combinations(V,2):
            if u1 == v or u2 == v:
                continue
            for r in range(nColors):
                constraints.append([-var_edge(u1,u2), -colors[r][u1], -colors[r][u2]])

        # colors form maximal independent set on V - v
        for u1 in V:
            if u1 == v: continue
            for c in range(nColors):
                constraints.append([colors[c][u1]] + [adjacentAndSecondVertexColorC[c][u1][u2] for u2 in V if u2 not in [u1,v]]) # color or adjacent to the color

        # v is adjacent to each each color, such that the vertex has a unique color; otherwise nColors-colorable
        for c in range(nColors):
            constraints.append([adjacentToVertex_v_andSingleColor_c[c][i] for i in V if i != v])

        removeAtMost = len(VI) + len(VR)
        for c in range(nColors):
            var_removeable = []
            var_removeable = extend_set_of_removeable_vertices_wrapper(V, var_edge, vpool, constraints, minDegreeSubgraph, removeAtMost=removeAtMost)
        
            for i in V:
                constraints.append([-colors[c][i], +var_removeable[i]]) # if it has the color then removeable

            if args.degreeToRemainingPart:
                var_removeable = check_removeableForExtendedGraph(V, var_edge, vpool, constraints, degreeToRemainingVertices, minDegreeSubgraph=6-1, removeAtMost=8)
                for i in V:
                    constraints.append([-colors[c][i], +var_removeable[i]]) # if it has the color then removeable
        

if args.largestIndependentSet:
        for vs in combinations(V, len(VI) + 1): # no independent set of size + 1
            constraints.append([+var_edge(u1,u2) for u1, u2 in combinations(vs, 2)]) # at least on edge

if args.explicitColorClauses:
    countColoringClauses = 0
    for coloring in itertools.product(range(3), repeat=n):
        coloringList = list(coloring)
        if coloringList[0] == coloringList[1] or coloringList[1] == coloringList[2] or coloringList[2] == coloringList[3] or coloringList[3] == coloringList[4] or coloringList[0] == coloringList[4]:
            continue
        unbalanced = False
        for i in range(3):
            if len([v for v in V if coloringList[v] == i]) < 4:
                unbalanced = True
                break

        if unbalanced:
            continue
        countColoringClauses += 1
        # one monochromatic edge
        constraints.append([var_edge(v1,v2) for v1, v2 in combinations(V,2) if coloringList[v1] == coloringList[v2]])
    print("c\tcoloring clauses: ", countColoringClauses)


if args.maxIndependentSet:
    for vs in combinations(V, args.maxIndependentSet + 1): # no independent set of size + 1
            constraints.append([+var_edge(u1,u2) for u1, u2 in combinations(vs, 2)]) # at least on edge


if args.numberOfEdges:
    counterImplementations.counterFunction([var_edge(i,j) for i,j in combinations(V,2)], args.numberOfEdges, 
        vpool, constraints, atMost=args.numberOfEdges, atLeast=args.numberOfEdges, type=DEFAULT_COUNTER)

if args.numberOfMaxdegreeVertices:
    counterImplementations.counterFunction(d, args.numberOfMaxdegreeVertices, vpool, constraints, atMost=args.numberOfMaxdegreeVertices, type=DEFAULT_COUNTER)

# vertex with minimum degree which is also critical
if args.minDegreeVertex:
    #for i in range(1, n - args.minDegree):
    #    constraints.append([-var_edge(0,i)])

    # first seven vertices have min degree
    # for i in range(7):
    #    constraints.append([-dAll[i][args.minDegree + 1 - 1]])
    minDegreeVariables = [-dAll[v][args.minDegree + 1 - 1] for v in V]  # if not at least args.minDegree + 1 then the have min degree
    counterImplementations.counterFunction(minDegreeVariables, 7, vpool, constraints, atLeast=7, type=DEFAULT_COUNTER)
    if True:
        # if both min degree than not same neighborhood
        for i,j in permutations(V,2):
            if i in VS and j in VS: # since subgraph is vertex critical this doesn't have to be checked
                continue
            # There must be a vertex adjecent to i which is not adjacent to j
            adjacentOnlyToI = []
            for k in V:
                if k == i or k == j: continue
                kIsAdjacentOnlyToI = vpool.id()
                constraints.append([+var_edge(i,k), -kIsAdjacentOnlyToI])
                constraints.append([-var_edge(j,k), -kIsAdjacentOnlyToI])
                adjacentOnlyToI.append(kIsAdjacentOnlyToI)
            constraints.append([+dAll[i][args.minDegree + 1 - 1], +dAll[j][args.minDegree + 1 - 1], +var_edge(i,j)] + adjacentOnlyToI)

    if False: # destroys number of edges
        # mtf for vertices which are not the first 7
        for i,j in combinations(range(7,n),2):
            constraints.append([d[i], d[j], var_edge(i,j)] + [commonNeighbor[(i,j,k)] for k in set(V)-{i,j}]) # either one of them maximum degree or adjacent or common neighbor

       

if args.edgeCritical:
    nColors = args.edgeCritical - 1
    for i,j in combinations(V,2):
        # check if G-v is args.critical - 1 colorable
        colors = [[vpool.id() for _ in V] for _ in range(nColors)]
        # at least one color
        for u in V:
            constraints.append([colors[r][u] for r in range(nColors)])
        # adjacent once cannot have the same color
        for u1,u2 in combinations(V,2):
            if (u1,u2) == (i,j):
                continue
            for r in range(nColors):
                constraints.append([-var_edge(i,j), -var_edge(u1,u2), -colors[r][u1], -colors[r][u2]])



# TODO try to remove more vertices from the subgraph i.e., discard graphs with smaller subgraphs

if args.partition:
	pos = 0
	for i in range(len(args.partition)):
		posNew = pos + args.partition[i]

		# Each vertex has exactly degree  args.degrees[i]
		for v in range(pos, posNew):
			degree = args.degrees[i]
			counterImplementations.counterFunction([var_edge(v,u) for u in V if u != v], countUpto=degree, vPool=vpool, clauses=constraints, atLeast=degree, atMost=degree, type=DEFAULT_COUNTER)
		pos = posNew


if args.MDE:
    assert(len(VR) == 1)
    assert(len(VI) == args.maxDegree)
    for v in VR:
        for u in VI:
            constraints.append([+var_edge(v,u)])

# remaining vertex has degree 1 or 0 with rest
if args.EMDE:
    assert(len(VR) == 1)
    for v in VR:
        for u1,u2 in combinations(VS, 2):
            constraints.append([-var_edge(v,u1), -var_edge(v,u2)])

# remaining vertices have low degree to the remaining part
if args.EMDE2:
    minDegreeSubgraph = args.subgraphMinDegree
    # print("minDegreesubgraph", minDegreeSubgraph, file=stderr)
    for i in VR:
        counterImplementations.counterFunction([+var_edge(i,j) for j in VS ], minDegreeSubgraph - 1, vPool=vpool,
            clauses=constraints, atMost=args.subgraphMinDegree - 1, type=DEFAULT_COUNTER)
    
    nr = len(VR)
    ordering = [[vpool.id() for _ in range(nr)] for _ in range(nr)]
    for i,j,k in permutations(range(nr), 3):
        constraints.append([-ordering[i][j], -ordering[j][k], ordering[i][k]]) # transitive
    for i,j in combinations(range(nr), 2):
        constraints.append([+ordering[i][j], +ordering[j][i]]) # total
    adjacentAndBefore = [[vpool.id() for _ in range(nr)] for _ in range(nr)]
    for i,j in permutations(range(nr), 2):
        constraints.append([-ordering[i][j], -var_edge(VR[i], VR[j]), +adjacentAndBefore[j][i]])
    for i in range(nr):
        counterImplementations.counterFunction([+var_edge(VR[i],j) for j in VS ] + [+adjacentAndBefore[i][j] for j in range(nr) if j != i], minDegreeSubgraph - 1, vPool=vpool,
            clauses=constraints, atMost=args.subgraphMinDegree - 1, type=DEFAULT_COUNTER)
        
if args.EMDE3:
    minDegreeSubgraph = args.subgraphMinDegree
    # print("minDegreesubgraph", minDegreeSubgraph, file=stderr)
    
    nr = len(VR)
    independentSet = [vpool.id() for _ in range(nr)]

    ordering = [[vpool.id() for _ in range(nr)] for _ in range(nr)]
    for i,j,k in permutations(range(nr), 3):
        constraints.append([-ordering[i][j], -ordering[j][k], ordering[i][k]]) # transitive
    for i,j in combinations(range(nr), 2):
        constraints.append([+ordering[i][j], +ordering[j][i], independentSet[i], independentSet[j]]) # total ordering on vertices not in independent set
        constraints.append([-independentSet[i], -independentSet[j], -var_edge(VR[i], VR[j])])
    adjacentAndBefore = [[vpool.id() for _ in range(nr)] for _ in range(nr)]
    for i,j in permutations(range(nr), 2):
        constraints.append([-ordering[i][j], -var_edge(VR[i], VR[j]), +adjacentAndBefore[j][i]])
    for i in range(nr):
        asdf = counterImplementations.counterFunction([+var_edge(VR[i],j) for j in VS ] + [+adjacentAndBefore[i][j] for j in range(nr) if j != i], minDegreeSubgraph, vPool=vpool,
            clauses=constraints, type=DEFAULT_COUNTER)
        constraints.append([+independentSet[i], -asdf[minDegreeSubgraph - 1]]) # degree constraints must hold if not in the independent set

if args.subgraphMinDegree and args.subgraphIsNotInduced:
     # remaining vertex has max degree and only adjacent to vertices in the independent set
    
    degreeVariablesSubgraph = []
    for i in VS:
        asdf = counterImplementations.counterFunction([+var_edge(i,j) for j in VS if j != i], args.maxDegreeSubgraph if args.maxDegreeSubgraph else args.subgraphMinDegree, vPool=vpool,
            clauses=constraints, atLeast=args.subgraphMinDegree, type=DEFAULT_COUNTER)
        if args.maxDegreeSubgraph: degreeVariablesSubgraph.append(+asdf[-1])
    

if args.basicCriticalSubgraph:
    # Also disjoinz neighborhood in the subgraph
    for i,j in permutations(VS,2):
        # There must be a vertex adjecent to i which is not adjacent to j
        adjacentOnlyToI = []
        for k in VS:
            if k == i or k == j: continue
            kIsAdjacentOnlyToI = vpool.id()
            constraints.append([+var_edge(i,k), -kIsAdjacentOnlyToI])
            constraints.append([-var_edge(j,k), -kIsAdjacentOnlyToI])
            adjacentOnlyToI.append(kIsAdjacentOnlyToI)
        constraints.append([+var_edge(i,j)] + adjacentOnlyToI)



#from sage.all import Graph
#from sage.graphs.independent_sets import IndependentSets


#special_t = perf_counter()
if args.special:
    removeAtMost = len(VR) + len(VI)
    minDegreeSubgraph = args.subgraphMinDegree

    #G = Graph(edgeListSubgraph)
    #IS = IndependentSets(G, maximal=True)

    largeMIS = [I for I in maximalIndependentSets if len(I) >= args.special]
    if args.nonDecreasingIndpendentSetInReducer:
        largeMIS = [I for I in maximalIndependentSets if len(I) >= len(VI) - 1]
    for I in maximalIndependentSets:
        if len(I) >= removeAtMost:
            print("c\t found larger independent set given by", I + VR)
            exitWithUnsatEncoding()

    
    for I in largeMIS: # I is a maximal independent set of the subgraph if already large enough
        if len(I) == removeAtMost - 1: # we have one other which can be removed
            for v in set(VS) - set(I):
                constraints.append([+var_edge(v,u) for u in I])

        # removeable thing starting with independent set
        if len(I + VR) < removeAtMost:
            var_removeable = extend_set_of_removeable_vertices_wrapper(V, var_edge, vpool, constraints, minDegreeSubgraph, 
                removeAtMost=removeAtMost, start_from_indset= I + VR)
        else:
            var_removeable = extend_set_of_removeable_vertices_wrapper(V, var_edge, vpool, constraints, minDegreeSubgraph, 
                 start_from_indset = I + VR)
            # all remaining must be not removeable
            for v in V:
                if v in I + VR: continue
                constraints.append([-var_removeable[v]])
#special_t_end = perf_counter()
#print(f"Special encoded in {special_t_end-special_t:.3f} seconds",file=stderr)


if args.special_2:
    removeAtMost = len(VR) + len(VI)
    minDegreeSubgraph = args.subgraphMinDegree

    #G = Graph(edgeListSubgraph)
    #IS = IndependentSets(G, maximal=True)
    largeMIS = [I for I in maximalIndependentSets if len(I) >= args.special_2]
    
    for I in maximalIndependentSets:
        if len(I) >= removeAtMost:
            print("c\t found larger independent set given by", I + VR)
            exitWithUnsatEncoding()

    
    for I in largeMIS: 
        # I is a maximal independent set of the subgraph if already large enough
        if len(I) == removeAtMost - 1: # we have one other which can be removed
            for v in set(VS) - set(I):
                constraints.append([+var_edge(v,u) for u in I])

        var_removeable = []
        # removeable thing starting with independent set
        if len(I + VR) < removeAtMost:
            var_removeable = extend_set_of_removeable_vertices_wrapper(V, var_edge, vpool, constraints, minDegreeSubgraph, 
                removeAtMost=removeAtMost, start_from_indset= I + VR)
        else:
            var_removeable = extend_set_of_removeable_vertices_wrapper(V, var_edge, vpool, constraints, minDegreeSubgraph, 
                 start_from_indset = I + VR)
            # all remaining must be not removeable
            for v in V:
                if v in I + VR: continue
                constraints.append([-var_removeable[v]])

        # ---------------------------Difference to previous one--------------------------
        var_eligable = [vpool.id() for _ in V]
        for v in set(V) - set(I):
            constraints.append([+var_edge(v,u) for u in I] + [var_eligable[v]]) # not adjacent implies elligable
            for u in I:
                [-var_eligable[v], -var_edge(v,u)] # elligable implies not adjacent
        # pick smallest elligable one to add to independent set
        VwithoutIndependentSet = set(V) - set(I + VR)
        for v in VwithoutIndependentSet:
            # if elligable and all smaller once not elligable then removeable
            constraints.append([-var_eligable[v]] + [+var_eligable[u] for u in VwithoutIndependentSet if u < v] + [var_removeable[v]])
        
if args.special2:
    removeAtMost = len(VR) + len(VI)
    minDegreeSubgraph = args.subgraphMinDegree

    #G = Graph(edgeListSubgraph)
    #IS = IndependentSets(G, maximal=True)

    largeMIS = [I for I in maximalIndependentSets if len(I) >= args.special2] # TODO 4 is kind of arbitrary
    for I in largeMIS:
        if len(I) > removeAtMost:
            print("c\tFound dependent set; nothing to solve")
            exitWithUnsatEncoding()

    
    for I in largeMIS:
        print("c\t", I, len(I))
        # I is a maximal independent set of the subgraph if already large enough
        if len(I) == removeAtMost:
            for v in set(VS) - set(I):
                constraints.append([+var_edge(v,u) for u in I])
        heuristic_removable_set_not_very_large(V, var_edge, vpool, constraints, minDegreeSubgraph, removeAtMost, args.maxDegree if args.maxDegree else len(V) - 1, counter="sequential", start_from_indset=I)


if False:
    import random
    for _ in range(30):
        VReordered = list(range(n))
        VReordered = VS
        random.shuffle(VReordered)
        VReordered = VR + VI + VReordered
        greedy_coloring_clauses_degree(VReordered, var_edge, vpool, constraints, nColors=4)
    largeMIS = [I for I in maximalIndependentSets if len(I) >= 6]
    for I in largeMIS:
        # version 1
        # VReordered = I + [v for v in V if v not in I]
        # greedy_coloring_clauses_degree_with_BCP_version2(VReordered, var_edge, vpool, constraints, nColors=4)

        # version2
        for _ in range(10):
            VReordered = [v for v in V if v not in I]
            random.shuffle(VReordered)
            greedy_coloring_clauses_degree(VReordered, var_edge, vpool, constraints, nColors=3)

if args.partialColorExpansion:
    greedy_coloring_clauses_degree_with_BCP_version_predefinedColorsVersion2(edgeListSubgraph, VS, VR, VI, var_edge, vpool, constraints, nColors = 4, 
        nColorings=args.partialColorExpansionNColors, imbalanced=args.partialColorExpansionImbalanced)
    # greedy_coloring_clauses_degree_with_BCP_version_predefinedColorsVersion2(edgeListSubgraph, list(range(11)), VR + [11, 12, 13], VI, var_edge, vpool, constraints, nColors = 4)

if args.special4:
    removeAtMost = len(VR) + len(VI)
    minDegreeSubgraph = args.subgraphMinDegree

    #G = Graph(edgeListSubgraph)
    #IS = IndependentSets(G, maximal=True)

    relevantVertices = VI + VR

    largeMIS = [I for I in maximalIndependentSets if len(I) >= args.special4]
    if args.nonDecreasingIndpendentSetInReducer:
        largeMIS = [I for I in maximalIndependentSets if len(I) >= len(VI) - 1] # improves at most by 1
    for I in largeMIS:
        # I = I + VR
        if len(I) > removeAtMost:
            print("c\tFound dependent set; nothing to solve")
            exitWithUnsatEncoding()

        var_eligable = [vpool.id() for _ in V]
        for v in relevantVertices:
            constraints.append([+var_edge(v,u) for u in I] + [var_eligable[v]]) # not adjacent implies elligable
            for u in I:
                [-var_eligable[v], -var_edge(v,u)] # elligable implies not adjacent

        
       
        if not args.nonDecreasingIndpendentSetInReducer or len(I) == len(VI):
            var_removeAble = extend_set_of_removeable_vertices_wrapper(V, var_edge, vpool, constraints, minDegreeSubgraph, 
                removeAtMost=removeAtMost, start_from_indset= I)
        else:
            var_removeAble = extend_set_of_removeable_vertices_wrapper(V, var_edge, vpool, constraints, minDegreeSubgraph, start_from_indset= I)
            countRemoveAble = counterImplementations.counterFunction([var_removeable for j in V if j not in I], len(VR) + 2, vpool, constraints, type=DEFAULT_COUNTER)
            for v in V:
                if v in I: continue
                constraints.append([-var_eligable[v], -countRemoveAble[len(VI) + len(VR) - len(I)]])
            # if there is an additional independent vertex than we can check if larger reducer.

        # pick smallest elligable one to add to independent set
        for v in relevantVertices:
            # if elligable and all smaller once not elligable then removeable
            constraints.append([-var_eligable[v]] + [+var_eligable[u] for u in relevantVertices if u < v] + [var_removeAble[v]])

        '''
        if args.degreeToRemainingPart:
            var_removeable = check_removeableForExtendedGraph(V, var_edge, vpool, constraints, degreeToRemainingVertices, minDegreeSubgraph=6-1, removeAtMost=8, start_from_indset=I)
            for v in relevantVertices:
                # if elligable and all smaller once not elligable then removeable
                constraints.append([-var_eligable[v]] + [+var_eligable[u] for u in relevantVertices if u < v] + [var_removeAble[v]])


        # add two vertices to independent set if possible; can use the same elligable variables
        addTwoVertices = vpool.id()
        addThreeVertices = vpool.id()
        
        for u,v in combinations(relevantVertices,2):
            constraints.append([-var_eligable[u], -var_eligable[v], +var_edge(v,u), +addTwoVertices]) # both elligable and not adjacent implies add two 

        for u,v,w in combinations(relevantVertices,3):
            constraints.append([-var_eligable[u], -var_eligable[v], -var_eligable[w], +var_edge(v,u), +var_edge(v,w), +var_edge(w,u), +addThreeVertices]) # both elligable and not adjacent implies add two 
        
        var_removeAble = extend_set_of_removeable_vertices_wrapper(V, var_edge, vpool, constraints, minDegreeSubgraph, 
            removeAtMost=removeAtMost, start_from_indset= I)
        
        # at least two removeable in # TODO at most two can be replaced by easier functions
        couter_vars = counterImplementations.counterFunction([var_removeAble[v] for v in relevantVertices], 3, vpool, constraints, type=DEFAULT_COUNTER)
        constraints.append([-addTwoVertices, +couter_vars[2 - 1]])
        constraints.append([-addThreeVertices, +couter_vars[3 - 1]]) ''' 

    ''' 
    for I in largeMIS:
        if len(I) > removeAtMost:
            print("c\tFound dependent set; nothing to solve")
            exitWithUnsatEncoding()

        potential_eligable_pairs = list(combinations(relevantVertices, 2))
        var_eligable = [vpool.id() for _ in V]
        var_eligable_pair = { (u,v): vpool.id() for u,v in potential_eligable_pairs}
        var_eligable_pair2 = { (u,v): vpool.id() for u,v in potential_eligable_pairs} # the selected eligable pair
        exists_eligable_pair = vpool.id() # true if there is some eligable pair
        
        addTwoVertices = vpool.id()
        for v in relevantVertices:
            constraints.append([+var_edge(v,u) for u in I] + [var_eligable[v]]) # not adjacent implies elligable
            for u in I:
                [-var_eligable[v], -var_edge(v,u)] # elligable implies not adjacent
        
        for u,v in potential_eligable_pairs:
            constraints.append([-var_eligable[u], -var_eligable[v], +var_edge(u,v), +var_eligable_pair[(u,v)]]) # both elligable and not adjacent implies add two 
        
        var_removeAble = extend_set_of_removeable_vertices_wrapper(V, var_edge, vpool, constraints, minDegreeSubgraph, 
            removeAtMost=removeAtMost, start_from_indset= I)

        for p in potential_eligable_pairs:
            constraints.append([-var_eligable_pair[p], +exists_eligable_pair])
            constraints.append([-var_eligable_pair2[p], var_removeAble[p[0]]])
            constraints.append([-var_eligable_pair2[p], var_removeAble[p[1]]])
        
        constraints.append([-exists_eligable_pair] + [+var_eligable_pair2[p] for p in potential_eligable_pairs]) '''

        




# TODO not in use
if args.special3:
    removeAtMost = len(VR) + len(VI)
    minDegreeSubgraph = args.subgraphMinDegree

    #G = Graph(edgeListSubgraph)
    #IS = IndependentSets(G, maximal=True)

    largeMIS = [I for I in maximalIndependentSets if len(I) >= args.special3] # TODO 4 is kind of arbitrary
    for I in largeMIS:
        if len(I) > removeAtMost:
            constraints = []
            print("c\tFound dependent set; nothing to solve")
            exitWithUnsatEncoding()

    
    for I in largeMIS:
        print("c\t", I, len(I))
        # I is a maximal independent set of the subgraph if already large enough
        if len(I) == removeAtMost:
            for v in set(VS) - set(I):
                constraints.append([+var_edge(v,u) for u in I])

        var_removeable = [ vpool.id() for _ in V]
        constraints.append([var_removeable[i]])
        for i in I:
            constraints.append([+var_removeable[j]])

        var_counter = counterVariablesRemoveable = counterImplementations.counterFunction([ var_removeable[u] for u in V if u not in I], removeAtMost  - len(I), vpool, constraints, atMost = removeAtMost  - len(I),    type=DEFAULT_COUNTER)

        var_edges_afterRemoval = [[None for _ in V] for _ in V]
        for a,b in combinations(V,2):
            var_edges_afterRemoval[b][a] = var_edges_afterRemoval[a][b] = vpool.id()
            constraints.append([+var_edge(a,b), -var_edges_afterRemoval[a][b]]) # no edge implies no edge after removal
            constraints.append([-var_removeable[a], -var_edges_afterRemoval[a][b]]) # if one of the vertices is removed than edge not present.
            constraints.append([-var_removeable[b], -var_edges_afterRemoval[a][b]])
        
        degreesOfRemainingGraph = []
        for j in V:
            if j == i: continue
            asdf = counterImplementations.counterFunction([var_edges_afterRemoval[a][j] for a in V if a != j], args.maxDegreeSubgraph if args.maxDegreeSubgraph else minDegreeSubgraph, vpool, constraints, type=DEFAULT_COUNTER)
            constraints.append([+var_removeable[j], +asdf[minDegreeSubgraph - 1]]) #if not removable then min degree

            if args.maxDegreeSubgraph: degreesOfRemainingGraph.append(+asdf[args.maxDegreeSubgraph - 1])

        # if removeable is the same number at least one with maximal degree
        if args.maxDegreeSubgraph:
            constraints.append([-counterVariablesRemoveable[-1]] + degreesOfRemainingGraph) # if maximal was removed then there must be a degree 6 vertex
        


        for v in V if args.subgraphIsNotInduced else VS + VI:
            if v in I: continue

            var_removeable = [ vpool.id() for _ in V]
            constraints.append([var_removeable[i]])
            for i in I + [v]:
                constraints.append([+var_removeable[j]])

            var_counter = counterVariablesRemoveable = counterImplementations.counterFunction([ var_removeable[u] for u in V if u not in I], removeAtMost + 1 - len(I), vpool, constraints, type=DEFAULT_COUNTER)
            constraints.append([-var_counter[removeAtMost - len(I)]] + [+var_edge(v,u) for u in I]) # either not a lot can be removed or

            var_edges_afterRemoval = [[None for _ in V] for _ in V]
            for a,b in combinations(V,2):
                var_edges_afterRemoval[b][a] = var_edges_afterRemoval[a][b] = vpool.id()
                constraints.append([+var_edge(a,b), -var_edges_afterRemoval[a][b]]) # no edge implies no edge after removal
                constraints.append([-var_removeable[a], -var_edges_afterRemoval[a][b]]) # if one of the vertices is removed than edge not present.
                constraints.append([-var_removeable[b], -var_edges_afterRemoval[a][b]])
            
            degreesOfRemainingGraph = []
            for j in V:
                if j == i: continue
                asdf = counterImplementations.counterFunction([var_edges_afterRemoval[a][j] for a in V if a != j], args.maxDegreeSubgraph if args.maxDegreeSubgraph else minDegreeSubgraph, vpool, constraints, type=DEFAULT_COUNTER)
                constraints.append([+var_removeable[j], +asdf[minDegreeSubgraph - 1]]) #if not removable then min degree

                if args.maxDegreeSubgraph: degreesOfRemainingGraph.append(+asdf[args.maxDegreeSubgraph - 1])

            # if removeable is the same number at least one with maximal degree
            if args.maxDegreeSubgraph:
                constraints.append([-counterVariablesRemoveable[-1]] + degreesOfRemainingGraph) # if maximal was removed then there must be a degree 6 vertex
        
print("c\tTotal number of constraints:", len(constraints))
print("c\tTotal number of variables:", vpool.id())


#t_end = perf_counter()
#t = t_end-t_begin
#print(f"Encoding completed in {t:.3f} seconds", file=stderr)

print('c\tbegin of CNF')
# disable for production runs, it's quite costly, especially when doing lots of small expansions
#for c in constraints:
#    for x in c:
#        if type(x) != int:
#            exit("Error")
#
#t_check = perf_counter()
#t = t_check-t_end
#print(f"Clauses checked in {t:.3f} seconds", file=stderr)

#for c in constraints:
#    print (' '.join(str(x) for x in c))
#print("\n".join(" ".join([str(x) for x in C]) for C in constraints))
print("\n".join(" ".join(map(str, C)) for C in constraints)) # performs best
print('c\tend of CNF')

#jt_close= perf_counter()
#jt = t_close-t_check
#jprint(f"Encoding printed in {t:.3f} seconds", file=stderr)

t_tot = perf_counter()
print(f"Encoding time: {perf_counter()-t_begin:.4g}", file=stderr)
