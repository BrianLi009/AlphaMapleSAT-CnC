#!/usr/bin/env python3

from itertools import combinations, permutations, chain
from argparse import ArgumentParser
from counterImplementations import counterFunction
from sys import *
from operator import indexOf

parser = ArgumentParser()
parser.add_argument("--vertices", "-n", type=int, required=True)
parser.add_argument("--partition", nargs="+", type=int, help="Use partition based version. The vertex set V can be partitioned in 4 independent sets")
parser.add_argument("-x", "--explicit-colorings", type=int, help="explicitly enumerate and avoid all colorings where at most X vertices have color 1 (the edge-proper color) (X == n for all colorings)")
parser.add_argument("-d", "--degree", action="store_true", help="enforce maxdegree constraints")
parser.add_argument('--exchangeAble', "-e", action="store_true", help="Test swapping pairs of vertices which can be swapped according to the independent sets")
parser.add_argument('--increaseFirstParition', "-i", action="store_true", help="Testing whether paritions can be increased based on connected components.")
parser.add_argument('--connectedComponents', "-c", action="store_true", help="Calulate connected components within encoding and try to increase partitions.")
parser.add_argument('--maximalFourCycleFree', action="store_true", help="Inserting any additional edge will result in a 4-cycle")
parser.add_argument('--edgeVersion', action="store_true", help="Each edge must be in a triangle")
parser.add_argument("--externalCNFs", type=str, nargs="+", help="Cnfs which should be included to the encoding")



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

triangle_vars = [[[None for _ in V] for _ in V] for _ in V] # = list(range(_num_vars+1, _num_vars+1+comb(n,3)))
for i,j,k in combinations(V,3):
    triangle_vars[i][j][k] = vpool.id()
    for i2, j2, k2 in permutations([i,j,k], 3):
        triangle_vars[i2][j2][k2] = triangle_vars[i][j][k]

def var_triangle(u,v,w):
    return triangle_vars[u][v][w]

constraints = []

# define triangle vars

for u, v, w in combinations(V, 3):
    constraints.append([-var_triangle(u, v, w), var_edge(u, v)])
    constraints.append([-var_triangle(u, v, w), var_edge(v, w)])
    constraints.append([-var_triangle(u, v, w), var_edge(u, w)])
    constraints.append([+var_triangle(u, v, w), -var_edge(u, v), -var_edge(v, w), -var_edge(u, w)])

#the sought graph should be
#    - C_4-free (as subgraph)
#    - every vertex should be a part of a triangle
#    - degree >= 3
#    - 4-colorable
#    - not 3-colorable (handled with alternating search)
#    - connected (not used for now)

for v, w, x, y in permutations(V, 4):
    if v != min([v,w,x,y]): continue
    if w > y: continue
    constraints.append([-var_edge(v, w), -var_edge(w, x), -var_edge(x, y), -var_edge(y, v)])

for u in V:
    u_is_in_a_triangle = []
    for v in V:
        if v == u:
            continue
        for w in range(v+1, n):
            if w != u:
                u_is_in_a_triangle.append(var_triangle(u, v, w))
    constraints.append(u_is_in_a_triangle)

if not args.edgeVersion:
    for v in V:
        counterFunction([var_edge(v, w) for w in V if w != v], 3, vpool, constraints, atLeast=3, type="sequential")
else:
    # at least one edge
    for v in V:
        constraints.append([var_edge(v, w) for w in V if w != v])

if args.edgeVersion:
    for v,u in combinations(V,2):
        constraints.append([-var_edge(v,u)] + [+var_triangle(v,u,w) for w in V if w not in [v,u]])

if not args.partition:
    # 4-colorability
    color = [ [vpool.id() for i in range(4)] for v in V]
    for v in V:
        constraints.append([color[v][i] for i in range(4)]) # each vertex should have a color

    for v, w in combinations(V, 2):
        for i in range(4):
            constraints.append([-var_edge(v, w), -color[v][i], -color[w][i]])

def CNF_OR(ins, out):
    return [[-out] + ins] + [[out, -x] for x in ins]

def CNF_AND(ins, out):
    return [[out] + [-x for x in ins]] + [[-out, x] for x in ins]

if args.maximalFourCycleFree:
    for i,j in combinations(V,2):
        rounds = range(3)
        reached = [[ (vpool.id() if r != 0 else var_edge(i,v1)) if v1 != i else None for r in rounds] for v1 in V]

    for r in range(1,3):
        for v1 in V:
            if v1 == i:
                continue
            reachedVia = []
            for v2 in V:
                if v2 in [v1, i]: continue
                # v1 reach via v2
                newVar = vpool.id()
                clauses = CNF_AND([reached[v2][r-1], var_edge(v1,v2)], newVar)
                constraints.extend(clauses)
                reachedVia.append(newVar)
            clauses = CNF_OR(reachedVia, reached[v1][r])
            constraints.extend(clauses)
    constraints.append([reached[j][3 - 1]]) # 3 edges to get to j also allowing using same edge twice




if args.partition:
    partitions = []
    current = 0
    for i in args.partition:
        partitions.append(list(range(current, current + i)))
        current += i
    print("c\tpartitions:", partitions, file=stderr)

    # vertices in partition are not adjacent
    for p in partitions:
        for i,j in combinations(p,2):
            constraints.append([-var_edge(i,j)])

    for p1,p2 in combinations(partitions,2):
        for v in p2:
            constraints.append([+var_edge(u,v) for u in p1])
        if len(p1) ==  len(p2):
            # also in other direction
            for v in p1:
                constraints.append([+var_edge(u,v) for u in p2])
    
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

                # Compare counts after swapping
                count1 = counterFunction([reached[(x,lastStep)] for x in p1], len(p1), vpool, clauses=constraints, type="sequential") # at least N

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
                count2 = counterFunction(list2, len(p1) + 1, vpool, clauses=constraints, type="sequential")
                constraints.append([-count2[len(p1)]]) # not allowed to be more

                for i in range(len(p1)):
                    constraints.append([+count1[i], -count2[i]]) # count2 <= count1 


# no vertex of full degree
# calculate maximum possible degree
if args.degree:
    #delta = (n - 1) // 2
    #if delta % 2 == 1 and delta > (n - 2) // 2:
    #    delta -= 1
    delta = (n - 2) // 2
    for v in V:
        #constraints.append([-var_edge(v, w) for w in V if v != w])
        counterFunction([+var_edge(u, v) for u in V if u != v], delta, vpool, constraints, atMost=delta)

''' 
for i,j in permutations(V,2):
    # There must be a vertex adjecent to i which is not adjacent to j
    adjacentOnlyToI = []
    for k in V:
        if k == i or k == j: continue
        kIsAdjacentOnlyToI = vpool.id()
        constraints.append([+var_edge(i,k), -kIsAdjacentOnlyToI])
        constraints.append([-var_edge(j,k), -kIsAdjacentOnlyToI])
        adjacentOnlyToI.append(kIsAdjacentOnlyToI)
    constraints.append([+var_edge(i,j)] + adjacentOnlyToI) '''

# 101 coloring for all proper subgraphs
if False:
    for x in V:
        color = [ vpool.id() if v != x else None for v in V]
        for v, w in combinations(V, 2):
            if x in [v,w]: continue
            constraints.append([-var_edge(v, w), -color[v], -color[w]])
        for v, w, u in combinations(V, 3):
            if x in [v,w,u]: continue
            constraints.append([-var_triangle(v, w,u), +color[v], +color[w], +color[u]])

# explicit zero one zero coloring
if args.explicit_colorings:
    Vset = set(V)
    # bounds:
    #  k = 0 means there should be at least one triangle, not necessary
    #  k = 1 means there should be a triangle avoiding every vertex, but this already follows from triangle covering, square-freeness, and min degree
    #  k = n means there should be at least one edge, not necessary
    #  k = n-1 means there should be an edge avoiding every vertex (i.e. not a star), but this follows from triangle covering or min degree
    #  k = n-2 means there should be an edge avoiding every pair, again follows from min degree
    #  this the meaningful values of k are 2..n-3
    lo = 2
    hi = min(args.explicit_colorings, n-3)
    for color1 in chain.from_iterable(combinations(V, k) for k in range(2, hi+1)):
        color0 = Vset - set(color1)
        
        # monochromatic edge with color 1 or monochromatic triangle with color 0
        constraints.append([+var_edge(i,j)  for i,j in combinations(color1,2)] + [+var_triangle(i,j,k)  for i,j,k in combinations(color0,3)] )

if False:
    # every non-edge must result in a 4-cycle if added
    for i,j in combinations(V,2):
        reason1 = [vpool.id() if v not in [i,j] else None for v in V]
        reason2 = [vpool.id() if v not in [i,j] else None for v in V]
        constraints.append([+var_edge(i,j)] + [reason1[k] for k in V if k not in [i,j]]) # edge implies there is a reason
        constraints.append([+var_edge(i,j)] + [reason2[k] for k in V if k not in [i,j]])

        for k,l in permutations(V,2):
            if k in [i,j] or l in [i,j]: continue
            constraints.append([-reason1[k], -reason2[l], var_edge(k,l)])
        for k in V:
            if k in [i,j]: continue
            constraints.append([-reason1[k], var_edge(k,i)])
            constraints.append([-reason2[k], var_edge(k,j)])

            constraints.append([-reason1[k], -reason2[k]]) # triangle is no reason


print("\n".join(" ".join(map(str, C)) for C in constraints))
if args.externalCNFs:
    for s in args.externalCNFs:
        with open(s, 'r') as f:
            for line in f:
                print(line, end="")

