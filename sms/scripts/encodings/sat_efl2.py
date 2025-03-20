#!/usr/bin/python

from itertools import combinations, permutations
from sys import *
import counterImplementations
# from pysat.card import *

DEFAULT_COUNTER="sequential"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--vertices', '-n', type=int, required=True)
parser.add_argument('--chi', type=int, help="Some properties for being chi critical")
parser.add_argument('--chi2', type=int, help="Some properties for being chi critical but only for vertices haven > 2 labels")
parser.add_argument('--chi3', type=int, help="Some critical subgraph")
parser.add_argument('--labels', '-l', type=int, help="Number of labels", required=True)
parser.add_argument('--differentNeighborHood', '-d',  help="ensure that neighborhood from a vertex subsumes no other neighborhood", action="store_true")
parser.add_argument('--vertexCritical', help="deleting any vertex results in a chi - 1 colorable graph", action="store_true")
parser.add_argument('--primary', choices=["graph", "hypergraph"], help="choose which object SMS should operate on: the original linear hypergraph, or the intersection graph derived from it", default="graph")
parser.add_argument('--codishSym', action="store_true", help="Swapping any two consequent rows (or columns) doesn't minimize the matrix")
parser.add_argument('--hindman', action="store_true", help="At least 11 vertices are in a non-graph edge")
parser.add_argument('--maxClosedNeighborhood', type=int, help="The maximum closed neighborhood of a vertex in the hypergraph")
parser.add_argument('--mtfD', action='store_true', help="Each pair of label occures together or same label has maximum neighborhood")
parser.add_argument('--deactivateCovering',  help="Not every pair of labels must occur in a vertex", action="store_true", default=False)

args = parser.parse_args()

print("c\targs:",args)

# we are looking for a linear hypergraph with
#   l  vertices
# whose intersection graph has
#   n  vertices
# and so the hypergraph has
#   n  hyperedges
#
# If the intersection graph is the primary object, we need
#   c(n, 2) variables for the intersection edges, and
#   l * n variables for the hypergraph incidence matrix
# and the intersection edges must come first for SMS.
#
# If the hypergraph is the primary object, we need
#   c(n+l, 2) variables for the hypergraph incidence matrix
#     including dummy vertex-vertex and hyperedge-hyperedge variables
#   c(n, 2) variables for the intersection edges
# and the hypergraph incidence variables must come first

n = args.vertices
V = range(n)
L = range(args.labels)
N = range(n + args.labels)

class IDPool:
	def __init__(self, start_from = 0) -> None:
		self.start_from = start_from
	start_from = 1

	def id(self):
		x = self.start_from
		self.start_from += 1
		return x

constraints = []
vpool = IDPool(start_from=1)

if args.primary == "graph":
    edge_vars = {(u, v) : vpool.id() for u, v in combinations(V,2)}
    labels = [[vpool.id() for l in L] for v in V]
    def has_label(v, l):
        return labels[v][l]
elif args.primary == "hypergraph":
    labels = {(x, y) : vpool.id() for x, y in combinations(N, 2)}
    def has_label(v, l):
        return labels[(l, v+args.labels)]
    # enforce bipartition N = L, V
    constraints += [[-labels[(l1, l2)]] for l1, l2 in combinations(L, 2)]
    constraints += [[-labels[(v1, v2)]] for v1, v2 in combinations(range(args.labels, args.labels + n), 2)]
    edge_vars = {(u, v) : vpool.id() for u, v in combinations(V,2)}

def var_edge(u, v):
    return edge_vars[(min(u,v), max(u,v))]


# ----------------------------- start encoding ----------------------------------------------------
def CNF_OR(ins, out):
    return [[-out] + ins] + [[out, -x] for x in ins]

def CNF_AND(ins, out):
    return [[out] + [-x for x in ins]] + [[-out, x] for x in ins]

# adjacent implies one with same label
common_label = [[[vpool.id() if v < u else None for l in L] for u in V] for v in V] # common_label[v][u][l] true if l is a label of both
def common_label_func(u,v,l):
    return common_label[min(u,v)][max(u,v)][l]

for u,v in combinations(V,2):
    for l in L:
        clauses = CNF_AND([has_label(u, l), has_label(v, l)], common_label[u][v][l])
        constraints.extend(clauses)
    
    constraints.append([-var_edge(u,v)] + [+common_label[u][v][l] for l in L]) # adjacent implies a common lable
    constraints.extend([[+var_edge(u, v), -common_label[u][v][l]] for l in L]) # and vice versa
    for l1, l2 in combinations(L,2):
        constraints.append([-common_label[u][v][l1], -common_label[u][v][l2]]) # ensures linearity, i.e., only one common label

# minimum degree of a label 2
for l in L:
    counterImplementations.counterFunction([has_label(v, l) for v in V], 2, vpool, constraints, atLeast=2, type=DEFAULT_COUNTER)

# minimum size of a hyperedge 2 = number of labels of a vertex
for v in V:
    counterImplementations.counterFunction([has_label(v, l) for l in L], 2, vpool, constraints, atLeast=2, type=DEFAULT_COUNTER)

# covering
# we can only assume this if we're not assuming vertex-criticality or different neighborhoods
common_vertex = [[[vpool.id() if l1 < l2 else None for v in V] for l2 in L] for l1 in L] # common_vertex[l1][l2][v] true if v is a labeled with both
def common_vertex_func(l1,l2,v):
    return common_vertex[min(l1,l2)][max(l1,l2)][v]

for l1,l2 in combinations(L, 2):
    for v in V:
        clauses = CNF_AND([has_label(v, l1), has_label(v, l2) ], common_vertex_func(l1,l2,v))
        constraints.extend(clauses)

if not args.vertexCritical and not args.differentNeighborHood and not args.chi and not args.deactivateCovering:
    for l1,l2 in combinations(L, 2):
        # some v has both labels   
        constraints.append([common_vertex_func(l1,l2,v) for v in V])

# Use hindman theorem stating that there must be at least 11 distinct vertices in hypereges with size > 2
if args.hindman:
    # at least eleven labels which occur in a large edge
    selected = [vpool.id() for _ in L]
    selectedWitness = [[vpool.id() for _ in V] for _ in L] # mark the large hyperedge with the label
    largeHyperEdge = [vpool.id() for _ in V] # at least 3 labels
    counterImplementations.counterFunction(selected, 11, vpool, constraints, atLeast=11, type=DEFAULT_COUNTER)
    for l in L:
        constraints.append([-selected[l]] + [selectedWitness[l][v] for v in V])
        for v in V:
            constraints.append([-selectedWitness[l][v], largeHyperEdge[v]])
            constraints.append([-selectedWitness[l][v], has_label(v,l)])
    for v in V:
        counter_vars = counterImplementations.counterFunction([has_label(v,l) for l in L], 3, vpool, constraints, type=DEFAULT_COUNTER)
        constraints.append([-largeHyperEdge[v], counter_vars[3-1]]) # large edge implies at least 3 labels

# ensure an upperbound un the closed neighborhood of vertices in the hypergraph.
if args.maxClosedNeighborhood:
    counter_vars_neighborhood = []

    neighboring_labels = [[vpool.id() if l1 < l2 else None for l2 in L] for l1 in L]
    def neighboring_labels_func(l1,l2):
        return neighboring_labels[min(l1,l2)][max(l1,l2)]
    
    for l1, l2 in combinations(L,2):
        clauses = CNF_OR([common_vertex_func(l1,l2, v) for v in V],  neighboring_labels_func(l1,l2))
        constraints.extend(clauses)

    if args.maxClosedNeighborhood > 1:
        # the normal case, limit the size of the neighborhood
        counter_range = args.maxClosedNeighborhood - 1
        upper_bound = counter_range
    else:
        # the combined case: allow any max closed neighborhood, but make sure edges cannot be added without raising it
        # and also request that the graph is colorable with fewer colors (this is to get non-extremal examples)
        counter_range = args.labels - 1
        upper_bound = None
    
    for l in L:
        # counter over the open neighborhood
        counter_vars = counterImplementations.counterFunction([ neighboring_labels_func(l,l2) for l2 in L if l2 != l], counter_range, vpool, constraints, atMost=upper_bound, type=DEFAULT_COUNTER)

        if args.maxClosedNeighborhood > 1:
            counter_vars_neighborhood.append(counter_vars[upper_bound - 1]) # check if contains exactly the maximum neighborhood
        else:
            counter_vars_neighborhood.append(counter_vars) # check if contains exactly the maximum neighborhood
        
    if args.maxClosedNeighborhood > 1:
        constraints.append(counter_vars_neighborhood)
        if args.mtfD:
            for l1,l2 in combinations(L, 2):            
                constraints.append([neighboring_labels[l1][l2], counter_vars_neighborhood[l1], counter_vars_neighborhood[l2]])
    else:
        # for each pair
        # the edge can be added if each endpoint has neighborhood size at most k-1
        # and there is another vertex that has neighborhood size at least k
        for l1,l2 in combinations(L, 2):
            for k in range(1, args.labels):
                for l in L:
                    if l == l1 or l == l2:
                        continue
                    constraints.append([neighboring_labels[l1][l2], counter_vars_neighborhood[l1][k-1], counter_vars_neighborhood[l2][k-1], -counter_vars_neighborhood[l][k-1]])
        #constraints.append(counter_vars_neighborhood) # at least one vertex with highest possible neighborhood

        color = [[vpool.id() for c in L] for v in V]
        for v in V:
            constraints.append(color[v])
        for u, v in combinations(V, 2):
            for c in L:
                constraints.append([-var_edge(u, v), -color[u][c], -color[v][c]])
        
        for v in V:
            constraints.append([-color[v][args.labels-1]])
            for c in range(1, args.labels):
                constraints.append([+counter_vars_neighborhood[l][c-1] for l in L] + [-color[v][c-1]])



# --------------------------------------arguments related to the chromatic number--------------------------------------
if args.chi:
    for i in V:
        counterImplementations.counterFunction([var_edge(i,j) for j in V if j != i], countUpto=args.chi - 1, vPool=vpool,
                    clauses=constraints, atLeast=args.chi - 1, type=DEFAULT_COUNTER)

if args.chi2:
    for i in V:
        counter_intersections = counterImplementations.counterFunction([var_edge(i,j) for j in V if j != i], countUpto=args.chi2, vPool=vpool,
                    clauses=constraints, type=DEFAULT_COUNTER)
        counter_labels = counterImplementations.counterFunction([has_label(i,l) for l in L], countUpto=3, vPool=vpool,
                    clauses=constraints, type=DEFAULT_COUNTER)
        # if at least 3 labels then degree at least chi - 1
        constraints.append([-counter_labels[3-1], +counter_intersections[args.chi2 - 1 - 1]])

if args.chi3:
    # select a subgraph which is critical
    selected = [vpool.id() for _ in V]
    counterImplementations.counterFunction(selected, countUpto=args.chi3, vPool=vpool,
                    clauses=constraints, atLeast=args.chi3, type=DEFAULT_COUNTER) # at least as many vertices as colors
    for i in V:
        counter_labels = counterImplementations.counterFunction([has_label(i,l) for l in L], countUpto=3, vPool=vpool,
                        clauses=constraints, type=DEFAULT_COUNTER)
        # if at least 3 labels then degree then selected
        constraints.append([-counter_labels[3-1], +selected[i]]) # if >= 3 labels then definitely selected.

    for i in V:
        adjacent_and_selected = []
        for j in V:
            if j == i:
                continue
            x = vpool.id()
            clauses = CNF_AND([var_edge(i,j), selected[j]], x)
            constraints.extend(clauses)
            adjacent_and_selected.append(x)
        counter_intersections = counterImplementations.counterFunction(adjacent_and_selected, countUpto=args.chi3, vPool=vpool,
                    clauses=constraints, type=DEFAULT_COUNTER)
        # if at least 3 labels then degree at least chi - 1
        constraints.append([-selected[i], +counter_intersections[args.chi3 - 1 - 1]])

# ------------------------------------not used currently------------------------------------------------------------------

symBreak = True
if args.primary == "hypergraph":
    symBreak = False

if symBreak:
    if True:
        # swapping any two consequent labels doesn't lead to a lexicographically smaller label
        for l in range(args.labels - 1):
            # check that lexicographically ordered

            allPreviousEqual = vpool.id()
            constraints.append([+allPreviousEqual])
            for v in V:
                constraints.append([-allPreviousEqual] + [-has_label(v, l), +has_label(v, l + 1)])

                allPreviousEqualNew = vpool.id()
                constraints.append([-allPreviousEqual, -has_label(v, l), +allPreviousEqualNew])
                constraints.append([-allPreviousEqual, +has_label(v, l + 1), +allPreviousEqualNew])
                allPreviousEqual = allPreviousEqualNew
    else:
        # Basic symmetry breaking for labels by lexicographic ordering
        # always us smallest labeling
        for u in V:
            for l in L:
                for l2 in L:
                    if l2 > l:
                        continue
                    # smaller label used in current hyperedge or previous one
                    constraints.append([-has_label(u, l), +has_label(u, l2)] + [+has_label(v, l2) for v in V if v < u])

if args.codishSym:
    for l in range(args.labels - 1):
        # check that lexicographically ordered

        allPreviousEqual = vpool.id()
        constraints.append([+allPreviousEqual])
        for v in V:
            constraints.append([-allPreviousEqual] + [-has_label(v, l), +has_label(v, l + 1)])

            allPreviousEqualNew = vpool.id()
            constraints.append([-allPreviousEqual, -has_label(v, l), +allPreviousEqualNew])
            constraints.append([-allPreviousEqual, +has_label(v, l + 1), +allPreviousEqualNew])
            allPreviousEqual = allPreviousEqualNew

    for v in V:
        if v + 1 not in V:
            continue
        allPreviousEqual = vpool.id()
        constraints.append([+allPreviousEqual])
        for l in L:
            constraints.append([-allPreviousEqual] + [-has_label(v, l), +has_label(v + 1, l)])

            allPreviousEqualNew = vpool.id()
            constraints.append([-allPreviousEqual, -has_label(v, l), +allPreviousEqualNew])
            constraints.append([-allPreviousEqual, +has_label(v + 1, l), +allPreviousEqualNew])
            allPreviousEqual = allPreviousEqualNew

if args.differentNeighborHood:
    # different neighborhood
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

if args.vertexCritical:
    # deletion of every vertex is colorable
    nColors = args.chi - 1
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

        #basic symmetry breaking for coloring
        # TODO smaller colors must not be available by previous vertices

#------------------------------------------------endEncoding----------------------------------------------------------------

print("c\tTotal number of constraints:", len(constraints))
print("c\tTotal number of variables:", vpool.id())

print('c\tbegin of CNF')
for c in constraints:
    print (' '.join(str(x) for x in c), 0)
print('c\tend of CNF')










