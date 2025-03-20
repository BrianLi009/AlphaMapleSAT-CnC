#!/usr/bin/python

from itertools import combinations, permutations
from sys import *
import counterImplementations
# from pysat.card import *

DEFAULT_COUNTER="sequential"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--hyperedges', '-m', type=int, required=True, help="Number of hyperedges")
parser.add_argument('--chi', type=int, help="Some properties for being chi critical")
parser.add_argument('--chi2', type=int, help="Some properties for being chi critical but only for vertices haven > 2 incidence_graph")
parser.add_argument('--chi3', type=int, help="Some critical subgraph in the intersection graph")
parser.add_argument('--vertices', '-n', type=int, help="Number of vertices", required=True)
parser.add_argument('--differentNeighborHood', '-d',  help="ensure that neighborhood from a vertex subsumes no other neighborhood", action="store_true")
parser.add_argument('--vertexCritical', help="deleting any vertex results in a chi - 1 colorable graph", action="store_true")
parser.add_argument('--primary', choices=["graph", "hypergraph"], help="choose which object SMS should operate on: the original linear hypergraph, or the intersection graph derived from it", default="graph")
parser.add_argument('--codishSym', action="store_true", help="Swapping any two consequent rows (or columns) doesn't minimize the matrix")
parser.add_argument('--hindman', action="store_true", help="At least 11 vertices are in a non-graph edge")
parser.add_argument('--maxClosedNeighborhood', type=int, help="The maximum closed neighborhood of a vertex in the hypergraph")
parser.add_argument('--mtfD', action='store_true', help="Each pair of label occures together or same label has maximum neighborhood")
parser.add_argument('--deactivateCovering',  help="Not every pair of incidence_graph must occur in a vertex", action="store_true", default=False)

args = parser.parse_args()

print("c\targs:",args)

# we are looking for a linear hypergraph with
#   n vertices
#   m hyperedges
# and so the intersection graph has
#   m  vertices
#
# If the intersection graph is the primary object, we need
#   c(m, 2) variables for the intersection edges, and
#   m * n variables for the hypergraph incidence matrix
# and the intersection edges must come first for SMS.
#
# If the hypergraph is the primary object, we need
#   c(n+m, 2) variables for the hypergraph incidence matrix
#     including dummy vertex-vertex and hyperedge-hyperedge variables
#   c(m, 2) variables for the intersection edges
# and the hypergraph incidence variables must come first

m = args.hyperedges
E = range(m)
V = range(args.vertices)
N = range(m + args.vertices) # all elements

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
    edge_vars_intersection = {(e1, e2) : vpool.id() for e1, e2 in combinations(E,2)}
    incidence_graph = [[vpool.id() for v in V] for e in E]
    def edge_contains_vertex(e, v):
        return incidence_graph[e][v]
elif args.primary == "hypergraph":
    incidence_graph = {(x, y) : vpool.id() for x, y in combinations(N, 2)}
    def edge_contains_vertex(e, v):
        return incidence_graph[(v, e+args.vertices)]
    # enforce bipartition N = V, E
    constraints += [[-incidence_graph[(v1, v2)]] for v1, v2 in combinations(V, 2)]
    constraints += [[-incidence_graph[(e1, e2)]] for e1, e2 in combinations(range(args.vertices, args.vertices + m), 2)]
    edge_vars_intersection = {(u, v) : vpool.id() for u, v in combinations(E,2)}

# variables to indicate whether an edge is present in the intersection graph
def var_edge_intersection_graph(u, v):
    return edge_vars_intersection[(min(u,v), max(u,v))]


# ----------------------------- start encoding ----------------------------------------------------
def CNF_OR(ins, out):
    return [[-out] + ins] + [[out, -x] for x in ins]

def CNF_AND(ins, out):
    return [[out] + [-x for x in ins]] + [[-out, x] for x in ins]

# adjacent implies one with same label
common_hyperedge = [[[vpool.id() if e1 < e2 else None for v in V] for e2 in E] for e1 in E] # common_hyperedge[e1][e2][v] true if v is in e1 and e2
def common_hyperedge_func(e1,e2,v):
    return common_hyperedge[min(e1,e2)][max(e1,e2)][v]

for e1,e2 in combinations(E,2):
    for v in V:
        clauses = CNF_AND([edge_contains_vertex(e1, v), edge_contains_vertex(e2, v)], common_hyperedge[e1][e2][v])
        constraints.extend(clauses)
    
    constraints.append([-var_edge_intersection_graph(e1,e2)] + [+common_hyperedge[e1][e2][v] for v in V]) # intersecting implies shared vertex
    constraints.extend([[+var_edge_intersection_graph(e1, e2), -common_hyperedge[e1][e2][v]] for v in V]) # and vice versa
    for v1, v2 in combinations(V,2):
        constraints.append([-common_hyperedge[e1][e2][v1], -common_hyperedge[e1][e2][v2]]) # ensures linearity, i.e., only one common vertex

# minimum degree of a vertex 2
for v in V:
    counterImplementations.counterFunction([edge_contains_vertex(e, v) for e in E], 2, vpool, constraints, atLeast=2, type=DEFAULT_COUNTER)

# minimum size of a hyperedge 2
for e2 in E:
    counterImplementations.counterFunction([edge_contains_vertex(e2, v) for v in V], 2, vpool, constraints, atLeast=2, type=DEFAULT_COUNTER)

# covering
# we can only assume this if we're not assuming vertex-criticality or different neighborhoods
common_hyperedge = [[[vpool.id() if v1 < v2 else None for e in E] for v2 in V] for v1 in V] # common_hyperedge[v1][v2][e] true if v1 and v2 are in e
def common_hyperedge_func(v1,v2,e):
    return common_hyperedge[min(v1,v2)][max(v1,v2)][e]

for v1,v2 in combinations(V, 2):
    for e2 in E:
        clauses = CNF_AND([edge_contains_vertex(e2, v1), edge_contains_vertex(e2, v2) ], common_hyperedge_func(v1,v2,e2))
        constraints.extend(clauses)

if not args.vertexCritical and not args.differentNeighborHood and not args.chi and not args.deactivateCovering:
    for v1,v2 in combinations(V, 2):
        # some v has both incidence_graph   
        constraints.append([common_hyperedge_func(v1,v2,v) for v in E])

# Use hindman theorem stating that there must be at least 11 distinct vertices in hypereges with size > 2
if args.hindman:
    # at least eleven incidence_graph which occur in a large edge
    selected = [vpool.id() for _ in V]
    selectedWitness = [[vpool.id() for _ in E] for _ in V] # mark the large hyperedge with the label
    largeHyperEdge = [] # at least 3 incidence_graph
    for e2 in E:
        counter_vars = counterImplementations.counterFunction([edge_contains_vertex(e2,v) for v in V], 3, vpool, constraints, type=DEFAULT_COUNTER)
        largeHyperEdge.append(counter_vars[3-1]) # large edge has at least 3 vertices

    counterImplementations.counterFunction(selected, 11, vpool, constraints, atLeast=11, type=DEFAULT_COUNTER)
    for v in V:
        constraints.append([-selected[v]] + [selectedWitness[v][v] for v in E])
        for e2 in E:
            constraints.append([-selectedWitness[v][e2], largeHyperEdge[e2]])
            constraints.append([-selectedWitness[v][e2], edge_contains_vertex(e2,v)])
    

# ensure an upperbound on the closed neighborhood of vertices in the hypergraph.
if args.maxClosedNeighborhood:
    counter_vars_neighborhood = []

    adjacent_vertices = [[vpool.id() if v1 < v2 else None for v2 in V] for v1 in V] # true if vertices occur in the same edge
    def adjacent_vertices_func(v1,v2):
        return adjacent_vertices[min(v1,v2)][max(v1,v2)]
    
    for v1, v2 in combinations(V,2):
        clauses = CNF_OR([common_hyperedge_func(v1,v2, e) for e in E],  adjacent_vertices_func(v1,v2))
        constraints.extend(clauses)

    if args.maxClosedNeighborhood > 1:
        # the normal case, limit the size of the neighborhood
        counter_range = args.maxClosedNeighborhood - 1
        upper_bound = counter_range
    else:
        # the combined case: allow any max closed neighborhood, but make sure edges cannot be added without raising it
        # and also request that the graph is colorable with fewer colors (this is to get non-extremal examples)
        counter_range = args.vertices - 1
        upper_bound = None
    
    for v in V:
        # counter over the open neighborhood
        counter_vars = counterImplementations.counterFunction([ adjacent_vertices_func(v,v2) for v2 in V if v2 != v], counter_range, vpool, constraints, atMost=upper_bound, type=DEFAULT_COUNTER)

        if args.maxClosedNeighborhood > 1:
            counter_vars_neighborhood.append(counter_vars[upper_bound - 1]) # check if contains exactly the maximum neighborhood
        else:
            counter_vars_neighborhood.append(counter_vars) # check if contains exactly the maximum neighborhood
        
    if args.maxClosedNeighborhood > 1:
        constraints.append(counter_vars_neighborhood)
        if args.mtfD:
            for v1,v2 in combinations(V, 2):            
                constraints.append([adjacent_vertices[v1][v2], counter_vars_neighborhood[v1], counter_vars_neighborhood[v2]])
    else:
        # for each pair
        # the edge can be added if each endpoint has neighborhood size at most k-1
        # and there is another vertex that has neighborhood size at least k
        for v1,v2 in combinations(V, 2):
            for k in range(1, args.vertices):
                for v in V:
                    if v == v1 or v == v2:
                        continue
                    constraints.append([adjacent_vertices[v1][v2], counter_vars_neighborhood[v1][k-1], counter_vars_neighborhood[v2][k-1], -counter_vars_neighborhood[v][k-1]])
        #constraints.append(counter_vars_neighborhood) # at least one vertex with highest possible neighborhood

        color = [[vpool.id() for c in V] for e in E]
        for e2 in E:
            constraints.append(color[e2])
        for e1, e2 in combinations(E, 2):
            for c in V:
                constraints.append([-var_edge_intersection_graph(e1, e2), -color[e1][c], -color[e2][c]])
        
        for e2 in E:
            constraints.append([-color[e2][args.vertices-1]])
            for c in range(1, args.vertices):
                constraints.append([+counter_vars_neighborhood[v][c-1] for v in V] + [-color[e2][c-1]])



# --------------------------------------arguments related to the chromatic number--------------------------------------
if args.chi:
    for i in E:
        counterImplementations.counterFunction([var_edge_intersection_graph(i,j) for j in E if j != i], countUpto=args.chi - 1, vPool=vpool,
                    clauses=constraints, atLeast=args.chi - 1, type=DEFAULT_COUNTER)

if args.chi2:
    for i in E:
        counter_intersections = counterImplementations.counterFunction([var_edge_intersection_graph(i,j) for j in E if j != i], countUpto=args.chi2, vPool=vpool,
                    clauses=constraints, type=DEFAULT_COUNTER)
        counter_incidence_graph = counterImplementations.counterFunction([edge_contains_vertex(i,v) for v in V], countUpto=3, vPool=vpool,
                    clauses=constraints, type=DEFAULT_COUNTER)
        # if at least 3 incidence_graph then degree at least chi - 1
        constraints.append([-counter_incidence_graph[3-1], +counter_intersections[args.chi2 - 1 - 1]])

if args.chi3:
    # select a subgraph which is critical
    selected = [vpool.id() for _ in E]
    counterImplementations.counterFunction(selected, countUpto=args.chi3, vPool=vpool,
                    clauses=constraints, atLeast=args.chi3, type=DEFAULT_COUNTER) # at least as many vertices as colors
    for i in E:
        counter_incidence_graph = counterImplementations.counterFunction([edge_contains_vertex(i,v) for v in V], countUpto=3, vPool=vpool,
                        clauses=constraints, type=DEFAULT_COUNTER)
        # if at least 3 incidence_graph then degree then selected
        constraints.append([-counter_incidence_graph[3-1], +selected[i]]) # if >= 3 incidence_graph then definitely selected.

    for i in E:
        adjacent_and_selected = []
        for j in E:
            if j == i:
                continue
            x = vpool.id()
            clauses = CNF_AND([var_edge_intersection_graph(i,j), selected[j]], x)
            constraints.extend(clauses)
            adjacent_and_selected.append(x)
        counter_intersections = counterImplementations.counterFunction(adjacent_and_selected, countUpto=args.chi3, vPool=vpool,
                    clauses=constraints, type=DEFAULT_COUNTER)
        # if at least 3 incidence_graph then degree at least chi - 1
        constraints.append([-selected[i], +counter_intersections[args.chi3 - 1 - 1]])

# ------------------------------------not used currently------------------------------------------------------------------

symBreak = True
if args.primary == "hypergraph":
    symBreak = False

if symBreak:
    if True:
        # swapping any two consequent incidence_graph doesn't lead to a lexicographically smaller label
        for v in range(args.vertices - 1):
            # check that lexicographically ordered

            allPreviousEqual = vpool.id()
            constraints.append([+allPreviousEqual])
            for e2 in E:
                constraints.append([-allPreviousEqual] + [-edge_contains_vertex(e2, v), +edge_contains_vertex(e2, v + 1)])

                allPreviousEqualNew = vpool.id()
                constraints.append([-allPreviousEqual, -edge_contains_vertex(e2, v), +allPreviousEqualNew])
                constraints.append([-allPreviousEqual, +edge_contains_vertex(e2, v + 1), +allPreviousEqualNew])
                allPreviousEqual = allPreviousEqualNew
    else:
        # Basic symmetry breaking for incidence_graph by lexicographic ordering
        # always us smallest labeling
        for e1 in E:
            for v in V:
                for v2 in V:
                    if v2 > v:
                        continue
                    # smaller label used in current hyperedge or previous one
                    constraints.append([-edge_contains_vertex(e1, v), +edge_contains_vertex(e1, v2)] + [+edge_contains_vertex(e2, v2) for e2 in E if e2 < e1])

if args.codishSym:
    for v in range(args.vertices - 1):
        # check that lexicographically ordered

        allPreviousEqual = vpool.id()
        constraints.append([+allPreviousEqual])
        for e2 in E:
            constraints.append([-allPreviousEqual] + [-edge_contains_vertex(e2, v), +edge_contains_vertex(e2, v + 1)])

            allPreviousEqualNew = vpool.id()
            constraints.append([-allPreviousEqual, -edge_contains_vertex(e2, v), +allPreviousEqualNew])
            constraints.append([-allPreviousEqual, +edge_contains_vertex(e2, v + 1), +allPreviousEqualNew])
            allPreviousEqual = allPreviousEqualNew

    for e2 in E:
        if e2 + 1 not in E:
            continue
        allPreviousEqual = vpool.id()
        constraints.append([+allPreviousEqual])
        for v in V:
            constraints.append([-allPreviousEqual] + [-edge_contains_vertex(e2, v), +edge_contains_vertex(e2 + 1, v)])

            allPreviousEqualNew = vpool.id()
            constraints.append([-allPreviousEqual, -edge_contains_vertex(e2, v), +allPreviousEqualNew])
            constraints.append([-allPreviousEqual, +edge_contains_vertex(e2 + 1, v), +allPreviousEqualNew])
            allPreviousEqual = allPreviousEqualNew

if args.differentNeighborHood:
    # different neighborhood
    for i,j in permutations(E,2):
        # There must be a vertex adjecent to i which is not adjacent to j
        adjacentOnlyToI = []
        for k in E:
            if k == i or k == j: continue
            kIsAdjacentOnlyToI = vpool.id()
            constraints.append([+var_edge_intersection_graph(i,k), -kIsAdjacentOnlyToI])
            constraints.append([-var_edge_intersection_graph(j,k), -kIsAdjacentOnlyToI])
            adjacentOnlyToI.append(kIsAdjacentOnlyToI)
        constraints.append([+var_edge_intersection_graph(i,j)] + adjacentOnlyToI)

if args.vertexCritical:
    # deletion of every vertex is colorable
    nColors = args.chi - 1
    for e2 in E:
        # check if G-v is args.critical - 1 colorable
        colors = [[vpool.id() for _ in E] for _ in range(nColors)]
        # at least one color
        for e1 in E:
            if e1 != e2:
                constraints.append([colors[r][e1] for r in range(nColors)])
        # adjacent once cannot have the same color
        for u1,u2 in combinations(E,2):
            if u1 == e2 or u2 == e2:
                continue
            for r in range(nColors):
                constraints.append([-var_edge_intersection_graph(u1,u2), -colors[r][u1], -colors[r][u2]])

        #basic symmetry breaking for coloring
        # TODO smaller colors must not be available by previous vertices

#------------------------------------------------endEncoding----------------------------------------------------------------

print("c\tTotal number of constraints:", len(constraints))
print("c\tTotal number of variables:", vpool.id())

print('c\tbegin of CNF')
for c in constraints:
    print (' '.join(str(x) for x in c))
print('c\tend of CNF')










