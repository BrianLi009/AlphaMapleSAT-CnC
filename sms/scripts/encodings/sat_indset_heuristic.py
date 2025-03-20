#!/usr/bin/env python3

import counterImplementations
from itertools import *
from helperFunctions.random_coloring import random_coloring, edgeList2adjList

def CNF_OR(ins, out):
    return [[-out] + ins] + [[out, -x] for x in ins]

def CNF_AND(ins, out):
    return [[out] + [-x for x in ins]] + [[-out, x] for x in ins]

def heuristic_removable_set_not_very_large(V, var_edge, vpool, constraints, minDegreeSubgraph, removeAtMost, maxDegree, maxRounds = 4, counter="sequential", start_from_indset=None, potentiallyIndependent = None, potentiallyRemoveable = None) :
    """
    find removable vertices with the following heuristic
    I = set()
    while progess:
      find the first vertex  v  (with the smallest degree) with no neighbors in I;
        I += {v}
        mark  v  removable
      mark each neighbor of  v  as ineligible for I
      remove edges incident to  v
    
      (some fixpoint of the following)
      while there are vertices of degree < minDegreeSubgraphinDegreeSubgraph: 
          mark them removable
    """
    if potentiallyIndependent == None:
        potentiallyIndependent = V
    if potentiallyRemoveable == None:
        potentiallyRemoveable = V

    max_rounds = maxRounds # removeAtMost+1-(len(start_from_indset) if start_from_indset else 0)
    max_possible_deg =  maxDegree  #TODO what is a more sensible bound?
    vert_in_indset_in_round =   [ [   vpool.id() if v in potentiallyIndependent else None for v in V             ] for runde in range(max_rounds+1)]
    vert_eligible_in_round =    [ [   vpool.id() if v in potentiallyIndependent else None for v in V             ] for runde in range(max_rounds+1)]
    vert_removable_in_round =   [ [   vpool.id() if v in potentiallyRemoveable else None for v in V              ] for runde in range(max_rounds+1)]
    edge_into_indset_in_round = [ [ [ vpool.id() for _ in V ] for _ in V ] for runde in range(max_rounds+1)] # TODO 
    vert_remains_and_is_neighbor =  [ [ [ vpool.id() for _ in V ] for _ in V ] for runde in range(max_rounds+1)] # TODO
    degree_in_round = [ [ counterImplementations.counterFunction(
        [vert_remains_and_is_neighbor[runde][v][w] for w in V if v != w],
        max_possible_deg, vpool, constraints) for v in V ] for runde in range(max_rounds+1) ]
    inelig_or_degree_at_least = [ [ [ vpool.id() for deg in range(max_possible_deg+1) ] for _ in V ] for runde in range(max_rounds+1) ]
    #vert_has_mindeg_in_round = [ [ vpool.id() for _ in V ] for runde in range(max_rounds+1) ]
    vert_mindeg_elig_in_round = [ [ vpool.id() for _ in V ] for runde in range(max_rounds+1) ]

    for runde in range(max_rounds+1): # using  runde  because  round  is a reserved word # max_rounds+1
        for v in V:
            # if a vertex is in the indset or removable, it is not eligible
            constraints.append([-vert_in_indset_in_round[runde][v], -vert_eligible_in_round[runde][v]])
            constraints.append([-vert_removable_in_round[runde][v], -vert_eligible_in_round[runde][v]])

            # a vertex is eligible if it doesn't have any neighbors in  indset
            # and at the end of the last round was neither removed nor already in  indset
            constraints.append(
                [+edge_into_indset_in_round[runde][v][w] for w in V[v+1: ]] +
                [+edge_into_indset_in_round[runde][w][v] for w in V[   :v]] +
                [+vert_in_indset_in_round[runde][v], +vert_removable_in_round[runde][v], +vert_eligible_in_round[runde][v]])

            for w in V[v+1:]:
                # indset is an independent set (WARNING: if start_from_indset is no independent, this will silently fail with UNSAT)
                constraints.append([-vert_in_indset_in_round[runde][v], -var_edge(v,w), -vert_in_indset_in_round[runde][w]])

                # define edges into indset
                constraints.append([-edge_into_indset_in_round[runde][v][w], +var_edge(v, w)]) 
                constraints.append([-edge_into_indset_in_round[runde][v][w], +vert_in_indset_in_round[runde][v], +vert_in_indset_in_round[runde][w]]) 
                constraints.append([+edge_into_indset_in_round[runde][v][w], -vert_in_indset_in_round[runde][v], -var_edge(v, w)])
                constraints.append([+edge_into_indset_in_round[runde][v][w], -vert_in_indset_in_round[runde][w], -var_edge(v, w)])

            for w in V:
                if v != w:
                    constraints.append([-vert_in_indset_in_round[runde][v], -var_edge(v, w), -vert_eligible_in_round[runde][w]])

                    # define remaining neighbors
                    constraints.append([-vert_remains_and_is_neighbor[runde][v][w], -vert_in_indset_in_round[runde][w]])
                    constraints.append([-vert_remains_and_is_neighbor[runde][v][w], -vert_removable_in_round[runde][w]])
                    constraints.append([-vert_remains_and_is_neighbor[runde][v][w], +var_edge(v, w)])
                    constraints.append([+vert_remains_and_is_neighbor[runde][v][w], -var_edge(v, w),
                                                                               +vert_in_indset_in_round[runde][w],
                                                                               +vert_removable_in_round[runde][w]])

                    constraints.append([-vert_mindeg_elig_in_round[runde][v], vert_eligible_in_round[runde][v]])

                    # if v has mindeg and degree at least deg+1, then w has degree at least deg+1
                    for deg in range(max_possible_deg):
                        constraints.extend(CNF_OR([+degree_in_round[runde][v][deg], -vert_eligible_in_round[runde][v]], inelig_or_degree_at_least[runde][v][deg]))
                        constraints.append([-vert_mindeg_elig_in_round[runde][v], -degree_in_round[runde][v][deg], +inelig_or_degree_at_least[runde][w][deg]])

            # if v has degree <= deg and all other w have degree >= deg, then v has mindeg
            for deg in range(1, max_possible_deg):
                constraints.append([+degree_in_round[runde][v][deg], -vert_eligible_in_round[runde][v]] + [-inelig_or_degree_at_least[runde][w][deg-1] for w in V if v != w] + [vert_mindeg_elig_in_round[runde][v]])

            # we no longer need this counter, because we are counting globally anyway
            #has_future_neighbors = counterImplementations.counterFunction(
            #        [vert_remains_and_is_neighbor[runde][v][w] for w in V if v != w],
            #        minDegreeSubgraph, vpool, constraints, type=counter)
            constraints.append([+vert_removable_in_round[runde][v], +vert_in_indset_in_round[runde][v], +degree_in_round[runde][v][minDegreeSubgraph-1]])
            constraints.append([-vert_removable_in_round[runde][v], -degree_in_round[runde][v][minDegreeSubgraph-1]])
            constraints.append([-vert_removable_in_round[runde][v], -vert_in_indset_in_round[runde][v]])

        if runde > 0:
            for v in V:
                # monotonicity of indset and the set of removable vertices
                constraints.append([-vert_in_indset_in_round[runde-1][v], +vert_in_indset_in_round[runde][v]])
                constraints.append([-vert_removable_in_round[runde-1][v], +vert_removable_in_round[runde][v]])

                # add the first eligible vertex
                #constraints.append([vert_eligible_in_round[runde-1][w] for w in V[:v]] + [-vert_eligible_in_round[runde-1][v], vert_in_indset_in_round[runde][v]])
                # add the first min-degree eligible vertex
                constraints.append([vert_mindeg_elig_in_round[runde-1][w] for w in V[:v]] + [-vert_mindeg_elig_in_round[runde-1][v], vert_in_indset_in_round[runde][v]])
                # add precisely the first mindeg-eligible vertex
                #constraints.append([-vert_in_indset_in_round[runde][v], vert_eligible_in_round[runde-1][v], vert_in_indset_in_round[runde-1][v]])
                constraints.append([-vert_in_indset_in_round[runde][v], vert_in_indset_in_round[runde-1][v], vert_mindeg_elig_in_round[runde-1][v]])

                for w in V[:v]:
                    #constraints.append([-vert_in_indset_in_round[runde][v], -vert_eligible_in_round[runde-1][v], -vert_eligible_in_round[runde-1][w]])
                    constraints.append([-vert_in_indset_in_round[runde][v], vert_in_indset_in_round[runde-1][v], -vert_mindeg_elig_in_round[runde-1][w]])
        else:
            for v in V:
                if start_from_indset and v in start_from_indset:
                    constraints.append([vert_in_indset_in_round[runde][v]])
                else:
                    constraints.append([-vert_in_indset_in_round[runde][v]])

    counterImplementations.counterFunction(vert_in_indset_in_round[max_rounds] + vert_removable_in_round[max_rounds], removeAtMost, vpool, constraints, atMost=removeAtMost, type=counter)
    return vert_in_indset_in_round, vert_removable_in_round, vert_eligible_in_round


def extend_set_of_removeable_vertices_wrapper(V, var_edge_function, vpool, constraints, minDegreeSubgraph, counter="sequential", start_from_indset=[], removeAtMost=None):
    var_edge = [[None for _ in V] for _ in V]
    for i,j in combinations(V,2):
        var_edge[j][i] = var_edge[i][j] = var_edge_function(i,j)
    return extend_set_of_removeable_vertices(V, var_edge, vpool, constraints, minDegreeSubgraph, counter, start_from_indset, removeAtMost)

# extends the independent set to a set of vertices which are removeable; returns the removeable variables so can also be intialized over variables if not known beforehand
def extend_set_of_removeable_vertices(V, var_edge, vpool, constraints, minDegreeSubgraph, counter="sequential", start_from_indset=[], removeAtMost=None, neighborhood = False):
    var_removeable = [ vpool.id() for _ in V]
    for v in start_from_indset:
        constraints.append([var_removeable[v]])
    
    var_edges_afterRemoval = [[None for _ in V] for _ in V]
    for a,b in combinations(V,2):
        var_edges_afterRemoval[b][a] = var_edges_afterRemoval[a][b] = vpool.id()
        constraints.append([+var_edge[a][b], -var_edges_afterRemoval[a][b]]) # no edge implies no edge after removal
        constraints.append([-var_removeable[a], -var_edges_afterRemoval[a][b]]) # if one of the vertices is removed than edge not present.
        constraints.append([-var_removeable[b], -var_edges_afterRemoval[a][b]])
    
    for j in V:
        if j in start_from_indset: continue

        asdf = counterImplementations.counterFunction([var_edges_afterRemoval[a][j] for a in V if a != j], minDegreeSubgraph, 
            vpool, constraints, type=counter)
        constraints.append([+var_removeable[j], +asdf[minDegreeSubgraph - 1]]) #if not removable then min degree

    # return counter variables
    if removeAtMost:
        additionallyRemoveAtMost = removeAtMost - len(start_from_indset)
        counter_removeAble = counterImplementations.counterFunction([var_removeable[i] for i in set(V) - set(start_from_indset)], additionallyRemoveAtMost, vpool, constraints, atMost=additionallyRemoveAtMost, type=counter)

        if False: # finding unique one
            constraints.append([-counter_removeAble[additionallyRemoveAtMost - 1]])
            '''
            numberOfEdges = 10
            asdf = counterImplementations.counterFunction([var_edges_afterRemoval[v][u] for v,u in combinations(V,2) if v not in start_from_indset and u not in start_from_indset], 
                numberOfEdges, vpool, constraints, type=counter)
            constraints.append([-counter_removeAble[additionallyRemoveAtMost - 1], +asdf[numberOfEdges - 1]]) '''


    if False:
        greedy_coloring_clauses_removeAble(V, var_edge, vpool, var_removeable, constraints, nColors = 3)
    if neighborhood:
        for i,j in permutations(V,2):
            if i in start_from_indset or j in start_from_indset: # since subgraph is vertex critical this doesn't have to be checked
                continue
            # There must be a vertex adjecent to i which is not adjacent to j
            adjacentOnlyToI = []
            for k in V:
                if k == i or k == j: continue
                kIsAdjacentOnlyToI = vpool.id()
                constraints.append([+var_edges_afterRemoval[i][k], -kIsAdjacentOnlyToI])
                constraints.append([-var_edges_afterRemoval[j][k], -kIsAdjacentOnlyToI])
                adjacentOnlyToI.append(kIsAdjacentOnlyToI)
            constraints.append([-var_edges_afterRemoval[i][j], +var_removeable[i], +var_removeable[j]] + adjacentOnlyToI)

    if False:
        # remaining graph is 4-critical so 3 coloring if we delete another one
        nColors = 3
        color = [[vpool.id() for _ in range(nColors)] for _ in V]
        firstNotRemovealbe = [vpool.id() for _ in V]
        for i,j in combinations(V,2):
            constraints.append([+var_removeable[i], -firstNotRemovealbe[j]]) # previous one implies not first not removeable one

        for i in V:
            if i in start_from_indset: continue
            constraints.append(color[i] + [firstNotRemovealbe[i], var_removeable[i]]) # must have a color or fist non removeable or removed
        # valid coloring
        for c in range(nColors):
            for i,j in combinations(V,2):
                if i in start_from_indset or j in start_from_indset: continue
                constraints.append([-var_edge[i][j], -color[i][c], -color[j][c]])

        # first non removeAble must be adjacent to each color
        pickVertex = [[vpool.id() for _ in range(nColors)] for _ in V] # pick vertex for each color
        for c in range(nColors):
            constraints.append([ pickVertex[i][c] for i in V])

            for i in V:
                constraints.append([-var_removeable[i], -pickVertex[i][c]]) # if  removed than it cannot be picked
                constraints.append([+color[i][c], -pickVertex[i][c]]) # if not color c then it cannot be picked

            for i,j in permutations(V,2):
                constraints.append([-firstNotRemovealbe[i], -pickVertex[j][c], var_edge[i][j]]) # if picked then it must be the one with

    if False:
        # four coloring of subgraph can be expanded to 5 coloring of the whole graph
        nColors = 5
        color = [[vpool.id() for _ in range(nColors)] for _ in V]

        for i in V:
            constraints.append(color[i]) # must have a color
            constraints.append([+var_removeable[i], -color[i][-1]]) # not removed means not color 5
            if i in start_from_indset:
                constraints.append([+color[i][-1]]) # last color for the independent set
        # valid coloring
        for c in range(nColors):
            for i,j in combinations(V,2):
                constraints.append([-var_edge[i][j], -color[i][c], -color[j][c]])

        # pick smallest available color for non-removeable vertices

    return var_removeable



# extends the independent set to a set of vertices which are removeable; returns the removeable variables so can also be intialized over variables if not known beforehand
# degree to remaining part gives the degree to the extended graph, i.e., the edges to the other part
def check_removeableForExtendedGraph(V, var_edge, vpool, constraints, degreesToRemainingPart, minDegreeSubgraph, counter="sequential", start_from_indset=[], removeAtMost=None):
    var_removeable = [ vpool.id() for _ in V]
    for v in start_from_indset:
        constraints.append([var_removeable[v]])
    
    var_edges_afterRemoval = [[None for _ in V] for _ in V]
    for a,b in combinations(V,2):
        var_edges_afterRemoval[b][a] = var_edges_afterRemoval[a][b] = vpool.id()
        constraints.append([+var_edge(a,b), -var_edges_afterRemoval[a][b]]) # no edge implies no edge after removal
        constraints.append([-var_removeable[a], -var_edges_afterRemoval[a][b]]) # if one of the vertices is removed than edge not present.
        constraints.append([-var_removeable[b], -var_edges_afterRemoval[a][b]])
    
    for j in V:
        if j in start_from_indset: continue

        asdf = counterImplementations.counterFunction([var_edges_afterRemoval[a][j] for a in V if a != j] + degreesToRemainingPart[j], minDegreeSubgraph, 
            vpool, constraints, type=counter)
        constraints.append([+var_removeable[j], +asdf[minDegreeSubgraph - 2]]) #if not removable then min degree (note that degreesToRemainingPart has x true variables if x + 1 additional edges)

    # return counter variables
    if removeAtMost:
        additionallyRemoveAtMost = removeAtMost - len(start_from_indset)
        counterImplementations.counterFunction([var_removeable[i] for i in set(V) - set(start_from_indset)], additionallyRemoveAtMost, vpool, constraints, atMost=additionallyRemoveAtMost, type=counter)

    return var_removeable



# V must not contain all vertices, and order of V is used for the greedy coloring
# We always work with the indices of the vertex, so if v = V[i] we use i 
def greedy_coloring_clauses_degree(V, var_edge, vpool, constraints, nColors = 3):
    n = len(V)
    vertexOrdering = V # might can
    colors = list(range(nColors)) 

    coloring = [[vpool.id() for _ in colors] for v in range(n)]
    available = [[vpool.id() for _ in colors] for v in range(n)] 
    isColored = [vpool.id() for _ in range(n)]
    adjacentAndColorC = [[[vpool.id() for c in colors] for j in range(n) ] for i in range(n)] # i is adjacent to j and j has color c

    # at least one not colored
    constraints.append([-isColored[i] for i in range(n)])

    # color c implies is colored
    for c in colors:
        for i in range(n):
            constraints.append([-coloring[i][c],  isColored[i]])

    # at most one color
    for c1,c2 in combinations(colors, 2):
        for i in range(n):
            constraints.append([-coloring[i][c1], -coloring[i][c2]])

    # available iff no smaller adjacent vertex has the color; i gives the position
    for i in range(n):
        v = vertexOrdering[i]
        for c in colors:
            constraints.append( [+adjacentAndColorC[i][j][c] for j in range(i) ] + [+available[i][c]] )
            for j in range(i):
                constraints.append([-adjacentAndColorC[i][j][c], -available[i][c]])


    # should get smallest available color
    for i in range(n):
        for c in colors:
            constraints.append([-available[i][c]] + [+available[i][c2] for c2 in range(c)] + [coloring[i][c]])

    # assing adjacentAndColorC to correct truth values
    for i in range(n):
        for j in range(i):
            for c in colors:
                constraints.append([+var_edge(vertexOrdering[i], vertexOrdering[j]), -adjacentAndColorC[i][j][c]])
                constraints.append([+coloring[j][c], -adjacentAndColorC[i][j][c]])
                constraints.append([-var_edge(vertexOrdering[i], vertexOrdering[j]), -coloring[j][c], +adjacentAndColorC[i][j][c]])

# V must not contain all vertices, and order of V is used for the greedy coloring
# We always work with the indices of the vertex, so if v = V[i] we use i 
def greedy_coloring_clauses_removeAble(V, var_edge, vpool, var_removable, constraints, nColors = 3):
    n = len(V)
    vertexOrdering = list(V) # might can
    # import random
    # random.shuffle(vertexOrdering)
    colors = list(range(nColors)) 

    coloring = [[vpool.id() for _ in colors] for v in range(n)]
    available = [[vpool.id() for _ in colors] for v in range(n)] 
    isColored = [vpool.id() for _ in range(n)]
    adjacentAndColorC = [[[vpool.id() for c in colors] for j in range(n) ] for i in range(n)] # i is adjacent to j and j has color c

    # at least one not colored
    constraints.append([-isColored[i] for i in range(n)])

    # color c implies is colored
    for c in colors:
        for i in range(n):
            constraints.append([-coloring[i][c],  isColored[i]])

    for i in range(n):
        constraints.append([-var_removable[i], isColored[i]]) # if removed then can be colored

    # at most one color
    for c1,c2 in combinations(colors, 2):
        for i in range(n):
            constraints.append([-coloring[i][c1], -coloring[i][c2]])

    # available iff no smaller adjacent vertex has the color; i gives the position
    for i in range(n):
        v = vertexOrdering[i]
        for c in colors:
            constraints.append( [+adjacentAndColorC[i][j][c] for j in range(i) ] + [+available[i][c]] )
            for j in range(i):
                constraints.append([-adjacentAndColorC[i][j][c], -available[i][c]])


    # should get smallest available color
    for i in range(n):
        for c in colors:
            constraints.append([-available[i][c]] + [+available[i][c2] for c2 in range(c)] + [coloring[i][c]])

    # assing adjacentAndColorC to correct truth values; not removed
    for i in range(n):
        for j in range(i):
            for c in colors:
                constraints.append([+var_edge[vertexOrdering[i]][vertexOrdering[j]], -adjacentAndColorC[i][j][c]])
                constraints.append([+coloring[j][c], -adjacentAndColorC[i][j][c]])
                constraints.append([-var_removable[j], -adjacentAndColorC[i][j][c]]) # if removed no impact
                constraints.append([-var_edge[vertexOrdering[i]][vertexOrdering[j]], -coloring[j][c], +var_removable[j],  +adjacentAndColorC[i][j][c]])



# Same as before but with propagation. Naming like in documentation
def greedy_coloring_clauses_degree_with_BCP(V, var_edge, vpool, constraints, nColors = 3):
    n = len(V)
    N = range(n)
    C = list(range(nColors)) 

    ord = [[vpool.id() for _ in N] for _ in N] # one vertex smaller then other
    ord3 = [[[vpool.id() for _ in N] for _ in N] for _ in N] # three vertices in order according to ord
    pre = [[vpool.id() for _ in N] for _ in N] # first vertex is direct predecessor of second
    col = [[vpool.id() for _ in C] for _ in N]
    available = [[vpool.id() for _ in C] for _ in N]
    prop = [vpool.id() for _ in N]
    noCol =  [vpool.id() for _ in N]
    witnessCol = [[vpool.id() for _ in C] for _ in N]
    adjSmallerAndCol = [[[vpool.id() for _ in C] for _ in N] for _ in N] # second vertex adjacent to first, comes before and has given color

    monoChromaticEdge = [[vpool.id() for _ in N] for _ in N]

    # TODO for debugging purpose
    ''' 

    for u,v in combinations(N,2):
        constraints.append([ord[u][v]]) # fix the ordering

    for v in N:
        constraints.append([col[v][0]])
        # constraints.append([col[v][1]])
        for c in C:
            constraints.append([-available[v][c]])
        # constraints.append([-col[v][0]]) '''

    # -----------------------------basic ordering ------------------
    for u,v,w in permutations(N,3):
        constraints.append([-ord[u][v], -ord[v][w], +ord[u][w]]) # transitive
    
    for u,v in combinations(N,2):
        constraints.append([-ord[u][v], -ord[v][u]]) # anti symmetric


    for u,v,w in permutations(N,3):
        clauses = CNF_AND([ord[u][v], ord[v][w]], ord3[u][v][w])
        constraints.extend(clauses)

    for u,v in permutations(N,2):
        clauses = CNF_AND([ord[u][v]] + [-ord3[u][w][v] for w in N if w not in [u, v]], pre[u][v])
        constraints.extend(clauses)

    # TODO fix first elements first color - 1 elements; smaller then all succesors in the vertex list, because nothing can be propagated
    for v in range(nColors):
        for u in range(v + 1, n):
            constraints.append([ord[v][u]])

    # ----------------------------basic coloring ----------------------
    for v in N:
        for c1, c2 in combinations(C, 2):
            constraints.append([-col[v][c1], -col[v][c2]])
    
    for v in N:
        for c in C:
            clauses = CNF_AND([-adjSmallerAndCol[v][u][c] for u in N if u != v], available[v][c])
            constraints.extend(clauses)

    for v in N:
        for c in C:
            for u in N:
                if u == v: continue
                clauses = CNF_AND([col[u][c], var_edge(V[u], V[v]), ord[u][v]], adjSmallerAndCol[v][u][c])
                constraints.extend(clauses)

    for v in N:
        for c in C:
            constraints.append([-available[v][c]] + [+available[v][c2] for c2 in C if c2 < c] + [+col[v][c]]) # give smallest available color

    for v in N:
        clauses = CNF_AND([-available[v][c] for c in C], noCol[v])
        constraints.extend(clauses)

    for v in N:
        helperVariables = [vpool.id() for _ in C]
        for c in C:
            clauses = CNF_AND([-available[v][c2] for c2 in C if c2 != c], helperVariables[c])
            constraints.extend(clauses)
        clauses = CNF_OR(helperVariables, prop[v])
        constraints.extend(clauses)

    for v,u in combinations(N, 2):
        constraints.append([+var_edge(V[v], V[u]), -monoChromaticEdge[v][u]])
        for c in C:
            constraints.append([+col[v][c], -col[u][c], -monoChromaticEdge[v][u]])

    constraints.append([monoChromaticEdge[v][u] for v,u in combinations(N,2)] + [noCol[v] for v in N]) # monochromatic edge or no color

    # -------------------refinement of ordering

    for u,v in combinations(N,2):
        constraints.append([+prop[v], ord[v][u], ord[u][v]])
        constraints.append([+prop[u], ord[v][u], ord[u][v]])

    for u,v in combinations(N,2):
        assert(v > u)
        constraints.append([+ord[u][v], prop[v], noCol[v]]) # if not in order then propagating or no color

    for v in N:
        for c in C:
            constraints.append([-available[v][c], -witnessCol[v][c]])

    for v in N:
        for u in N:
            if u == v: continue
            for c in C:
                constraints.append([-ord[u][v], +pre[u][v], -var_edge(V[v], V[u]), -col[u][c], -witnessCol[v][c]])

    for v in N:
        constraints.append([-prop[v]] + witnessCol[v] )

    for v in N:
        for c in C:
            constraints.append([-noCol[v]] + [witnessCol[v][c2] for c2 in C if c2 != c]) # at least two witness colors


def greedy_coloring_clauses_degree_with_BCP_version2(V, var_edge, vpool, constraints, nColors = 3):
    n = len(V)
    N = range(n)
    C = list(range(nColors)) 
    R = list(range(n)) # rounds

    col = [[vpool.id() for _ in C] for _ in N]
    selected = [[vpool.id() for _ in R ] for _ in N] # whether vertex v was selected in a certain round
    selected_before = [[None for _ in R ] for _ in N]
    propagating_round = [vpool.id() for _ in R ]
    unit_vertex = [[vpool.id() for _ in R ] for _ in N] # whether a a vertex is unit or uncolorable in a round.
    unit_vertex_and_not_selected_before = [[vpool.id() for _ in R ] for _ in N]
    unit_vertex_c= [[[vpool.id() for _ in C] for _ in R ] for _ in N] # reason why unit
    available = [[[vpool.id() for _ in C] for _ in R ] for _ in N]
    adj_before_and_color = [[[[vpool.id() for _ in C] for _ in R ]for _ in N ] for _ in N]
    uncolorable = [vpool.id() for _ in N]
    mono_edge = [[vpool.id() if u < v else None for v in N] for u in N]


    # -----------------------clauses---------------------------------
    trueLit = vpool.id()
    constraints.append([trueLit])

    for v in N:
        for r in R:
            if r == 0:
                selected_before[v][0] = -trueLit
            else:
                selected_before[v][r] = selected[v][r - 1]

    for v in N:
        for r in R:
            constraints.append([-selected_before[v][r], selected[v][r]])

    for v in N:
        for c1, c2 in combinations(C, 2):
            constraints.append([-col[v][c1], -col[v][c2]]) 

    for v in N:
        for r in R:
            for c in C:
                clauses = CNF_AND([-adj_before_and_color[v][u][r][c] for u in N if u != v], available[v][r][c])
                constraints.extend(clauses)
    
    for v in N:
        for u in N:
            if u == v: continue
            for r in R:
                for c in C:
                    clauses = CNF_AND([var_edge(V[v], V[u]), selected_before[u][r], col[u][c]], adj_before_and_color[v][u][r][c])
                    constraints.extend(clauses) 

    for v in N:
        for r in R:
            for c in C:
                constraints.append([-selected[v][r], +selected_before[v][r], -available[v][r][c]] + [+available[v][r][c2] for c2 in C if c2 < c] + [+col[v][c]]) # give smallest available color in selected round
                constraints.append([-selected[v][r], +selected_before[v][r], +available[v][r][c], -col[v][c]]) # if selected and not available don't color (TODO: add to documentation)
    # -------------------- choose the vertices which should be selected-----------------------------------

    for v in N:
        for r in R:
            constraints.append([-unit_vertex[v][r], +selected[v][r]])

    for v in N:
        for r in R:
            constraints.append([+propagating_round[r]] + [ -selected_before[u][r] for u in N if u < v ] + [ +selected[v][r]])
   

    for v in N:
        for r in R:
            for u in N:
                if u >= v: continue
                constraints.append([+unit_vertex[v][r], +selected_before[v][r], +selected_before[u][r], -selected[v][r]])

    for v in N:
        for r in R:
            constraints.append([-propagating_round[r], +unit_vertex[v][r], +selected_before[v][r], -selected[v][r]])

    # -----------------remaining clauses--------------------------------
    for v in N:
        clauses = CNF_AND( [-col[v][c] for c in C], uncolorable[v])
        constraints.extend(clauses)

    for v,u in combinations(N, 2):
        constraints.append([+var_edge(V[v], V[u]), -mono_edge[v][u]])
        for c in C:
            constraints.append([+col[v][c], -col[u][c], -mono_edge[v][u]])

    constraints.append([mono_edge[v][u] for v,u in combinations(N,2)] + [uncolorable[v] for v in N]) # monochromatic edge or no color '''
    

    for v in N:
        for r in R:
            for c in C:

                clauses = CNF_AND( [-available[v][r][c2] for c2 in C if c2 != c], unit_vertex_c[v][r][c])
                constraints.extend(clauses)

    for v in N:
        for r in R:

            clauses = CNF_OR( [unit_vertex_c[v][r][c] for c in C], unit_vertex[v][r])
            constraints.extend(clauses)

    for v in N:
        for r in R:

            clauses = CNF_AND([unit_vertex[v][r], -selected_before[v][r]], unit_vertex_and_not_selected_before[v][r])
            constraints.extend(clauses)
    
    for r in R:
        clauses = CNF_OR([unit_vertex_and_not_selected_before[v][r] for v in N], propagating_round[r])
        constraints.extend(clauses) 

    return col, selected, unit_vertex, selected_before

# Allowing start with a set not known before; i.e., there are vertices which are not known before.
# Basically just add to everything var_removeAble[v]
def greedy_coloring_clauses_degree_with_BCP_version3(V, var_edge, vpool, constraints, var_removeAble, nColors = 3, rounds = 15):
    n = len(V)
    N = range(n)
    C = list(range(nColors)) 
    R = list(range(n)) # rounds

    col = [[vpool.id() for _ in C] for _ in N]
    selected = [[vpool.id() for _ in R ] for _ in N] # whether vertex v was selected in a certain round
    selected_before = [[None for _ in R ] for _ in N]
    propagating_round = [vpool.id() for _ in R ]
    unit_vertex = [[vpool.id() for _ in R ] for _ in N] # whether a a vertex is unit or uncolorable in a round.
    unit_vertex_and_not_selected_before = [[vpool.id() for _ in R ] for _ in N]
    unit_vertex_c= [[[vpool.id() for _ in C] for _ in R ] for _ in N] # reason why unit
    available = [[[vpool.id() for _ in C] for _ in R ] for _ in N]
    adj_before_and_color = [[[[vpool.id() for _ in C] for _ in R ]for _ in N ] for _ in N]
    uncolorable = [vpool.id() for _ in N]
    mono_edge = [[vpool.id() if u < v else None for v in N] for u in N]


    # -----------------------clauses---------------------------------
    trueLit = vpool.id()
    constraints.append([trueLit])

    for v in N:
        for r in R:
            if r == 0:
                selected_before[v][0] = -trueLit
            else:
                selected_before[v][r] = selected[v][r - 1]

    for v in N:
        for r in R:
            constraints.append([-selected_before[v][r], selected[v][r]])

    for v in N:
        for c1, c2 in combinations(C, 2):
            constraints.append([-col[v][c1], -col[v][c2]]) 

    for v in N:
        for r in R:
            for c in C:
                clauses = CNF_AND([-adj_before_and_color[v][u][r][c] for u in N if u != v], available[v][r][c])
                constraints.extend(clauses)
    
    for v in N:
        for u in N:
            if u == v: continue
            for r in R:
                for c in C:
                    clauses = CNF_AND([-var_removeAble[v], -var_removeAble[u], var_edge(V[v], V[u]), selected_before[u][r], col[u][c]], adj_before_and_color[v][u][r][c])
                    constraints.extend(clauses) 

    for v in N:
        for r in R:
            for c in C:
                constraints.append([var_removeAble[v],  -selected[v][r], +selected_before[v][r], -available[v][r][c]] + [+available[v][r][c2] for c2 in C if c2 < c] + [+col[v][c]]) # give smallest available color in selected round
                constraints.append([var_removeAble[v], -selected[v][r], +selected_before[v][r], +available[v][r][c], -col[v][c]]) # if selected and not available don't color (TODO: add to documentation)
    # -------------------- choose the vertices which should be selected-----------------------------------

    for v in N:
        for r in R:
            constraints.append([-unit_vertex[v][r], +selected[v][r]])

    for v in N:
        for r in R:
            constraints.append([+propagating_round[r]] + [ -selected_before[u][r] for u in N if u < v ] + [ +selected[v][r]])
   

    for v in N:
        for r in R:
            for u in N:
                if u >= v: continue
                constraints.append([+unit_vertex[v][r], +selected_before[v][r], +selected_before[u][r], -selected[v][r]])

    for v in N:
        for r in R:
            constraints.append([-propagating_round[r], +unit_vertex[v][r], +selected_before[v][r], -selected[v][r]])

    # -----------------remaining clauses--------------------------------
    for v in N:
        clauses = CNF_AND( [-col[v][c] for c in C], uncolorable[v])
        constraints.extend(clauses)

        constraints.append([-var_removeAble[v], col[v][0]]) # just give color zero

    for v,u in combinations(N, 2):
        constraints.append([+var_edge(V[v], V[u]), -mono_edge[v][u]])
        for c in C:
            constraints.append([+col[v][c], -col[u][c], -mono_edge[v][u]])
        constraints.append([-var_removeAble[v], -mono_edge[v][u]])
        constraints.append([-var_removeAble[u], -mono_edge[v][u]])

    constraints.append([mono_edge[v][u] for v,u in combinations(N,2)] + [uncolorable[v] for v in N]) # monochromatic edge or no color '''
    

    for v in N:
        for r in R:
            for c in C:

                clauses = CNF_AND( [-available[v][r][c2] for c2 in C if c2 != c], unit_vertex_c[v][r][c])
                constraints.extend(clauses)

    for v in N:
        for r in R:

            clauses = CNF_OR( [unit_vertex_c[v][r][c] for c in C], unit_vertex[v][r])
            constraints.extend(clauses)

    for v in N:
        for r in R:

            clauses = CNF_AND([unit_vertex[v][r], -selected_before[v][r]], unit_vertex_and_not_selected_before[v][r])
            constraints.extend(clauses)
    
    for r in R:
        clauses = CNF_OR([unit_vertex_and_not_selected_before[v][r] for v in N], propagating_round[r])
        constraints.extend(clauses) 

    return col, selected, unit_vertex, selected_before

predefinedColorings = []

def greedy_coloring_clauses_degree_with_BCP_version_predefinedColors(edgeListSubgraph, VS, VR, VI, var_edge, vpool, constraints, nColors = 4, nColorings = 100):

    V = VR + VI # only have to color the last vertices
    # print("c\t", VR + VI, list(range(16,25)))
    n = len(V)
    N = range(n)
    C = list(range(nColors)) 
    #from sage.all import Graph
    #from sage.graphs.graph_coloring import all_graph_colorings


    #G = Graph(edgeListSubgraph)
    #colorings = list( all_graph_colorings(G, 4))
    import random
    random.seed(13249807)
    #random.shuffle(colorings)
    #sample = colorings[:500]
    #
    #predefinedColorings = [list(c.values()) for c in sample]

    predefinedColorings = []
    R = list(VS)
    adjList = edgeList2adjList(R, edgeListSubgraph)
    assert(R == list(range(len(VS))))
    while len(predefinedColorings) < nColorings:
        coloring = random_coloring(R, adjList, nColors)
        if coloring:
            predefinedColorings.append(coloring)


    C = list(range(nColors))
    for predefinedColoring in predefinedColorings:

        col = [[vpool.id() for _ in C] for _ in N]
        selected = [[vpool.id() for _ in R ] for _ in N] # whether vertex v was selected in a certain round
        selected_before = [[None for _ in R ] for _ in N]
        propagating_round = [vpool.id() for _ in R ]
        unit_vertex = [[vpool.id() for _ in R ] for _ in N] # whether a a vertex is unit or uncolorable in a round.
        unit_vertex_and_not_selected_before = [[vpool.id() for _ in R ] for _ in N]
        unit_vertex_c= [[[vpool.id() for _ in C] for _ in R ] for _ in N] # reason why unit
        available = [[[vpool.id() for _ in C] for _ in R ] for _ in N]
        adj_before_and_color = [[[[vpool.id() for _ in C] for _ in R ]for _ in N ] for _ in N]
        uncolorable = [vpool.id() for _ in N]
        mono_edge = [[vpool.id() if u < v else None for v in N] for u in N]




        # -----------------------clauses---------------------------------
        trueLit = vpool.id()
        constraints.append([trueLit])

        for v in N:
            for r in R:
                if r == 0:
                    selected_before[v][0] = -trueLit
                else:
                    selected_before[v][r] = selected[v][r - 1]

        for v in N:
            for r in R:
                constraints.append([-selected_before[v][r], selected[v][r]])

        for v in N:
            for c1, c2 in combinations(C, 2):
                constraints.append([-col[v][c1], -col[v][c2]]) 

        for v in N:
            for r in R:
                for c in C:
                    clauses = CNF_AND([-adj_before_and_color[v][u][r][c] for u in N if u != v] + [-var_edge(V[v], u) for u in VS if u in predefinedColoring[c]], available[v][r][c]) # only difference
                    constraints.extend(clauses)
        
        for v in N:
            for u in N:
                if u == v: continue
                for r in R:
                    for c in C:
                        clauses = CNF_AND([var_edge(V[v], V[u]), selected_before[u][r], col[u][c]], adj_before_and_color[v][u][r][c])
                        constraints.extend(clauses) 

        for v in N:
            for r in R:
                for c in C:
                    constraints.append([-selected[v][r], +selected_before[v][r], -available[v][r][c]] + [+available[v][r][c2] for c2 in C if c2 < c] + [+col[v][c]]) # give smallest available color in selected round
                    constraints.append([-selected[v][r], +selected_before[v][r], +available[v][r][c], -col[v][c]]) # if selected and not available don't color (TODO: add to documentation)
        # -------------------- choose the vertices which should be selected-----------------------------------

        for v in N:
            for r in R:
                constraints.append([-unit_vertex[v][r], +selected[v][r]])

        for v in N:
            for r in R:
                constraints.append([+propagating_round[r]] + [ -selected_before[u][r] for u in N if u < v ] + [ +selected[v][r]])
    

        for v in N:
            for r in R:
                for u in N:
                    if u >= v: continue
                    constraints.append([+unit_vertex[v][r], +selected_before[v][r], +selected_before[u][r], -selected[v][r]])

        for v in N:
            for r in R:
                constraints.append([-propagating_round[r], +unit_vertex[v][r], +selected_before[v][r], -selected[v][r]])

        # -----------------remaining clauses--------------------------------
        for v in N:
            clauses = CNF_AND( [-col[v][c] for c in C], uncolorable[v])
            constraints.extend(clauses)

        for v,u in combinations(N, 2):
            constraints.append([+var_edge(V[v], V[u]), -mono_edge[v][u]])
            for c in C:
                constraints.append([+col[v][c], -col[u][c], -mono_edge[v][u]])

        constraints.append([mono_edge[v][u] for v,u in combinations(N,2)] + [uncolorable[v] for v in N]) # monochromatic edge or no color '''
        

        for v in N:
            for r in R:
                for c in C:

                    clauses = CNF_AND( [-available[v][r][c2] for c2 in C if c2 != c], unit_vertex_c[v][r][c])
                    constraints.extend(clauses)

        for v in N:
            for r in R:

                clauses = CNF_OR( [unit_vertex_c[v][r][c] for c in C], unit_vertex[v][r])
                constraints.extend(clauses)

        for v in N:
            for r in R:

                clauses = CNF_AND([unit_vertex[v][r], -selected_before[v][r]], unit_vertex_and_not_selected_before[v][r])
                constraints.extend(clauses)
        
        for r in R:
            clauses = CNF_OR([unit_vertex_and_not_selected_before[v][r] for v in N], propagating_round[r])
            constraints.extend(clauses) 

    # return col, selected, unit_vertex, selected_before

# start with some colros 

def greedy_coloring_clauses_degree_with_BCP_version_predefinedColorsVersion2(edgeListSubgraph, VS, VR, VI, var_edge, vpool, constraints, nColors = 4, nColorings = 100, imbalanced = True):

    V = VR + VI # only have to color the last vertices
    # print("c\t", VR + VI, list(range(16,25)))
    n = len(V)
    N = range(n)
    C = list(range(nColors)) 
    #from sage.all import Graph
    #from sage.graphs.graph_coloring import all_graph_colorings


    #G = Graph(edgeListSubgraph)
    #colorings = list( all_graph_colorings(G, 4))
    import random
    random.seed(13249807)
    #random.shuffle(colorings)
    #sample = colorings[:500]
    #
    #predefinedColorings = [list(c.values()) for c in sample]

    predefinedColorings = []
    R = list(VS)
    adjList = edgeList2adjList(R, edgeListSubgraph)
    assert(R == list(range(len(VS))))
    while len(predefinedColorings) < nColorings:
        coloring = []
        if random.randrange(100) % 100 == 0: # take completely random coloring from time to time to get some progress
            coloring = random_coloring(R, adjList, nColors, imbalanced=False)
        else:
                coloring = random_coloring(R, adjList, nColors, imbalanced=imbalanced)
        if coloring:
            predefinedColorings.append(coloring)


    colors = list(range(nColors))
    for predefinedColoring in predefinedColorings:
        vertexOrdering = V
        random.shuffle(vertexOrdering)

        coloring = [[vpool.id() for _ in colors] for v in range(n)]
        available = [[vpool.id() for _ in colors] for v in range(n)] 
        isColored = [vpool.id() for _ in range(n)]
        adjacentAndColorC = [[[vpool.id() for c in colors] for j in range(n) ] for i in range(n)] # i is adjacent to j and j has color c

        # at least one not colored
        constraints.append([-isColored[i] for i in range(n)])

        # color c implies is colored
        for c in colors:
            for i in range(n):
                constraints.append([-coloring[i][c],  isColored[i]])

        # at most one color
        for c1,c2 in combinations(colors, 2):
            for i in range(n):
                constraints.append([-coloring[i][c1], -coloring[i][c2]])

        # available iff no smaller adjacent vertex has the color; i gives the position
        for i in range(n):
            v = vertexOrdering[i]
            for c in colors:
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!only big difference between 
                constraints.append( [+adjacentAndColorC[i][j][c] for j in range(i) ] + [+var_edge(v, u) for u in VS if u in predefinedColoring[c]] + [+available[i][c]] )
                for j in range(i):
                    constraints.append([-adjacentAndColorC[i][j][c], -available[i][c]])
                for u in VS:
                    if u in predefinedColoring[c]:
                        constraints.append([-var_edge(v, u), -available[i][c]])


        # should get smallest available color
        for i in range(n):
            for c in colors:
                constraints.append([-available[i][c]] + [+available[i][c2] for c2 in range(c)] + [coloring[i][c]])

        # assing adjacentAndColorC to correct truth values
        for i in range(n):
            for j in range(i):
                for c in colors:
                    constraints.append([+var_edge(vertexOrdering[i], vertexOrdering[j]), -adjacentAndColorC[i][j][c]])
                    constraints.append([+coloring[j][c], -adjacentAndColorC[i][j][c]])
                    constraints.append([-var_edge(vertexOrdering[i], vertexOrdering[j]), -coloring[j][c], +adjacentAndColorC[i][j][c]])
