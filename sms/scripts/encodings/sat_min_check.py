'''
    Create an encoding testing whether a graph is lexicographically minimal or not.

    The encoding contains edge variables which must be set as assumption for testing the graph. 
'''

from graphBuilder import *

class GraphEncodingBuilderMinimality(GraphEncodingBuilder):
    def __init__(self, n, directed=False):
        super().__init__(n, directed)

    def addMinimalityConstrained(self):
        V = self.V
        var_edge = self.var_edge
        id = self.id
        permutation = [[self.id() for _ in V] for _ in V] # permutation[v][u] is true if v is mapped to u
        self.extend(permutation) # each vertex must be mapped to some vertex
        for u in V:
            for v1,v2 in combinations(V, 2):
                self.append([-permutation[u][v1], -permutation[u][v2]])

        edgesPermutation = [[None for _ in V] for _ in V]
        for v1,v2 in combinations(V, 2):
            edgesPermutation[v1][v2] = edgesPermutation[v2][v1] = id()
        def var_edge_perm(v1,v2):
            return edgesPermutation[v1][v2]

        for v1,v2 in combinations(V, 2):
            for v1_perm, v2_perm in permutations(V, 2):
                self.append([-var_edge(v1,v2), -permutation[v1][v1_perm], -permutation[v2][v2_perm],  +var_edge_perm(v1_perm,v2_perm)])
                self.append([+var_edge(v1,v2), -permutation[v1][v1_perm], -permutation[v2][v2_perm],  -var_edge_perm(v1_perm, v2_perm)])

        # find strict indicator pair
        indicatorPair = [[id() if v1 < v2 else None for v2 in V] for v1 in V]
        self.append(list(chain(*[indicatorPair[v][v + 1:] for v in V]))) # at least one indicator pair

        permEqual = [[id() if v1 < v2 else None for v2 in V] for v1 in V] # check if elements A[v1][v2] are equal under the permutation 

        for v1,v2 in combinations(V, 2):
            self.append([+var_edge(v1,v2), -var_edge_perm(v1,v2), -permEqual[v1][v2]])
            self.append([-var_edge(v1,v2), +var_edge_perm(v1,v2), -permEqual[v1][v2]])

        # TODO eventually summerizes some cases with additional variables to minimize the encoding
        for u1,u2 in combinations(V, 2):
            for v1,v2 in combinations(V, 2):
                if v1 > u1 or (v1 == u1 and v2 >= u2):
                    continue
                # vertex pair (v1,v2) is smaller than (u1,u2)
                self.append([-indicatorPair[u1][u2], +permEqual[v1][v2]]) # if u1,u2 is the indicator pair then all previous must be equal


b5 = GraphEncodingBuilderMinimality(5) # builder with 5 vertices.
b5.addMinimalityConstrained()

for clause in b5:
    print(clause)






