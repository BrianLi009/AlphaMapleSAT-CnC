'''
    Create an encoding testing whether a graph is lexicographically minimal or not.

    The encoding contains edge variables which must be set as assumption for testing the graph. 
'''

from graphBuilderQCIR import *

class GraphEncodingBuilderMinimality(GraphEncodingBuilder):
    def __init__(self, n, directed=False):
        super().__init__(n, directed)

    def addMinimalityConstrained(self):

        V = self.V
        var_edge = self.var_edge
        id = self.id
        permutation = [[self.id() for _ in V] for _ in V] # permutation[v][u] is true if v is mapped to u

        # at least mapped to one element
        for p in permutation:
            self.outputGate.appendToInput(OrGate(id(), p))

        for u in V:
            for v1,v2 in combinations(V, 2):
                self.outputGate.appendToInput(OrGate(id(), [-permutation[u][v1], -permutation[u][v2]]))
                self.outputGate.appendToInput(OrGate(id(), [-permutation[v1][u], -permutation[v2][u]]))

        edgesPermutation = [[None for _ in V] for _ in V]
        for v1,v2 in combinations(V, 2):
            edgesPermutation[v1][v2] = edgesPermutation[v2][v1] = OrGate(id(), [])

        def var_edge_perm(v1,v2):
            return edgesPermutation[v1][v2]

        for v1,v2 in combinations(V, 2):
            for v1_perm, v2_perm in permutations(V, 2):
                g = AndGate(id(), [+var_edge(v1,v2), +permutation[v1][v1_perm], +permutation[v2][v2_perm]]) # concrete permutation
                edgesPermutation[v1][v2].appendToInput(g)

        piStable = [[ None for v2 in V] for v1 in V]
        for v1,v2 in combinations(V,2):
            piStable1 = AndGate(id(), [+permutation[v1][v1], +permutation[v2][v2]])
            piStable2 = AndGate(id(), [+permutation[v1][v2], +permutation[v2][v1]])
            piStable[v1][v2] = OrGate(id(), [piStable1, piStable2])

        # find strict indicator pair
        indicatorPair = [[None for v2 in V] for v1 in V]

        permGreaterEqual = [[None for v2 in V] for v1 in V] # check if elements A[v1][v2] are equal under the permutation 
        permGreater = [[None for v2 in V] for v1 in V]

        for v1,v2 in combinations(V, 2):
            permGreater[v1][v2] = AndGate(id(), [+var_edge(v1,v2), -var_edge_perm(v1,v2)])
            permGreaterEqual[v1][v2] = OrGate(id(), [var_edge(v1,v2), -var_edge_perm(v1,v2), piStable[v1][v2]])

        # TODO eventually summerizes some cases with additional variables to minimize the encoding
        for v1,v2 in combinations(V, 2):
            indicatorPair[v1][v2] = AndGate(id(), [+var_edge(v1,v2), -var_edge_perm(v1,v2)] + [permGreaterEqual[u1][u2] for u1,u2 in combinations(V, 2) if u1 < v1 or (u1 == v1 and u2 < v2)])

        oneIndicatorPair = OrGate(id(), list(chain(*[indicatorPair[v][v + 1:] for v in V]))) # only upper triangle matrix
        self.outputGate.appendToInput(oneIndicatorPair)
        return permutation

import sys
        
if __name__ == "__main__":
    n = int(sys.argv[1])
    b5 = GraphEncodingBuilderMinimality(n) # builder with n vertices.
    permutation = b5.addMinimalityConstrained()
    b5.print_qcir()





