'''
    Create an encoding testing whether a graph is k-colorable.

    The encoding contains edge variables which must be set as assumption for testing the graph. 
'''

from graphBuilderQCIR import *

class GraphEncodingBuilderNoncolorable(GraphEncodingBuilder):
    def __init__(self, n, directed=False):
        super().__init__(n, directed)

    def addNonKColorable(self, k):
        V = self.V
        var_edge = self.var_edge

        color = [[self.id() for _ in range(k)] for _ in V]

        for c in color:
            self.outputGate.appendToInput(OrGate(self.id(), c))

        for u,v in combinations(V, 2):
            for c in range(k):
                self.outputGate.appendToInput(OrGate(self.id(), [-var_edge(u,v), -color[u][c], -color[v][c]]))

        #W = sorted(V, reverse=True)
        #n = len(W)
        #for i in range(n):
        #    for c, d in combinations(range(k), 2):
        #        self.outputGate.appendToInput(OrGate(self.id(), [-color[W[i]][d]] + [color[W[j]][c] for j in range(i)]))



        return color
        
import sys

n = int(sys.argv[1])
k = int(sys.argv[2])
enc = GraphEncodingBuilderNoncolorable(n) # builder with n vertices.
#permutation = enc.addMinimalityConstrained()
color = enc.addNonKColorable(k)
enc.negateFormula()
enc.print_qcir()
