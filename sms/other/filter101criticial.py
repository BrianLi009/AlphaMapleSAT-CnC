from ast import literal_eval
import sys
from pysat.solvers import Solver
from pysat.formula import CNF, IDPool
from itertools import *

vpool = IDPool(start_from=1)

n = int(sys.argv[1])
V = range(n)

edgeVars = [[None for _ in V] for _ in V]
triangleVars =  [[[None for _ in V] for _ in V] for _ in V]
for i,j in combinations(V,2):
    edgeVars[i][j] = edgeVars[j][i] = vpool.id()

for i,j,k in combinations(V,3):
    triangleVars[i][j][k] = vpool.id()
    for i2, j2, k2 in permutations([i,j,k], 3):
        triangleVars[i2][j2][k2]  = triangleVars[i][j][k]

colorVars = [[[None for _ in V ] for _ in V] for _ in V] # for edge potential edge we have a coloring colorVars[i][j][v] is the coloring of vertex v in the graph G - ij

for i,j in combinations(V,2):
    for v in V:
        colorVars[i][j][v] = vpool.id()


constraints = []

for i,j in combinations(V,2):
    # if edge ij is present then no 0-chromatic triangle and 1-chromatic edge in G-ij
    for i2,j2 in combinations(V,2):
        if [i2, j2] == [i,j]:
            continue
        constraints.append([-edgeVars[i][j], -edgeVars[i2][j2], -colorVars[i][j][i2], -colorVars[i][j][j2]])
    for i2,j2,k2 in combinations(V,3):
        if [i,j] in [[i2,j2], [i2,k2], [j2,k2]]:
            continue
        constraints.append([-edgeVars[i][j]] + [colorVars[i][j][x] for x in [i2,j2,k2]] + [-edgeVars[x][y] for x,y in combinations([i2,j2,k2], 2)])

solver = Solver(name='cadical', bootstrap_with=constraints, use_timer=True)

critical = 0
nonCritical = 0
for line in open(sys.argv[2], "r"):
    G = literal_eval(line)
    res = solver.solve(assumptions=[edgeVars[i][j] for i,j in G]) # solve while assuming this graph
    # print("Results from solving:", res)
    if res:
        critical += 1
        print(G)
    else:
        nonCritical +=1

    # if (critical + nonCritical) % 1000 == 0: 
    #    print(critical, "vs", nonCritical)
        