#!/usr/bin/env sage -python

from sage.all import Graph
from sage.graphs.independent_sets import IndependentSets
import ast

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("inputFile", type=str, help="the file containing the graphs")
parser.add_argument("outputFile", type=str, help="the file where the non-reduceable graphs should be printed")
parser.add_argument("--g6format", action="store_true", help="By default edge list; if graphs in g6 format use this argument")
parser.add_argument("--minDegreeSubgraph", type=int, default=3)
parser.add_argument("--minSizeIndependentSet", type=int, default=0)
parser.add_argument("--maxDegree", type=int, default=6)
parser.add_argument("--removeAtMost", type=int, default=8)

args = parser.parse_args()

# given a subgraph G, independent set of subgraph, and degree of G, calculate the vertices which will definitely have low degree in the whole graph
def getLowDegreeVertices(G, I, deg, minDegreeSubgraph, maxDegree):
    lowDegreeVertices = []
    additionalRemoveAble = I
    H = G.copy()
    last = []
    while len(additionalRemoveAble) != 0:
        H.delete_vertices(additionalRemoveAble)
        l = list(H.degree_iterator(labels=True))
        last = [(v,(minDegreeSubgraph - d)) for v,d in l]
        additionalRemoveAble = [v for v,d in l if (minDegreeSubgraph - d) + deg[v] > maxDegree] # min minAdditionalEdges = minDegreeSubgraph - d
        lowDegreeVertices = lowDegreeVertices + additionalRemoveAble
    print(last)
    return lowDegreeVertices

def preprocessing(G, removeAtmost, minSizeIndependentSet = 0, minDegreeSubgraph = 3, maxDegree = 6):
    print("new")
    deg = list(range(G.order()))
    for v,d in list(G.degree_iterator(labels=True)):
        deg[v] = d
    if max(deg) > maxDegree - 1: return True # Degree at most \Delta - 1
    for I in IndependentSets(G, maximal=True):
        if len(I) < minSizeIndependentSet: continue
        if len(I) > removeAtmost: return True # print("Done")
        # print(I)

        lowDegreeVertices = getLowDegreeVertices(G, I, deg, minDegreeSubgraph, maxDegree)
        print(len(I), len(lowDegreeVertices))
        
        # print(l)
        #print("Min additional edges", sum([max(1,(minDegreeSubgraph - d)) for v,d in l]))
        # print("Number of removeables", G.order() - H.order(), "Length of independent set", len(I), G.order(), H.order())
        if  len(lowDegreeVertices) + len(I) > removeAtmost:
            #print("Number of removeables", G.order() - H.order(), "Length of independent set", len(I))
            #print("Start with", I)
            return True
    return False

with open(args.outputFile, 'w') as f:
    filterAble = 0
    notFilterAble = 0
    for line in open(args.inputFile, "r"):
        G = None
        if args.g6format:
            G = Graph(line) # g6 string
        else:
            G = Graph(ast.literal_eval(line.split(";")[0]))
        if preprocessing(G, args.removeAtMost, maxDegree = args.maxDegree, minSizeIndependentSet=args.minSizeIndependentSet):
            filterAble += 1
        else:
            print(line.strip(), file=f)
            notFilterAble += 1
        if (filterAble + notFilterAble) % 1000 == 0:
            print(filterAble, "vs", notFilterAble)
            
       
    print(filterAble, "vs", notFilterAble)