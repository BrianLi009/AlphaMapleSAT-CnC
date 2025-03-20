#!/usr/bin/env python3

from pysat.card import CardEnc
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver

import sys
from argparse import ArgumentParser
from itertools import combinations
from math import comb

parser = ArgumentParser()
parser.add_argument("-n", type=int, default=7, help="number of vertices")
parser.add_argument("-k", type=int, default=3, help="degree of uniformity")
parser.add_argument("-m", type=int, default=7, help="number of edges")
parser.add_argument("--generate", action="store_true", help="directly generate hypergraphs as opposed to just outputting the formula")
args = parser.parse_args()

def rank(k_tuple):
    return sum(comb(v, i) for v, i in zip(k_tuple, range(len(k_tuple), 0, -1))) + 1

def unrank(x, k=3):
    t = x-1
    T = []
    for i in range(k, 0, -1):
        y = int(t ** (1/i) * i) if t > 0 else i-1
        c = comb(y, i)
        while c > t:
            y -= 1
            c = comb(y, i)
        T.append(y)
        t -= c
    return tuple(T)

if __name__ == "__main__":
    # encode the existence of a k-uniform hypergraph on n vertices having m hyperedges
    n = args.n
    m = args.m
    k = args.k

    h = comb(n, k)

    vp = IDPool(start_from=h+1)

    F = CardEnc.equals(list(range(1, h+1)), bound=m, vpool=vp)

    # each pair of vertices should be covered by a hyperedge
    for v in range(n):
        for w in range(v+1, n):
            S = list(range(v)) + list(range(v+1, w)) + list(range(w+1, n))
            F.append([rank(sorted(H + (v, w), reverse=True)) for H in combinations(S, k-2)])
        

    if args.generate:
        solver = Solver(name='cadical', bootstrap_with=F.clauses)
        if solver.solve():
            for x in solver.get_model():
                if abs(x) > h:
                    break
                if x > 0:
                    print(unrank(x, k))
        else:
            print("UNSAT")
    else:
        F.to_fp(sys.stdout)

