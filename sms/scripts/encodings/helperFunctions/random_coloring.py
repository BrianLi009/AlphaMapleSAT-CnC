#!/usr/bin/env python3

import random

def random_coloring(V, D, k, imbalanced=True):
    
    coloring = {}
    C = set(range(k))

    Vnew = V

    if imbalanced:
        v = random.choice(V)
        coloring[v] = k - 1
        C = set(range(k - 1))
        Vnew = [w for w in V if w != v] # exclude v from all vertices

    random.shuffle(Vnew)

    for v in Vnew:
        A = sorted(C - {coloring.get(w, -1) for w in D[v]})
        if not A:
            return None
        coloring[v] = random.choice(A)
    return [[v for v in V if coloring[v] == c] for c in range(k)]

def edgeList2adjList(V, E):
    return {v : [(x if y == v else y) for (x, y) in E if x == v or y == v] for v in V}


if __name__ == "__main__":
    import sys
    from time import perf_counter
    from argparse import ArgumentParser
    from ast import literal_eval

    parser = ArgumentParser()
    parser.add_argument("--iter", type=int, default=1000)
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("-k", type=int, default=4, help="how many colors")
    args = parser.parse_args()

    for line in sys.stdin:
        E = literal_eval(line.strip().split(";")[0])
        n = max(max(a, b) for (a, b) in E) + 1
        V = list(range(n))
        D = edgeList2adjList(V, E) 

        s = 0
        iterations = 0
        t0 = perf_counter()
        while s < args.target:
            iterations += 1
            s += random_coloring(V, D, args.k) != None
        t1 = perf_counter()
        print(f"{iterations}\t{t1-t0:.3f}")
