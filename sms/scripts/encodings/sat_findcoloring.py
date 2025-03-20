#!/usr/bin/env python3

from pysat.solvers import Cadical
from pysat.formula import CNF, IDPool
from pysat.card import CardEnc

from itertools import combinations, product
from argparse import ArgumentParser
import sys
from ast import literal_eval
from time import perf_counter

parser = ArgumentParser()
parser.add_argument("-n", type=int)
parser.add_argument("-m", type=int)
parser.add_argument("-k", type=int)
parser.add_argument("--pure", "-p", action="store_true", help="only search for pure colorings. Should be easier to solve, but is not fully general")
args = parser.parse_args()

def batch_coloring(n, m):
    vp = IDPool()
    constraints = []
    nc = args.k if args.k else n
    print(nc)
    # hyperedge i |-> color j
    def p(i, j):
        return vp.id((i,j))
    # hyperedges i and j intersect
    def e(i, j):
        return vp.id((i,j,0))

    for i in range(m):
        constraints.append([p(i, k) for k in range(nc)])

    for i, j in combinations(range(m), 2):
        constraints.extend([[-e(i, j), -p(i, k), -p(j, k)] for k in range(nc)])

    return constraints, p, e


def batch_coloring_assumptions(G, n, p, e):
    # find a maximal clique and assign arbitrary colors to it
    B = []
    nc = args.k if args.k else n
    for i in range(n):
        Q = [j for j, e in enumerate(G) if i in e]
        if len(Q) > len(B):
            B = Q
            if len(B) == n-1:
                break

    B = set(B)

    for i, f in enumerate(G):
        if not i in B:
            for j in B:
                if len(f & G[j]) == 0:
                    break
            else:
                B.add(i)

    # will be UNSAT, return contradicting assumptions
    if len(B) > nc:
        return [1, -1]

    return  [+e(i, j) if G[i] & G[j] else -e(i, j) for i, j in combinations(range(len(G)), 2)] + [p(i, k) for k, i in enumerate(B)]

def basic_coloring(G):
    vp = IDPool()
    n = max(max(e) for e in G) + 1
    constraints = []
    def p(i, j):
        return vp.id((i,j))

    for i in range(len(G)):
        constraints.append([p(i, k) for k in range(n)])

    for i, j in combinations(range(len(G)), 2):
        if G[i] & G[j]:
            constraints.extend([[-p(i, k), -p(j, k)] for k in range(n)])

    # find a maximal clique and assign arbitrary colors to it
    B = []
    for i in range(n):
        Q = [j for j, e in enumerate(G) if i in e]
        if len(Q) > len(B):
            B = Q

    for i, e in enumerate(G):
        if not i in B:
            for j in B:
                if len(e & G[j]) == 0:
                    break
            else:
                B.append(i)

    if len(B) > n:
        return [], n, p

    for k, i in enumerate(B):
        constraints.append([p(i, k)])

    return constraints, n, p

def pure_coloring(G):
    vp = IDPool()
    n = max(max(e) for e in G) + 1
    constraints = []
    def p(i, j):
        return vp.id((i,j))

    for i in range(n):
        constraints.append([p(i, j) for j in range(n)])
        for j, k in combinations(range(n), 2):
            constraints.append([-p(i, j), -p(i, k)])

    for k in range(n):
        for i, j in combinations(range(n), 2):
            constraints.append([-p(i, k), -p(j, k)])

    for e in G:
        if len(e) < 3:
            continue
        for f in G:
            if e != f and e & f:
                diffs = [sorted(e - f), sorted(f - e)]
                sums = [[p(D[0], k) for k in range(n)] for D in diffs]
                for x, D in enumerate(diffs):
                    for m in range(1, len(D)):
                        new_sum = [vp.id() for k in range(n)]
                        for i, j in product(range(n), repeat=2):
                            constraints.append([-sums[x][i], -p(D[m], j), new_sum[(i+j) % n]])
                        for i, j in combinations(range(n), 2):
                            constraints.append([-new_sum[i], -new_sum[j]])
                        sums[x] = new_sum
                for i in range(n):
                    constraints.append([-sums[0][i], -sums[1][i]])

    return constraints, n, p

def process_graph_batch(G, solver, n, p, e):
    t0 = perf_counter()
    colorable = solver.solve(assumptions=batch_coloring_assumptions(G, n, p, e))
    nc = args.k if args.k else n
    if colorable:
        print("Y ", end="")
        M = solver.get_model()
        P = {x for x in M if x > 0}
        C = [[] for i in range(nc)]
        for i, e in enumerate(G):
            for j in range(nc):
                if p(i,j) in M:
                    C[j].append(e)
                    break
        print(C, end=" ")
    else:
        print("N ", end="")
        print(G, end= " ")
    t1 = perf_counter()
    print(f"{t1-t0}")
    return colorable
    

def process_graph(G, pure):
    t0 = perf_counter()
    if pure:
        constraints, n, p = pure_coloring(G)
    else:
        constraints, n, p = basic_coloring(G)
    solver = Cadical(bootstrap_with=constraints)
    colorable = solver.solve()
    nc = args.k if args.k else n
    if colorable:
        print("Y ", end="")
        M = solver.get_model()
        P = {x for x in M if x > 0}
        if pure:
            C = [[] for i in range(nc)]
            pi = []
            for i in range(n):
                #print(f"{i:2d}", end="")
                for j in range(nc):
                    if p(i,j) in M:
                        #print(f"{j} ", end="")
                        pi.append(j)
                        break
            #print()
            for i, e in enumerate(G):
                C[sum(pi[x] for x in e) % n].append(e)
            print(C)
        else:
            C = [[] for i in range(nc)]
            for i, e in enumerate(G):
                for j in range(n):
                    if p(i,j) in M:
                        C[j].append(e)
                        break
            print(C, end=" ")
    else:
        print("N ", end="")
        print(G, end= " ")
    t1 = perf_counter()
    print(f"{t1-t0}")
    return colorable
    

constraints, p, e = batch_coloring(args.n, args.m)
solver = Cadical(bootstrap_with=constraints)

n_graphs = 0
n_uncol = 0
t_begin = perf_counter()
for line in sys.stdin:
    n_graphs += 1
    if not process_graph_batch(literal_eval(line), solver, args.n, p, e):
        n_uncol += 1
t_end = perf_counter()
t = t_end - t_begin

print(f"Processed {n_graphs} graphs in {t} seconds")
print(f"Uncolorable graphs: {n_uncol}")
