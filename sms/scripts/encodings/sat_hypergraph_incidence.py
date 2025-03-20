#!/usr/bin/python

from itertools import *
from sys import *
from time import perf_counter

from numpy import product
from math import ceil
# from pysat.card import *

import counterImplementations
DEFAULT_COUNTER = "sequential"

import argparse
parser = argparse.ArgumentParser()

# -----------------------------basic------------------------------------------
parser.add_argument('--vertices', '-n', type=int, required=True, help="Number of vertices")
parser.add_argument('--hyperedges', '-m', type=int, required=True, help="Number of hyperedges")
parser.add_argument('--uniform', '-k', type=int, required=True, help="Degree of uniformity")
parser.add_argument('--static', '-s', action="store_true", help="Forbid 2-colorings statically (exponential in n)")
parser.add_argument('--matching', '-g', action="store_true", help="Enforce the existence of a vertex-covering matching in the incidence graph")

parser.add_argument('--partition', '-p', type=int, nargs="+", help="paritioning of vertices")
parser.add_argument('--degrees', type=int, nargs="+", help="degree of vertices of the partition")

t_begin = perf_counter()


args = parser.parse_args()

print("c\targs:",args, file=stderr)
vargs = vars(args)
print("c\tactive args:",{x:vargs[x] for x in vargs if vargs[x] != None and vargs[x] != False}, file=stderr)

n = args.vertices
m = args.hyperedges
k = args.uniform
N = n + m

V = range(n)
H = range(n, n+m)

# swapped version performs worse, i.e., if edge come first
# V = range(m, m+n)
# H = range(m)
W = range(n+m)


all_variables = []
all_variables += [('edge',(u,v)) for u,v in combinations(W,2)]
all_variables_index = {}

_num_vars = 0
for v in all_variables:
    _num_vars += 1
    all_variables_index[v] = _num_vars

# print(all_variables_index)

class IDPool:
	def __init__(self, start_from = 0) -> None:
		self.start_from = start_from
	start_from = 1

	def id(self):
		x = self.start_from
		self.start_from += 1
		return x

vpool = IDPool(start_from=_num_vars+1)  # for auxiliary pysat variables

def var(L):	return all_variables_index[L]
def var_edge(u,v): return var(('edge',(min(u,v), max(u,v))))


constraints = []

# bipartite incidence graph
for u, v in combinations(V, 2):
    constraints.append([-var_edge(u,v)])
for e, f in combinations(H, 2):
    constraints.append([-var_edge(e,f)])

# minimum degree 2 or more because covered
delta = max(2, int(ceil((n-1)/(k-1))))
for v in V:
    counterImplementations.counterFunction([+var_edge(v, e) for e in H], delta, vpool, constraints, atLeast=delta, type=DEFAULT_COUNTER)

# k-uniform
for e in H:
    counterImplementations.counterFunction([+var_edge(v, e) for v in V], k, vpool, constraints, atLeast=k, atMost=k, type=DEFAULT_COUNTER)

# all hyperedges different
for epair in combinations(H, 2):
    for i in range(len(epair)):
        not_subsumed = []
        for v in V:
            occurs_once = vpool.id()
            constraints.append([-occurs_once, +var_edge(v, epair[i])])
            constraints.append([-occurs_once, -var_edge(v, epair[1-i])])
            constraints.append([+occurs_once, -var_edge(v, epair[i]), +var_edge(v, epair[1-i])])
            not_subsumed.append(occurs_once)
        constraints.append(not_subsumed)

# for any two vertices the sets of hyperedges where they respectively occur should not subsume one another
for vpair in combinations(V, 2):
    for i in range(len(vpair)):
        not_subsumed = []
        for e in H:
            only_one_occurs = vpool.id()
            constraints.append([-only_one_occurs, +var_edge(vpair[i], e)])
            constraints.append([-only_one_occurs, -var_edge(vpair[1-i], e)])
            constraints.append([+only_one_occurs, -var_edge(vpair[i], e), +var_edge(epair[1-i], v)])
            not_subsumed.append(+only_one_occurs)
        constraints.append(not_subsumed)

# covering
for u, v in combinations(V, 2):
    u_and_v_jointly_occur_somewhere = []
    for e in H:
        u_and_v_both_occur_in_e = vpool.id()
        constraints.append([-u_and_v_both_occur_in_e, +var_edge(u, e)]);
        constraints.append([-u_and_v_both_occur_in_e, +var_edge(v, e)]);
        constraints.append([+u_and_v_both_occur_in_e, -var_edge(u, e), -var_edge(v, e)]);
        u_and_v_jointly_occur_somewhere.append(u_and_v_both_occur_in_e)
    constraints.append(u_and_v_jointly_occur_somewhere)

# not 2-colorable
if args.static:
    for color1 in chain.from_iterable(combinations(V, i) for i in range(1, n // 2 + 1)):
        color2 = [v for v in V if v not in color1]
        exists_monochrom = []
        for e in H:
            monochrom1 = vpool.id()
            monochrom2 = vpool.id()
            constraints.append([+monochrom1] + [+var_edge(v, e) for v in color2])
            constraints.extend([[-monochrom1, -var_edge(v, e)] for v in color2])
            constraints.append([+monochrom2] + [+var_edge(v, e) for v in color1])
            constraints.extend([[-monochrom2, -var_edge(v, e)] for v in color1])
            exists_monochrom += [monochrom1, monochrom2]
        constraints.append(exists_monochrom)


if args.partition:
	pos = min(V) # assumes that vertices are ordered
	for i in range(len(args.partition)):
		posNew = pos + args.partition[i]

		# Each vertex has exactly degree  args.degrees[i]
		for v in range(pos, posNew):
			degree = args.degrees[i]
			counterImplementations.counterFunction([var_edge(v,u) for u in H if u != v], countUpto=degree, vPool=vpool, clauses=constraints, atLeast=degree, atMost=degree, type=DEFAULT_COUNTER)
		pos = posNew

# matching
strict_matching = True
if args.matching:
    M = [{e : vpool.id() for e in H} for v in V] # v matches with e  <->  M[v][e]

    # every vertex matches ...
    for v in V:
        constraints.append(sorted(M[v].values()))
        # ... but only one hyperedge (this is an optional constraint)
        if strict_matching:
            for e, f in combinations(H, 2):
                constraints.append([-M[v][e], -M[v][f]])

    # no edge matches twice (this is not optional)
    for e in H:
        for v, w in combinations(V, 2):
            constraints.append([-M[v][e], -M[w][e]])

       
print("c\tTotal number of constraints:", len(constraints), file=stderr)
print("c\tTotal number of variables:", vpool.id(), file=stderr)



print('c\tstart writting encoding to file', file=stderr)

print("\n".join(" ".join(map(str, C)) for C in constraints)) # performs best
print('c\tencoding finished', file=stderr)


t_tot = perf_counter()
print(f"Encoding time: {perf_counter()-t_begin:.4g}", file=stderr)
