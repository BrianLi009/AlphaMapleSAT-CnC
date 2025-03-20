#!/usr/bin/python


from itertools import combinations, permutations
from sys import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("n",type=int,help="number of vertices")

parser.add_argument("--chi_upp",type=int,help="chromatic number upper bound")
parser.add_argument("--thickness",type=int,default=0,help="maximum thickness")

parser.add_argument("--DEBUG","-D",type=int,default=1,help="debug level")

parser.add_argument("--loadedges",type=str,help="load edges from file")
parser.add_argument("--justone","-1",action='store_true',help="stop after one solution was found")

args = parser.parse_args()

if args.DEBUG:
	print("c\targs:",args)
vargs = vars(args)
print("c\tactive args:",{x:vargs[x] for x in vargs if vargs[x] != None and vargs[x] != False})

n = args.n
V = range(n)


DEBUG = args.DEBUG

all_variables = []
all_variables += [('edge',(u,v)) for u,v in combinations(V,2)] 

if args.chi_upp:
	all_variables += [('vertex_color',(v,k)) for v in V for k in range(args.chi_upp)]

if args.thickness: # thickness/simple planarity
	all_variables += [('edge_layer',(t,u,v)) for t in range(args.thickness) for u,v in combinations(V,2)]

all_variables_index = {}

_num_vars = 0
for v in all_variables:
    _num_vars += 1
    all_variables_index[v] = _num_vars


from pysat.card import *
vpool = IDPool(start_from=_num_vars+1)  # for auxiliary pysat variables

def var(L):	return all_variables_index[L]
def var_edge(u,v): return var(('edge',(min(u,v), max(u,v))))
def var_edge_layer(t,u,v): return var(('edge_layer',(t,min(u,v),max(u,v))))
def var_vertex_color(*L): return var(('vertex_color',L))


constraints = []


if DEBUG: print("c\tassert undirected edges")
for u,v in permutations(V,2):
	constraints.append([-var_edge(u,v),+var_edge(u,v)])


if args.thickness:
	if DEBUG: print("c\tfor each edge select a layer from 1 to",args.thickness)
	for (u,v) in combinations(V,2):
		constraints.append([-var_edge(u,v)]+[+var_edge_layer(t,u,v) for t in range(args.thickness)])

		for t in range(args.thickness):
			constraints.append([+var_edge(u,v)]+[-var_edge_layer(t,u,v)])

		for t1,t2 in combinations(range(args.thickness),2):
			constraints.append([-var_edge_layer(t1,u,v),-var_edge_layer(t2,u,v)])


if args.loadedges:
	from ast import literal_eval
	edges = literal_eval(open(args.loadedges).readline())
	edges = [(tuple(e)) for e in edges]
	print("c\tload edges and fix:",edges)
	for a,b in edges: assert(a in V and b in V) 
	for a,b in combinations(V,2):
		constraints.append([+var_edge(a,b) if (a,b) in edges else -var_edge(a,b)])


if args.chi_upp:
	if DEBUG: print("c\tassert chromatic number <=",args.chi_upp)
	for v in V:
		constraints.append([+var_vertex_color(v,k) for k in range(args.chi_upp)])
		for k1,k2 in combinations(range(args.chi_upp),2):
			constraints.append([-var_vertex_color(v,k1),-var_vertex_color(v,k2)])
	for u,v in combinations(V,2):
		for k in range(args.chi_upp):
			constraints.append([-var_edge(u,v),-var_vertex_color(u,k),-var_vertex_color(v,k)])

	# break symmetries on colors
	for u in V:
		for k1,k2 in combinations(range(args.chi_upp),2):
			constraints.append([-var_vertex_color(u,k2)]+[+var_vertex_color(v,k1) for v in range(u)])

if DEBUG: 
	print("c\tTotal number of constraints:"+str(len(constraints)))

if DEBUG: 
	print ("c\tuse pysat/Cadical")

from pysat.solvers import Cadical    
solver = Cadical()
for c in constraints: solver.add_clause(c)


import planarity # pip install planarity
import networkx as nx

ct = 0
for sol in solver.enum_models():
	ct += 1
	sol = set(sol)

	valid = True
	if args.thickness:
		for t in range(args.thickness):
			E_t = [(u,v) for (u,v) in combinations(V,2) if var_edge_layer(t,u,v) in sol]
			G_t = nx.Graph(E_t)

			if E_t and not planarity.is_planar(G_t):
				K = planarity.kuratowski_subgraph(G_t)
				solver.add_clause([-var_edge_layer(t,u,v) for (u,v) in K.edges()])
				valid = False
				break
	if not valid: 
		if DEBUG>1:
			print("skip:",ct,end="\r")
			stdout.flush()
		continue

	E = [(u,v) for (u,v) in combinations(V,2) if var_edge(u,v) in sol]
	solver.add_clause([(-var_edge(u,v) if var_edge(u,v) in sol else +var_edge(u,v)) for (u,v) in combinations(V,2)])

	print("s\tsol",ct,"-> edges:",E)

	if args.justone:
		print("s\tstop because args.justone=True")
		break

print("s\ttotal number of solutions:",ct)