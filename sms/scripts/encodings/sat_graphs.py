#!/usr/bin/python

from counterImplementations import *

from itertools import combinations, permutations
from sys import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("n",type=int,help="number of vertices")
parser.add_argument("--directed","-d",action='store_true',help="directed graph")

parser.add_argument("--planar_kuratowski","--planar","-p",action='store_true',help="use SMS with Kuratowsi planarity encoding")

parser.add_argument("--planar_order","--planarO","-po",action='store_true',help="order based planarity encoding")
parser.add_argument("--thickness",type=int,help="maximum thickness")

parser.add_argument("--planar_universal","--planarU","-pu",action='store_true',help="using universal sets to assert that the graph is planar")

parser.add_argument("--clawfree",action='store_true',help="assert clawfree graph (no K_{1,3})")
parser.add_argument("--linegraph",action='store_true',help="assert linegraph (forbidden induced subgraphs)")

parser.add_argument("--connectivity_low","-c","--kappa_low",default=0,type=int,help="lower bound on connectivity")

parser.add_argument("--num_edges_upp",type=int,help="maximum number of edges")
parser.add_argument("--num_edges_low",type=int,help="minimum number of edges")

parser.add_argument("--chi_upp",type=int,help="chromatic number upper bound")
parser.add_argument("--chi_low",type=int,help="chromatic number lower bound; only supported for SMS")
parser.add_argument("--alpha_upp",type=int,help="maximum size of an anti clique")
parser.add_argument("--omega_upp",type=int,help="maximum size of a clique")
parser.add_argument("--Ckfree",type=int,help="no Ck-subgraph")
parser.add_argument("--Delta_upp",type=int,help="maximum degree")
parser.add_argument("--delta_low",type=int,help="minimum degree")
parser.add_argument("--regularity","-reg",type=int,help="minimum+maximum degree")
parser.add_argument("--evendegrees",action='store_true',help="degrees should be even")

parser.add_argument("--cnf2file","-c2f",type=str,help="write instance to file")
parser.add_argument("--dontsolve",action='store_true',help="do not solve instance")
parser.add_argument("--justone","-1",action='store_true',help="stop after one solution was found")
parser.add_argument("--hideGraphs","-hg",action='store_true',help="do not display graphs")
parser.add_argument("--countertype",choices=['sequential', 'totalizer'], default='sequential',help="type of counter function to use")

parser.add_argument("--DEBUG","-D",type=int,default=1,help="debug level")

parser.add_argument("--solver", choices=['cadical', 'pycosat'], default=None, help="SAT solver") # can be removed
parser.add_argument("--loadgraph",type=str,help="load graph from file")



args = parser.parse_args()


if args.DEBUG:
	print("c\targs:",args)
vargs = vars(args)
print("c\tactive args:",{x:vargs[x] for x in vargs if vargs[x] != None and vargs[x] != False})


if args.planar_order:
	args.thickness = 1
	print("c\t planar_order => setting thickness to 1")

if args.regularity:
	print("c\t setting delta_low and Delta_upp to",args.regularity)
	args.delta_low = args.Delta_upp = args.regularity


n = args.n
V = range(n)


DEBUG = args.DEBUG



all_variables = []
if args.directed:
	# Note: to use SMS/digraphs it is important that these variables in this particular order come first
	all_variables += [('directed_edge',(u,v)) for u,v in permutations(V,2)]

# Note: to use SMS/graphs it is important that these variables in this particular order come first
all_variables += [('edge',(u,v)) for u,v in combinations(V,2)] 

if args.chi_upp:
	all_variables += [('vertex_color',(v,k)) for v in V for k in range(args.chi_upp)]

if args.connectivity_low:
	all_variables += [('reachable',(v,t,I)) for k in range(args.connectivity_low) for I in combinations(sorted(set(V)),k) for v in set(V) - {min(set(V) - set(I))} - set(I) for t in V ] # u can reach v without I in t steps
	all_variables += [('reachable_via',(v,w,t,I)) for k in range(args.connectivity_low) for I in combinations(sorted(set(V)),k) for v in set(V) - {min(set(V) - set(I))} - set(I) for t in V for w in set(V) - {min(set(V) - set(I)),v} - set(I)] # u can reach v via w without I in t steps 

if args.thickness: # thickness/simple planarity
	all_variables += [('edge_layer',(t,u,v)) for t in range(args.thickness) for u,v in combinations(V,2)]
	all_variables += [('u_smaller_v_i',(t,u,v,i)) for t in range(args.thickness) for i in range(3) for u,v in permutations(V,2)] # u < v in i-th linear order 
	all_variables += [('uv_smaller_w_i',(t,u,v,w,i)) for t in range(args.thickness) for i in range(3) for u,v,w in permutations(V,3) if u<v] # u < w and v < w in i-th linear order 

if args.planar_universal:
	P_universal = {
		3:  [(0,0),(1,0),(0,1)],
		4:  [(0,0),(3,0),(0,3),(1,1)],
		5:  [(0,3),(3,3),(2,2),(1,2),(2,0)],
		6:  [(6,0),(0,0),(2,2),(3,2),(5,3),(5,6)],
		7:  [(1,10),(10,2),(9,2),(5,3),(2,8),(1,9),(0,0)],
		8:  [(18,0),(0,0),(2,2),(10,7),(13,5),(16,6),(17,6),(16,18)],
		9:  [(24,15),(5,0),(4,5),(4,14),(7,15),(8,16),(11,16),(13,19),(0,24)],
		10: [(38,0),(0,92),(7,90),(8,88),(10,85),(12,79),(44,8),(55,22),(59,27),(92,66)],
		11: [(214,0),(0,13),(2,16),(9,26),(124,12),(133,11),(148,9),(213,1),(211,4),(210,6),(116,179),(122,197)]
	}
	
	# use minimal universal set or grid if not available
	P = P_universal[n] if n in P_universal else [(x,y) for x in V for y in V if x<y]

	all_variables += [('mapping',(v,p)) for v in V for p in P]
	all_variables += [('segment',(p,q)) for p,q in permutations(P,2)]



all_variables_index = {}

_num_vars = 0
for v in all_variables:
    _num_vars += 1
    all_variables_index[v] = _num_vars


from pysat.card import *
vpool = IDPool(start_from=_num_vars+1)  # for auxiliary pysat variables

def var(L):	return all_variables_index[L]

def var_edge(u,v): return var(('edge',(min(u,v), max(u,v))))
def var_directed_edge(*L): return var(('directed_edge',L))

def var_vertex_color(*L): return var(('vertex_color',L))


def var_reachable(*L): return var(('reachable',L))
def var_reachable_via(*L): return var(('reachable_via',L))

def var_edge_layer(t,u,v): return var(('edge_layer',(t,min(u,v),max(u,v))))
def var_u_smaller_v_i(*L): return var(('u_smaller_v_i',L))
def var_uv_smaller_w_i(*L): return var(('uv_smaller_w_i',L))
def var_uv_smaller_w(*L): return var(('uv_smaller_w',L))
def var_v_maximal_i(*L): return var(('v_maximal_i',L))
def var_v_maximal(*L): return var(('v_maximal',L))
def var_uv_maximal(*L): return var(('uv_maximal',L))
def var_GSconnected(*L): return var(('GSconnected',L))

def var_colored_directed_edge(*L): return var(('colored_directed_edge',L))
def var_directed_edge(*L): return var(('directed_edge',L))

def var_colored_path_length_k(*L): return var(('colored_path_length_k',L))
def var_colored_path_length_k_via_z(*L): return var(('colored_path_length_k_via_z',L))

def var_u_smaller_v_unique_i(*L): return var(('u_smaller_v_unique_i',L))
def var_u_smaller_v_unique(*L): return var(('u_smaller_v_unique',L))
def var_uv_blocked_by_w_in_i(*L): return var(('uv_blocked_by_w_in_i',L))

def var_triangulation_edge(*L): return var(('triangulation_edge',L))
def var_count(*L): return var(('count',L))

def var_path_u_i_without_v(*L): return var(('path_u_i_without_v',L))
def var_u_part_of_path_v(*L): return var(('u_part_of_path_v',L))
def var_u_next_w_in_path_v(*L): return var(('u_next_w_in_path_v',L))


def var_mapping(*L): return var(('mapping',L))
def var_segment(*L): return var(('segment',L))

constraints = []


if DEBUG: print("c\tassert undirected edges")
for u,v in permutations(V,2):
	constraints.append([-var_edge(u,v),+var_edge(v,u)])
	#	print("c\t", [-var_edge(u,v),+var_edge(v,u)])


if args.directed:
	if DEBUG: print("c\tassert directed edges")
	for u,v in combinations(V,2):
		constraints.append([-var_edge(u,v),+var_directed_edge(u,v),+var_directed_edge(v,u)])
		constraints.append([+var_edge(u,v),-var_directed_edge(u,v)])
		constraints.append([+var_edge(u,v),-var_directed_edge(v,u)])


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


if args.connectivity_low:
	if DEBUG: print("c\tassert reachable variables and connectivity")
	assert(n > args.connectivity_low) # an k-connected graph has at least k+1 vertices

	for k in range(args.connectivity_low):
		for I in combinations(sorted(set(V)),k): # remove I and check if still connected
			u = min(set(V)-set(I))
			for v in set(V) - {u} - set(I):
				for t in V:
					if t == 0:
						# reachable in first step if adjacent
						constraints.append([-var_edge(v,u), +var_reachable(v,0,I)])
						constraints.append([+var_edge(v,u), -var_reachable(v,0,I)])

					else:
						constraints.append([-var_reachable(v,t,I),+var_reachable(v,t-1,I)]+[+var_reachable_via(v,w,t,I) for w in set(V)-set(I)-{v,u}])
						constraints.append([+var_reachable(v,t,I),-var_reachable(v,t-1,I)])
						for w in set(V)-set(I)-{v,u}:
							constraints.append([+var_reachable(v,t,I),-var_reachable_via(v,w,t,I)])

							constraints.append([+var_reachable_via(v,w,t,I),-var_reachable(w,t-1,I),-var_edge(w,v)])
							constraints.append([-var_reachable_via(v,w,t,I),+var_reachable(w,t-1,I)])
							constraints.append([-var_reachable_via(v,w,t,I),+var_edge(w,v)])
				#must be reached
				constraints.append([+var_reachable(v,max(V),I)])



if args.thickness:
	if DEBUG: print("c\tassert thickness <=",args.thickness)
	for t in range(args.thickness):
		if DEBUG: print("c\tdefine linear orders")
		for i in range(3):
			# anti-symmetrie + connexity
			for u,v in permutations(V,2):
				constraints.append([+var_u_smaller_v_i(t,u,v,i),+var_u_smaller_v_i(t,v,u,i)])
				constraints.append([-var_u_smaller_v_i(t,u,v,i),-var_u_smaller_v_i(t,v,u,i)])

			# transitivity
			for u,v,w in permutations(V,3):
				constraints.append([-var_u_smaller_v_i(t,u,v,i),-var_u_smaller_v_i(t,v,w,i),+var_u_smaller_v_i(t,u,w,i)])


		if DEBUG: print("c\tassert uv_smaller_w_i variable")
		for i in range(3):
			for u,v,w in permutations(V,3):
				if u<v:
					constraints.append([-var_uv_smaller_w_i(t,u,v,w,i),+var_u_smaller_v_i(t,u,w,i)])
					constraints.append([-var_uv_smaller_w_i(t,u,v,w,i),+var_u_smaller_v_i(t,v,w,i)])
					constraints.append([+var_uv_smaller_w_i(t,u,v,w,i),-var_u_smaller_v_i(t,u,w,i),-var_u_smaller_v_i(t,v,w,i)])


		if DEBUG: print("c\tplanarity criterion")
		# definition 1.1 from http://page.math.tu-berlin.de/~felsner/Paper/ppg-rev.pdf
		for (u,v) in combinations(V,2):
			for w in set(V)-{u,v}:
				constraints.append([-var_edge_layer(t,u,v)]+[var_uv_smaller_w_i(t,u,v,w,i) for i in range(3)])


	if DEBUG: print("c\tfor each edge select a layer from 1 to",args.thickness)
	for (u,v) in combinations(V,2):
		constraints.append([-var_edge(u,v)]+[+var_edge_layer(t,u,v) for t in range(args.thickness)])




if args.alpha_upp:
	if DEBUG: print("c\talpha(G) <=",args.alpha_upp)
	for I in combinations(V,args.alpha_upp+1):
		constraints.append([+var_edge(u,v) for (u,v) in combinations(I,2)])

if args.omega_upp:
	if DEBUG: print("c\tomega(G) <=",args.omega_upp)
	for I in combinations(V,args.omega_upp+1):
		constraints.append([-var_edge(u,v) for (u,v) in combinations(I,2)])

if args.Ckfree:
	k = args.Ckfree
	if DEBUG: print("c\tCk-free",k)
	for I in permutations(V,k):
		if I[0] == min(I) and I[1] < I[-1]: 
			constraints.append([-var_edge(I[j-1],I[j]) for j in range(k)])

if args.Delta_upp:
	if DEBUG: print("c\tdelta(G) <=",args.Delta_upp)
	for u in V:
		#constraints += list(CardEnc.atmost(lits=[var_edge(u,v) for v in V if v != u], bound=args.Delta_upp, encoding=EncType.totalizer, vpool=vpool))
		counterFunction([var_edge(u,v) for v in V if v != u], args.Delta_upp, vpool, constraints, atMost=args.Delta_upp, type=args.countertype)

if args.delta_low:
	assert(len(V) > args.delta_low)
	if DEBUG: print("c\tdelta(G) >=",args.delta_low)
	for u in V:
		#constraints += list(CardEnc.atleast(lits=[var_edge(u,v) for v in V if v != u], bound=args.delta_low, encoding=EncType.totalizer, vpool=vpool))
		counterFunction([var_edge(u,v) for v in V if v != u], args.delta_low, vpool, constraints, atLeast=args.delta_low, type=args.countertype)

if args.evendegrees:
	if DEBUG: print("c\tforce even degrees")
	for u in V:
		shouldBe([+var_edge(u,v) for v in V if v != u], [i for i in V if i%2 == 0], vpool, constraints, type=args.countertype)


if args.num_edges_upp:
	if DEBUG: print("c\t#edges <=",args.num_edges_upp)
	#constraints += list(CardEnc.atmost(lits=[var_edge(u,v) for u,v in combinations(V,2)], bound=args.num_edges_upp, encoding=EncType.totalizer, vpool=vpool))
	counterFunction([+var_edge(u,v) for u,v in combinations(V,2)], args.num_edges_upp, vpool, constraints, atMost=args.num_edges_upp, type=args.countertype)

if args.num_edges_low:
	if DEBUG: print("c\t#edges >=",args.num_edges_low)
	#constraints += list(CardEnc.atleast(lits=[var_edge(u,v) for u,v in combinations(V,2)], bound=args.num_edges_low, encoding=EncType.totalizer, vpool=vpool))
	counterFunction([+var_edge(u,v) for u,v in combinations(V,2)], args.num_edges_low, vpool, constraints, atLeast=args.num_edges_low, type=args.countertype)

if args.clawfree:
	if DEBUG: print("c\tassert clawfree")
	for u in V:
		for I in combinations(set(V)-{u},3):
			constraints.append([-var_edge(u,v) for v in I])


if args.linegraph:
	if DEBUG: print("c\tassert linegraph")
	# classification via forbidden induced subgraphs
	# by Beineke / Soltes, see https://doi.org/10.1016/0012-365X(92)00577-E
	forbidden_induced_subgraphs = [ 
		[(0,1),(1,2),(2,4),(4,5),(3,4),(1,3),(2,3)],
		[(0,1),(1,3),(2,3),(0,2),(1,2),(3,4),(4,5),(0,5)],
		[(0,1),(1,3),(2,3),(0,2),(0,4),(1,4),(2,4)],
		[(4,5),(3,4),(2,3),(2,4),(1,2),(0,1),(0,3),(0,2),(1,3)],
		[(1,2),(1,3),(3,4),(2,4),(0,2),(0,3),(3,5),(2,5),(0,1),(4,5),(2,3)],
		[(0,1),(0,2),(0,3)],
		[(0,2),(1,2),(1,3),(0,3),(2,3),(3,4),(1,4),(2,4),(0,4)],
		[(0,5),(4,5),(3,4),(2,3),(1,2),(0,1),(1,5),(1,4),(2,4)],
		[(1,2),(2,5),(0,5),(0,1),(1,5),(3,5),(2,3),(3,4),(0,4),(4,5)],
	]
	for F in forbidden_induced_subgraphs:
		for I in permutations(V,6):
			constraints.append([var_edge(tuple(sorted(I[a],I[b])))*(-1 if (a,b) in F else +1) for a,b in combinations(V,2)])




if args.planar_universal:
	if DEBUG: print("c\tusing universal sets to assert that the graph is planar")

	def sgn(x): 
		return (x>0)-(x<0)

	def o3(a,b,c):
		(ax,ay) = a
		(bx,by) = b
		(cx,cy) = c
		return sgn((bx-ax)*(cy-ay)-(cx-ax)*(by-ay))

	def pq_contains_r(p,q,r):
		dx1 = q[0]-p[0]
		dy1 = q[1]-p[1]
		dx2 = r[0]-p[0]
		dy2 = r[1]-p[1]
		assert(dx1 != 0 or dy1 != 0)
		if dx1 != 0 and (dx1*dx2 < 0 or abs(dx2)>abs(dx1)): return False # r not in segment pq
		if dy1 != 0 and (dy1*dy2 < 0 or abs(dx2)>abs(dx1)): return False # r not in segment pq
		return (dx2*dy1 == dx1*dy2) # <=> r on line pq

	def pq_crosses_rs(p,q,r,s):
		return (o3(p,q,r) != o3(p,q,s)) and (o3(p,r,s) != o3(q,r,s)) or \
			pq_contains_r(p,q,r) or pq_contains_r(p,q,s) or pq_contains_r(r,s,p) or pq_contains_r(r,s,q)

	# injective mapping
	for v in V:
		constraints.append([+var_mapping(v,p) for p in P])
	for p in P:
		for v1,v2 in combinations(V,2):
			constraints.append([-var_mapping(v1,p),-var_mapping(v2,p)])
	for p1,p2 in combinations(P,2):
		for v in V:
			constraints.append([-var_mapping(v,p1),-var_mapping(v,p2)])

	# segments
	for u,v in combinations(V,2):
		for p,q in permutations(P,2):
			constraints.append([-var_edge(u,v),-var_mapping(u,p),-var_mapping(v,q),+var_segment(p,q)])

	# no crossings
	for p,q,r,s in permutations(P,4):
		if pq_crosses_rs(p,q,r,s):
			constraints.append([-var_segment(p,q),-var_segment(r,s)])



if args.loadgraph:
	from ast import literal_eval
	edges = literal_eval(open(args.loadgraph).readline())
	edges = [(tuple(e)) for e in edges]
	print("c\tload graph and fix edges:",edges)
	for a,b in edges: assert(a in V and b in V) 
	for a,b in combinations(V,2):
			constraints.append([+var_edge(a,b) if (a,b) in edges else -var_edge(a,b)])



if DEBUG: print("c\tTotal number of constraints:"+str(len(constraints)))




if not args.cnf2file: args.cnf2file = "tmp.enc"

print ("c\twrite cnf instance to file:",args.cnf2file)
from pysat.formula import CNF
cnf = CNF()
for c in constraints: cnf.append(c)
cnf.to_file(args.cnf2file)


if not args.solver:
	# use SMS
	import os
	default_params = " --useCadical --checkSolutionInProp --printStats --frequency 5 "
	if args.directed:
		if DEBUG: print ("c\trun SMS/digraphs")
		sms_path = "../../build/src/graphgendirected " + default_params
	else:
		if DEBUG: print ("c\trun SMS/graphs")
		sms_path = "../../build/src/graphgen  " + default_params

	sms_command = "time "+sms_path+" -v "+str(n)+" --printStats --dimacs "+args.cnf2file
	if not args.justone: sms_command += " --allModels"
	if args.hideGraphs: sms_command += " --hideGraphs"
	if args.planar_kuratowski: sms_command += " -planar 1"
	if DEBUG: print("run command:",sms_command)
	os.system(sms_command)


else:
	# use other solver
	if args.solver == "cadical":
		if DEBUG: print ("c\tuse pysat/Cadical")
		from pysat.solvers import Cadical    
		solver = Cadical()
		for c in constraints: solver.add_clause(c)
		solution_iterator = solver.enum_models()
	else:
		if DEBUG: print ("c\tuse pycosat")
		import pycosat
		solution_iterator = pycosat.itersolve(constraints)

	for ct,sol in enumerate(solution_iterator):
		ct += 1
		sol = set(sol)
		if args.directed:
			E = [(u,v) for (u,v) in permutations(V,2) if var_directed_edge(u,v) in sol]
			print("s\tsol",ct,"-> directed edges:",E)
		else:
			E = [(u,v) for (u,v) in combinations(V,2) if var_edge(u,v) in sol]
			print("s\tsol",ct,"-> edges:",E)

		if args.justone:
			print("s\tstop because args.justone=True")
			break

	print("s\ttotal number of solutions:",ct)