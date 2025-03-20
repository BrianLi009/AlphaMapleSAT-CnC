import pycosat
from sys import*
from ast import literal_eval
from itertools import *


def sgn(x): return (x>0)-(x<0)

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



if True:
	n = int(argv[1])
	V = range(n)

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

	P = P_universal[n]

	assert(len(P) >= len(V))

	all_variables = []
	all_variables += [('edge',(u,v)) for u,v in combinations(V,2)]
	all_variables += [('mapping',(v,p)) for v in V for p in P]
	all_variables += [('segment',(p,q)) for p,q in permutations(P,2)]

	all_variables_index = {}

	_num_vars = 0
	for v in all_variables:
		_num_vars += 1
		all_variables_index[v] = _num_vars

	def var(L):	return all_variables_index[L]
	def var_edge(*L): return var(('edge',L))
	def var_mapping(*L): return var(('mapping',L))
	def var_segment(*L): return var(('segment',L))

	CNF = []

	# injective mapping
	for v in V:
		CNF.append([+var_mapping(v,p) for p in P])
	for p in P:
		for v1,v2 in combinations(V,2):
			CNF.append([-var_mapping(v1,p),-var_mapping(v2,p)])
	for p1,p2 in combinations(P,2):
		for v in V:
			CNF.append([-var_mapping(v,p1),-var_mapping(v,p2)])

	# segments
	for u,v in combinations(V,2):
		for p,q in permutations(P,2):
			CNF.append([-var_edge(u,v),-var_mapping(u,p),-var_mapping(v,q),+var_segment(p,q)])

	# no crossings
	for p,q,r,s in permutations(P,4):
		if pq_crosses_rs(p,q,r,s):
			CNF.append([-var_segment(p,q),-var_segment(r,s)])

	
	ct = 0
	while True:
		print("searching...",end="\r")
		stdout.flush()
		
		sol = pycosat.solve(CNF) 
		if sol == "UNSAT": break

		ct += 1
		E = [(u,v) for u,v in combinations(V,2) if var_edge(u,v) in sol]
		print("found graph",ct,":",E)

		E_perms = []
		for pi in permutations(V):
			E_perm = [(min(pi[u],pi[v]),max(pi[u],pi[v])) for u,v in E]
			if E_perm not in E_perms:
				E_perms.append(E_perm)
				CNF.append([(-var_edge(u,v) if (u,v) in E_perm else +var_edge(u,v)) for (u,v) in combinations(V,2)])

	print("done.")


