from itertools import combinations, permutations
from sys import *
import counterImplementations
from sat_planar import *
DEFAULT_COUNTER = "sequential"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n",type=int,help="number of vertices", required=True)
parser.add_argument("--chi",type=int,help="at least chromatic number chi")
parser.add_argument("--explicitColoring",action="store_true",help="Add explictly all coloring clauses at the beginning")
parser.add_argument("--triangulation", help="ensure that one graph is a triangulation/maximal planar graph", action="store_true")
parser.add_argument("--explizitlyForbidSubgraph", help="Exludes K_5 and K_\{3,3\} from the subgraph", action="store_true")
parser.add_argument("--critical", help="Graph must be vertex critical with respect to the coloring", action="store_true")
parser.add_argument("--graphFile", help="Contains all planar graphs for the given order", type=str)

parser.add_argument('--planar_order', help="Both graphs G_1 and G_2 must be planar, order based planarity encoding (standard)", action="store_true")
parser.add_argument('--planar_order_heavy', help="Both graphs G_1 and G_2 must be planar; with symmetry breaking, order based planarity encoding (heavy)", action="store_true")

parser.add_argument("--excludeK9", help="Exludes K_9 as a subgraph of the complete graph", action="store_true")

parser.add_argument("--DEBUG","-D",type=int,default=1,help="debug level")
args = parser.parse_args()

if args.DEBUG:
	print("c\targs:",args)
vargs = vars(args)
print("c\tactive args:",{x:vargs[x] for x in vargs if vargs[x] != None and vargs[x] != False})

n = args.n
V = range(n)

DEBUG = args.DEBUG

all_variables = []
all_variables += [('edge',(u,v)) for u in V for v in V if u != v]

thickness = 2
all_variables += [('edge_layer',(t,u,v)) for t in range(thickness) for u,v in combinations(V,2)]
all_variables += [('u_smaller_v_i',(t,u,v,i)) for t in range(thickness) for i in range(3) for u,v in permutations(V,2)] # u < v in i-th linear order 
all_variables += [('uv_smaller_w_i',(t,u,v,w,i)) for t in range(thickness) for i in range(3) for u,v,w in permutations(V,3) if u<v] # u < w and v < w in i-th linear order 


all_variables_index = {}

_num_vars = 0
for v in all_variables:
    _num_vars += 1
    all_variables_index[v] = _num_vars


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
def var_edge(u,v): return var(('edge',(u,v)))

def var_edge_layer(*L): return var(('edge_layer',L))
def var_u_smaller_v_i(*L): return var(('u_smaller_v_i',L))
def var_uv_smaller_w_i(*L): return var(('uv_smaller_w_i',L))

def CNF_OR(ins, out):
    return [[-out] + ins] + [[out, -x] for x in ins]

def CNF_AND(ins, out):
    return [[out] + [-x for x in ins]] + [[-out, x] for x in ins]

constraints = []

G = [[None for _ in V] for _ in V]
for v,u in combinations(V,2):
	G[v][u] = G[u][v] = var_edge(u,v) #lower triangle gives union of both

G1 = [[None for _ in V] for _ in V]
for v,u in combinations(V,2):
	G1[v][u] = G1[u][v] = var_edge(v,u) #upper triangle matrix means at both

G2 = [[None for _ in V] for _ in V]
for v,u in combinations(V,2):
	G2[v][u] = G2[u][v] = vpool.id()
	clauses = CNF_AND([+var_edge(u,v), -var_edge(v,u)], G2[v][u]) # edge in G2 if single edge in lower half
	constraints.extend(clauses)


# only single edges in lower triangle matrix; i.e., if edge in upper than also in lower
for v,u in combinations(V,2):
	constraints.append([-var_edge(v,u), +var_edge(u,v)]) # if one upper half than also on lower half

# w.l.o.g.: first graph is triangulation
if args.triangulation:
	triangulation = vpool.id() # true if upper triangle should be a triangulation
	# euler and F=E*2/3 to get number of edges: V + F - E = 2; F = E * 2 / 3 -> V - 2= 1/3 E -> E = 3(V - 2)
	edgesInTriangulation = 3*(n - 2)
	counterGraph1 = counterImplementations.counterFunction([var_edge(i,j) for i,j in combinations(V,2)], countUpto=edgesInTriangulation, vPool=vpool,
					clauses=constraints, atMost=edgesInTriangulation, atLeast=edgesInTriangulation, type=DEFAULT_COUNTER)

# graphs given by each direction (upper and lower triangular) must be planar
if args.planar_order_heavy:
	planar_encoding_heavy(lambda v,u: G1[v][u], vpool, constraints)
	planar_encoding_heavy(lambda v,u: G2[v][u], vpool, constraints)

if args.planar_order:
	planar_encoding(V, lambda v,u: G1[v][u], vpool, constraints)
	planar_encoding(V, lambda v,u: G2[v][u], vpool, constraints)
	

# mindegree on undirected version
if args.chi:
	for i in V:
		counterImplementations.counterFunction([var_edge(i,j) for j in V if j < i] + [var_edge(j,i) for j in V if j > i], countUpto=args.chi - 1, vPool=vpool,
					clauses=constraints, atLeast=args.chi - 1, type=DEFAULT_COUNTER)


# vertex critical with respect to color (might be okay if the other encodings are large otherwise to much overhead)
if args.critical:
	nColors = args.chi - 1
	for v in V:
		# check if G-v is args.critical - 1 colorable
		colors = [[vpool.id() for _ in V] for _ in range(nColors)]
		# at least one color
		for u in V:
			if u != v:
				constraints.append([colors[r][u] for r in range(nColors)])
		# adjacent once cannot have the same color
		for u1,u2 in permutations(V,2):
			if u1 == v or u2 == v:
				continue
			for r in range(nColors):
				constraints.append([-var_edge(u1,u2), -colors[r][u1], -colors[r][u2]])

# explicitly exclude K5 and K3_3
if args.explizitlyForbidSubgraph:
	for A in combinations(V,5):
		constraints.append([-G1[i][j] for i,j in combinations(A,2)])
		constraints.append([-G2[i][j] for i,j in combinations(A,2)])

	for A in combinations(V,6):
		for B in combinations(A,3):
			if min(A) not in B:
				continue
			constraints.append([-G1[i][j] for i in set(A) - set(B) for j in B])
			constraints.append([-G2[i][j] for i in set(A) - set(B) for j in B])

if args.graphFile:
	from sage.all import Graph

	graphs = []
	for line in open(args.graphFile):
		graphs.append([(t[0], t[1]) for t in Graph(line).edges()])

	guessGraph1 = [vpool.id() for _ in range(len(graphs))]
	guessGraph2 = [vpool.id() for _ in range(len(graphs))]

	constraints.append(guessGraph1)
	constraints.append(guessGraph2)



	perm1 = [[vpool.id() for _ in V] for _ in V]
	perm2 = [[vpool.id() for _ in V] for _ in V]
	for perm in [perm1, perm2]:
		for row in perm:
			constraints.append(row) # each vertex must be mapped somewhere
		for x in V:
			for v,u in combinations(V, 2):
				for x in V:
					constraints.append([-perm[v][x], -perm[u][x]]) # not allowed to be mapped to the same vertex

	perm = perm1
	for i, H in enumerate(graphs):
		# perm[v][u] vertex v \in H is mapped to vertex u in G
		for v,u in combinations(V,2):
			if (v,u) in H or (u,v) in H:
				for v2, u2 in permutations(V,2):
					constraints.append([-guessGraph1[i], -perm[v][v2], -perm[u][u2], var_edge(min(v2,u2),max(v2,u2))]) # edge must be present
			else:
				for v2, u2 in permutations(V,2):
					constraints.append([-guessGraph1[i], -perm[v][v2], -perm[u][u2], -var_edge(min(v2,u2),max(v2,u2))]) # edge must NOT be present

	perm = perm2
	for i, H in enumerate(graphs):
		# perm[v][u] vertex v \in H is mapped to vertex u in G
		for v,u in combinations(V,2):
			if (v,u) in H or (u,v) in H:
				for v2, u2 in permutations(V,2):
					constraints.append([-guessGraph2[i], -perm[v][v2], -perm[u][u2], var_edge(max(v2,u2),min(v2,u2))]) # edge must be present
			else:
				for v2, u2 in permutations(V,2):
					constraints.append([-guessGraph2[i], -perm[v][v2], -perm[u][u2], -var_edge(max(v2,u2),min(v2,u2))]) # edge must NOT be present

	
if args.excludeK9:
	for A in combinations(V,9):
		constraints.append([-var_edge(j,i) for i,j in combinations(V,2)]) # some edge in a at least 

from random import *
if args.explicitColoring:
	from more_itertools import set_partitions
	for coloring in set_partitions(V,args.chi - 1):
		# print(x)
		clause = [] # monochromatic edge
		for color in coloring:
			clause.extend([G[v][u] for v,u in combinations(color, 2)]) 
		constraints.append(clause)

if DEBUG: print("c\tTotal number of constraints:"+str(len(constraints)))


print('c\tbegin of CNF')
for c in constraints:
	print (' '.join(str(x) for x in c))
print('c\tend of CNF')



