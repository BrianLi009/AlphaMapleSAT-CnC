#!/usr/bin/python


from itertools import combinations, permutations
from sys import *
from ast import literal_eval
import pycosat


def find_realizer(V,E,d=3,assert_triangulation=True,DEBUG=False,just_one=False):
	break_symmetries1 = True
	break_symmetries2 = True
	n = len(V)
	#assert(n >= 4)
	D = range(d)

	all_variables = []
	all_variables += [('edge',(u,v)) for u,v in combinations(V,2)] # u,v is an edge in the original graph
	
	all_variables += [('triangulation_edge',(u,v)) for u,v in combinations(V,2)] # u,v is an edge in the triangulation (which contains the graph)

	all_variables += [('u_smaller_v_i',(u,v,i)) for i in D for u,v in permutations(V,2)] # u < v in i-th linear u_smaller_v_i 
	all_variables += [('uv_smaller_w_i',(u,v,w,i)) for i in D for u,v,w in permutations(V,3)] # u < w and v < w in i-th linear u_smaller_v_i 
	all_variables += [('uv_smaller_w',(u,v,w)) for u,v,w in permutations(V,3)] # for at least one ordering i
	all_variables += [('v_maximal_i',(v,i)) for i in D for v in V] # v is the maximal element in i-th linear order u_smaller_v_i 
	all_variables += [('v_maximal',(v,)) for v in V] # v is the maximal element in some order 
	all_variables += [('uv_maximal',(u,v)) for u,v in permutations(V,2)] # u and v are the maximal element in some order 
	
	all_variables += [('u_smaller_v_unique_i',(u,v,i)) for i in D for u,v in permutations(V,2)] # u < v is only in i-th linear (in others it is v > u)
	all_variables += [('u_smaller_v_unique',(u,v)) for u,v in permutations(V,2)] # u < v is in a unique order (in others it is v > u)

	all_variables += [('directed_edge',(u,v)) for u,v in permutations(V,2)] # u < v in some linear u_smaller_v_i 
	all_variables += [('colored_directed_edge',(u,v,i)) for i in D for u,v in permutations(V,2)] # u < v in i-th linear u_smaller_v_i 

	all_variables += [('colored_path_length_k',(u,v,i,k)) for i in D for u,v in permutations(V,2) for k in range(1,n)] # u < v in some linear u_smaller_v_i 
	all_variables += [('colored_path_length_k_via_z',(u,v,i,k,z)) for i in D for u,v in permutations(V,2) for z in set(V)-{u,v} for k in range(2,n)] # u < v in some linear u_smaller_v_i 
	all_variables += [('uv_blocked_by_w_in_i',(u,v,w,i)) for i in D for u,v,w in combinations(V,3)] # z is a reason why y (x<y) comes before x in order i


	assert_max_num_edges = True if d == 3 else False
	if assert_max_num_edges:
		max_count = 3*n-6
		M = range(1,max_count+1)
		all_variables += [('count',(I,k)) for I in combinations(V,2) for k in M] # sequential counting variables for edges

	all_variables_index = {}

	_num_vars = 0
	for v in all_variables:
	    all_variables_index[v] = _num_vars
	    _num_vars += 1

	def var(L):	return 1+all_variables_index[L]
	def var_u_smaller_v_i(*L): return var(('u_smaller_v_i',L))
	def var_uv_smaller_w_i(*L): return var(('uv_smaller_w_i',L))
	def var_uv_smaller_w(*L): return var(('uv_smaller_w',L))
	def var_v_maximal_i(*L): return var(('v_maximal_i',L))
	def var_v_maximal(*L): return var(('v_maximal',L))
	def var_uv_maximal(*L): return var(('uv_maximal',L))


	def var_colored_directed_edge(*L): return var(('colored_directed_edge',L))
	def var_directed_edge(*L): return var(('directed_edge',L))

	def var_colored_path_length_k(*L): return var(('colored_path_length_k',L))
	def var_colored_path_length_k_via_z(*L): return var(('colored_path_length_k_via_z',L))

	def var_u_smaller_v_unique_i(*L): return var(('u_smaller_v_unique_i',L))
	def var_u_smaller_v_unique(*L): return var(('u_smaller_v_unique',L))
	def var_uv_blocked_by_w_in_i(*L): return var(('uv_blocked_by_w_in_i',L))

	def var_edge(*L): return var(('edge',L))
	def var_triangulation_edge(*L): return var(('triangulation_edge',L))
	def var_count(*L): return var(('count',L))



	constraints = []


	if DEBUG: print("(*) define u_smaller_v_i")
	for i in D:
		# anti-symmetrie + total
		for u,v in permutations(V,2):
			constraints.append([+var_u_smaller_v_i(u,v,i),+var_u_smaller_v_i(v,u,i)])
			constraints.append([-var_u_smaller_v_i(u,v,i),-var_u_smaller_v_i(v,u,i)])

		# transitivity
		for u,v,w in permutations(V,3):
			constraints.append([-var_u_smaller_v_i(u,v,i),-var_u_smaller_v_i(v,w,i),+var_u_smaller_v_i(u,w,i)])


	if DEBUG: print("(*) define u_smaller_v_unique_i")
	for i in D:
		for u,v in permutations(V,2):
			constraints.append([-var_u_smaller_v_unique_i(u,v,i),+var_u_smaller_v_i(u,v,i)])
			for j in set(D)-{i}:
				constraints.append([-var_u_smaller_v_unique_i(u,v,i),-var_u_smaller_v_i(u,v,j)])
			constraints.append([+var_u_smaller_v_unique_i(u,v,i),-var_u_smaller_v_i(u,v,i)]+[+var_u_smaller_v_i(u,v,j) for j in set(D)-{i}])


	if DEBUG: print("(*) define u_smaller_v_unique")
	for u,v in permutations(V,2):
		constraints.append([-var_u_smaller_v_unique(u,v)]+[+var_u_smaller_v_unique_i(u,v,i) for i in D])
		for i in D:
			constraints.append([+var_u_smaller_v_unique(u,v),-var_u_smaller_v_unique_i(u,v,i)])


	if DEBUG: print("(*) maximal elements in i (for suspension vertex s_i)")
	for i in D:
		for v in V:
			for u in set(V)-{v}: 
				constraints.append([-var_v_maximal_i(v,i),+var_u_smaller_v_i(u,v,i)])
			constraints.append([+var_v_maximal_i(v,i)]+[-var_u_smaller_v_i(u,v,i) for u in set(V)-{v}])


	if DEBUG: print("(*) maximal elements (to indicate suspension vertices)")
	for v in V:
		constraints.append([-var_v_maximal(v)]+[+var_v_maximal_i(v,i) for i in D])
		for i in D:
			constraints.append([+var_v_maximal(v),-var_v_maximal_i(v,i)])


	if DEBUG: print("(*) pairs of maximal elements (to assign edges)")
	for u,v in permutations(V,2):
		constraints.append([-var_uv_maximal(u,v),+var_v_maximal(u)])
		constraints.append([-var_uv_maximal(u,v),+var_v_maximal(v)])
		constraints.append([+var_uv_maximal(u,v),-var_v_maximal(u),-var_v_maximal(v)])


	if DEBUG: print("(*) assert uv_smaller_w_i variable")
	for i in D:
		for u,v,w in permutations(V,3):
			constraints.append([-var_uv_smaller_w_i(u,v,w,i),+var_u_smaller_v_i(u,w,i)])
			constraints.append([-var_uv_smaller_w_i(u,v,w,i),+var_u_smaller_v_i(v,w,i)])
			constraints.append([+var_uv_smaller_w_i(u,v,w,i),-var_u_smaller_v_i(u,w,i),-var_u_smaller_v_i(v,w,i)])


	if DEBUG: print("(*) assert uv_smaller_w variable") 
	for u,v,w in permutations(V,3):
		constraints.append([-var_uv_smaller_w(u,v,w)]+[+var_uv_smaller_w_i(u,v,w,i) for i in D])
		for i in D:
			constraints.append([+var_uv_smaller_w(u,v,w),-var_uv_smaller_w_i(u,v,w,i)])


	if DEBUG: print("(*) assert directed_edge variable")
	for u,v in permutations(V,2):
		for w in set(V)-{u,v}:
			constraints.append([-var_directed_edge(u,v),+var_uv_smaller_w(u,v,w)])
		constraints.append([-var_directed_edge(u,v),+var_u_smaller_v_unique(u,v)])
		constraints.append([-var_directed_edge(u,v),-var_v_maximal(u)])
		constraints.append([+var_directed_edge(u,v),-var_u_smaller_v_unique(u,v),+var_v_maximal(u)]+[-var_uv_smaller_w(u,v,w) for w in set(V)-{u,v}])


	if DEBUG: print("(*) assert colored directed_edge variable")
	for u,v in permutations(V,2):
		for i in D:
			constraints.append([-var_colored_directed_edge(u,v,i),+var_directed_edge(u,v)])
			constraints.append([-var_colored_directed_edge(u,v,i),+var_u_smaller_v_unique_i(u,v,i)])
			constraints.append([+var_colored_directed_edge(u,v,i),-var_directed_edge(u,v),-var_u_smaller_v_unique_i(u,v,i)])

	"""
	for u,v in permutations(V,2):
		constraints.append([-var_directed_edge(u,v)]+[+var_colored_directed_edge(u,v,i) for i in D])
		for i in D:
			constraints.append([+var_directed_edge(u,v),-var_colored_directed_edge(u,v,i)])
	"""


	if DEBUG: print("(*) antichain")
	for u,v in permutations(V,2):
		constraints.append([+var_u_smaller_v_i(u,v,i) for i in D])


	if DEBUG: print("(*) distinct maximal elements")
	for v in V:
		for i,j in combinations(D,2):
			constraints.append([-var_v_maximal_i(v,i),-var_v_maximal_i(v,j)])
		

	if DEBUG: print("(*) assign edges")
	for u,v in combinations(V,2):
		constraints.append([-var_triangulation_edge(u,v),+var_directed_edge(u,v),+var_directed_edge(v,u),+var_uv_maximal(u,v)])
		if assert_triangulation:
			constraints.append([+var_triangulation_edge(u,v),-var_directed_edge(u,v)])
			constraints.append([+var_triangulation_edge(u,v),-var_directed_edge(v,u)])
			constraints.append([+var_triangulation_edge(u,v),-var_uv_maximal(u,v)])


	if DEBUG: print("(*) assert path variables")
	for i in D:
		for u,v in permutations(V,2):
			for k in range(1,n):
				if k==1:
					constraints.append([-var_colored_path_length_k(u,v,i,k),+var_colored_directed_edge(u,v,i)]+[+var_colored_directed_edge(v,u,j) for j in set(D)-{i}])
					constraints.append([+var_colored_path_length_k(u,v,i,k),-var_colored_directed_edge(u,v,i)])
					for j in set(D)-{i}:
						constraints.append([+var_colored_path_length_k(u,v,i,k),-var_colored_directed_edge(v,u,j)])
				else:
					constraints.append([-var_colored_path_length_k(u,v,i,k)]+[+var_colored_path_length_k(u,v,i,k-1)]+[+var_colored_path_length_k_via_z(u,v,i,k,z) for z in set(V)-{u,v}])
					constraints.append([+var_colored_path_length_k(u,v,i,k),-var_colored_path_length_k(u,v,i,k-1)])
					for z in set(V)-{u,v}:
						constraints.append([+var_colored_path_length_k(u,v,i,k),-var_colored_path_length_k_via_z(u,v,i,k,z)])
					
					for z in set(V)-{u,v}:
						constraints.append([-var_colored_path_length_k_via_z(u,v,i,k,z),+var_colored_path_length_k(u,z,i,k-1)])
						constraints.append([-var_colored_path_length_k_via_z(u,v,i,k,z),+var_colored_path_length_k(z,v,i,1)])
						constraints.append([+var_colored_path_length_k_via_z(u,v,i,k,z),-var_colored_path_length_k(u,z,i,k-1),-var_colored_path_length_k(z,v,i,1)])


	if DEBUG: print("(*) symmetry breaking for orders") 
	if True:
		for i in D:
			for u,v in combinations(V,2):
				assert(u<v)
				constraints.append([+var_v_maximal_i(u,i),+var_colored_path_length_k(v,u,i,n-1),+var_u_smaller_v_i(u,v,i)]+[+var_uv_blocked_by_w_in_i(u,v,w,i) for w in V if w>v]) 

			for u,v,w in combinations(V,3):
				constraints.append([-var_uv_blocked_by_w_in_i(u,v,w,i),+var_u_smaller_v_i(v,w,i)])
				constraints.append([-var_uv_blocked_by_w_in_i(u,v,w,i),+var_colored_path_length_k(w,u,i,n-1)])
				constraints.append([+var_uv_blocked_by_w_in_i(u,v,w,i),-var_u_smaller_v_i(v,w,i),-var_colored_path_length_k(w,u,i,n-1)])


	if True:
		if assert_max_num_edges:
			if DEBUG: print("(*) symmetry breaking for Schnyder woods") # TODO: stefan nochmal fragen
			for (x,y,z) in permutations(V,3):
				constraints.append([-var_colored_directed_edge(x,y,0),-var_colored_directed_edge(y,z,1),-var_colored_directed_edge(z,x,2)])



	if E != None:
		if DEBUG: print("(*) sync with prescribed edges")
		for u,v in combinations(V,2):
			if (u,v) in E:
				constraints.append([+var_edge(u,v)])
			else:
				constraints.append([-var_edge(u,v)])


	if DEBUG: print("(*) original graph can be triangulated")
	for u,v in combinations(V,2):
		constraints.append([-var_edge(u,v),+var_triangulation_edge(u,v)])


	for u,v in combinations(V,2):
		for i,j in combinations(D,2):
			constraints.append([-var_v_maximal_i(u,j),-var_v_maximal_i(v,i)])


	# TODO: for symmetry breaking we can assume that 0,n-1 is an edge.
	# thus in order i=0, s_0=0 is maximal, and in i=2, s_2=n-1 is maximal. 	
	# moreover, for s_1 we can assume that s_1 is chosen minimal among 1,...,n-2,
	# that is, there are no triangles 0,v,n-1 for v<s_1.

	if 1:
		print("(*) symmetry breaking for the three suspending vertices")
		constraints.append([+var_v_maximal_i(0,0)])
		constraints.append([+var_v_maximal_i(n-1,2)])
		for v in V:
			for w in set(V)-{0,n-1}:
				if w < v:
					constraints.append([-var_v_maximal_i(v,1),-var_triangulation_edge(0,w),-var_triangulation_edge(w,n-1)])



	if assert_max_num_edges: 
		print ("(*) assert counting variables")
		to_count = [I for I in combinations(V,2)]
		

		I0 = to_count[0]
		if max(M) > 0:
			constraints.append([-var_triangulation_edge(*I0),+var_count(I0,1)])
			constraints.append([+var_triangulation_edge(*I0),-var_count(I0,1)])
		else:
			constraints.append([-var_triangulation_edge(*I0)])

		for k in set(M)-{1}:
			constraints.append([-var_count(I0,k)])


		for t in range(1,len(to_count)):
			I = to_count[t]
			prevI = to_count[t-1]
			constraints.append([-var_triangulation_edge(*I),+var_count(I,1)])
			for k in M:
				constraints.append([-var_count(prevI,k),+var_count(I,k)])
				constraints.append([+var_count(prevI,k),+var_triangulation_edge(*I),-var_count(I,k)])
				if k < max(M):
					constraints.append([-var_count(prevI,k),-var_triangulation_edge(*I),+var_count(I,k+1)])
					constraints.append([+var_count(prevI,k),-var_count(I,k+1)])
				else:
					constraints.append([-var_count(prevI,k),-var_triangulation_edge(*I)])


		I_last = to_count[-1]
		constraints.append([+var_count(I_last,max_count)])
		assert(max_count == 3*n-6)


	if DEBUG: print("Total number of constraints:"+str(len(constraints)))

	found = 0
	prev_sol = None

	signatures = {}

	for ct,sol in enumerate(pycosat.itersolve(constraints)):
		sol = set(sol)
		found += 1
		if just_one: break
		
		if DEBUG: print("debug: found solution#",ct+1)#,sol)


		if 1:
			print("edges:",[(u,v) for u,v in combinations(V,2) if var_edge(u,v) in sol])
			print("t-edges:",[(u,v) for u,v in combinations(V,2) if var_triangulation_edge(u,v) in sol])
		if 0:
			directed_edges = [(u,v) for u,v in permutations(V,2) if var_directed_edge(u,v) in sol]
			print("directed_edges:",directed_edges)
			colored_directed_edges = [(u,v,i) for u,v in permutations(V,2) for i in D if var_colored_directed_edge(u,v,i) in sol]
			print("colored_directed_edges:",colored_directed_edges)
			all_orders = []
			for i in D:
				order_i = []
				for j in V:
					for v in set(V)-set(order_i):
						if not [(u,v) for u in set(V)-set(order_i)-{v} if var_u_smaller_v_i(u,v,i) in sol]: # if no element u is before v
							order_i.append(v)
							break

				#print("order",i,":",order_i)
				all_orders.append(order_i)

			#sig = tuple(tuple(o) for o in all_orders)
			sig = tuple(directed_edges)
			#sig = tuple(colored_directed_edges)
				
			print ("all_orders",all_orders)

			if sig in signatures:
				prev_sol = signatures[sig]
				print("difference+",[I for I in all_variables_index if var(I) in sol and var(I) not in prev_sol])
				print("difference-",[I for I in all_variables_index if var(I) not in sol and var(I) in prev_sol])
				exit("error!")
			else:
				signatures[sig] = sol


	print("debug: number of solutions",found)
	return found > 0


d = int(argv[1])
n = int(argv[2]) # number of vertices
V = range(n)

#m = int(argv[3]) # number of edges
#assert(m >= 0 and m <= (n*(n-1))/2)

find_realizer(V,[],d)


"""
for G in graphs.nauty_geng(str(n)+' '+str(m)):

	print(80*"-")
	assert(set(G.vertices())==set(V))
	E = list(G.edges(labels=0))
	print((V,E),"is",str(d)+"-realizable:",find_realizer(V,E,d))
	#print("2n-4 should be",2*n-4)
"""

"""
ct = 0
for line in open(argv[1]):
	ct += 1
	E = literal_eval(line)
	n = max(1+max(u,v) for (u,v) in E)
	V = range(n)
	print(80*"=")
	print("test graph:",ct,E)
	assert(find_realizer(V,E,just_one=True))
"""
