from itertools import *

def planar_encoding_heavy(V, var_edge, vpool, constraints, DEBUG=False):
	D = range(3)
	all_variables = []
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
	all_variables_index = {}

	for v in all_variables:
		all_variables_index[v] = vpool.id()

	def var(L):	return all_variables_index[L]
	def var_u_smaller_v_unique_i(*L): return var(('u_smaller_v_unique_i',L))
	def var_u_smaller_v_unique(*L): return var(('u_smaller_v_unique',L))
	def var_uv_blocked_by_w_in_i(*L): return var(('uv_blocked_by_w_in_i',L))
	def var_u_smaller_v_i(*L): return var(('u_smaller_v_i',L))
	def var_uv_smaller_w_i(*L): return var(('uv_smaller_w_i',L))
	def var_uv_smaller_w(*L): return var(('uv_smaller_w',L))
	def var_v_maximal_i(*L): return var(('v_maximal_i',L))
	def var_v_maximal(*L): return var(('v_maximal',L))
	def var_uv_maximal(*L): return var(('uv_maximal',L))
	
	def var_colored_directed_edge(*L): return var(('colored_directed_edge',L))
	def var_directed_edge(*L): return var(('directed_edge',L))

	def var_triangulation_edge(*L): return var(('triangulation_edge',L))

	def var_colored_path_length_k(*L): return var(('colored_path_length_k',L))
	def var_colored_path_length_k_via_z(*L): return var(('colored_path_length_k_via_z',L))


	if DEBUG: print("c\tdefine u_smaller_v_i")
	for i in D:
		# anti-symmetrie + total
		for u,v in permutations(V,2):
			constraints.append([+var_u_smaller_v_i(u,v,i),+var_u_smaller_v_i(v,u,i)])
			constraints.append([-var_u_smaller_v_i(u,v,i),-var_u_smaller_v_i(v,u,i)])

		# transitivity
		for u,v,w in permutations(V,3):
			constraints.append([-var_u_smaller_v_i(u,v,i),-var_u_smaller_v_i(v,w,i),+var_u_smaller_v_i(u,w,i)])


	if DEBUG: print("c\tdefine u_smaller_v_unique_i")
	for i in D:
		for u,v in permutations(V,2):
			constraints.append([-var_u_smaller_v_unique_i(u,v,i),+var_u_smaller_v_i(u,v,i)])
			for j in set(D)-{i}:
				constraints.append([-var_u_smaller_v_unique_i(u,v,i),-var_u_smaller_v_i(u,v,j)])
			constraints.append([+var_u_smaller_v_unique_i(u,v,i),-var_u_smaller_v_i(u,v,i)]+[+var_u_smaller_v_i(u,v,j) for j in set(D)-{i}])


	if DEBUG: print("c\tdefine u_smaller_v_unique")
	for u,v in permutations(V,2):
		constraints.append([-var_u_smaller_v_unique(u,v)]+[+var_u_smaller_v_unique_i(u,v,i) for i in D])
		for i in D:
			constraints.append([+var_u_smaller_v_unique(u,v),-var_u_smaller_v_unique_i(u,v,i)])


	if DEBUG: print("c\tmaximal elements in i (for suspension vertex s_i)")
	for i in D:
		for v in V:
			for u in set(V)-{v}: 
				constraints.append([-var_v_maximal_i(v,i),+var_u_smaller_v_i(u,v,i)])
			constraints.append([+var_v_maximal_i(v,i)]+[-var_u_smaller_v_i(u,v,i) for u in set(V)-{v}])


	if DEBUG: print("c\tmaximal elements (to indicate suspension vertices)")
	for v in V:
		constraints.append([-var_v_maximal(v)]+[+var_v_maximal_i(v,i) for i in D])
		for i in D:
			constraints.append([+var_v_maximal(v),-var_v_maximal_i(v,i)])


	if DEBUG: print("c\tpairs of maximal elements (to assign edges)")
	for u,v in permutations(V,2):
		constraints.append([-var_uv_maximal(u,v),+var_v_maximal(u)])
		constraints.append([-var_uv_maximal(u,v),+var_v_maximal(v)])
		constraints.append([+var_uv_maximal(u,v),-var_v_maximal(u),-var_v_maximal(v)])


	if DEBUG: print("c\tassert uv_smaller_w_i variable")
	for i in D:
		for u,v,w in permutations(V,3):
			constraints.append([-var_uv_smaller_w_i(u,v,w,i),+var_u_smaller_v_i(u,w,i)])
			constraints.append([-var_uv_smaller_w_i(u,v,w,i),+var_u_smaller_v_i(v,w,i)])
			constraints.append([+var_uv_smaller_w_i(u,v,w,i),-var_u_smaller_v_i(u,w,i),-var_u_smaller_v_i(v,w,i)])


	if DEBUG: print("c\tassert uv_smaller_w variable") 
	for u,v,w in permutations(V,3):
		constraints.append([-var_uv_smaller_w(u,v,w)]+[+var_uv_smaller_w_i(u,v,w,i) for i in D])
		for i in D:
			constraints.append([+var_uv_smaller_w(u,v,w),-var_uv_smaller_w_i(u,v,w,i)])


	if DEBUG: print("c\tassert directed_edge variable")
	for u,v in permutations(V,2):
		for w in set(V)-{u,v}:
			constraints.append([-var_directed_edge(u,v),+var_uv_smaller_w(u,v,w)])
		constraints.append([-var_directed_edge(u,v),+var_u_smaller_v_unique(u,v)])
		constraints.append([-var_directed_edge(u,v),-var_v_maximal(u)])
		constraints.append([+var_directed_edge(u,v),-var_u_smaller_v_unique(u,v),+var_v_maximal(u)]+[-var_uv_smaller_w(u,v,w) for w in set(V)-{u,v}])


	if DEBUG: print("c\tassert colored directed_edge variable")
	for u,v in permutations(V,2):
		for i in D:
			constraints.append([-var_colored_directed_edge(u,v,i),+var_directed_edge(u,v)])
			constraints.append([-var_colored_directed_edge(u,v,i),+var_u_smaller_v_unique_i(u,v,i)])
			constraints.append([+var_colored_directed_edge(u,v,i),-var_directed_edge(u,v),-var_u_smaller_v_unique_i(u,v,i)])


	if DEBUG: print("c\tantichain")
	for u,v in permutations(V,2):
		constraints.append([+var_u_smaller_v_i(u,v,i) for i in D])


	if DEBUG: print("c\tdistinct maximal elements")
	for v in V:
		for i,j in combinations(D,2):
			constraints.append([-var_v_maximal_i(v,i),-var_v_maximal_i(v,j)])
		

	if DEBUG: print("c\tassign edges")
	for u,v in combinations(V,2):
		constraints.append([-var_triangulation_edge(u,v),+var_directed_edge(u,v),+var_directed_edge(v,u),+var_uv_maximal(u,v)])
		constraints.append([+var_triangulation_edge(u,v),-var_directed_edge(u,v)])
		constraints.append([+var_triangulation_edge(u,v),-var_directed_edge(v,u)])
		constraints.append([+var_triangulation_edge(u,v),-var_uv_maximal(u,v)])


	if DEBUG: print("c\tassert path variables")
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


	if DEBUG: print("c\tsymmetry breaking for orders") 
	if True:
		for i in D:
			for u,v in combinations(V,2):
				assert(u<v)
				constraints.append([+var_v_maximal_i(u,i),+var_colored_path_length_k(v,u,i,n-1),+var_u_smaller_v_i(u,v,i)]+[+var_uv_blocked_by_w_in_i(u,v,w,i) for w in V if w>v]) 

			for u,v,w in combinations(V,3):
				constraints.append([-var_uv_blocked_by_w_in_i(u,v,w,i),+var_u_smaller_v_i(v,w,i)])
				constraints.append([-var_uv_blocked_by_w_in_i(u,v,w,i),+var_colored_path_length_k(w,u,i,n-1)])
				constraints.append([+var_uv_blocked_by_w_in_i(u,v,w,i),-var_u_smaller_v_i(v,w,i),-var_colored_path_length_k(w,u,i,n-1)])

	if DEBUG: print("c\toriginal graph can be triangulated")
	for u,v in combinations(V,2):
		constraints.append([-var_edge(u,v),+var_triangulation_edge(u,v)])


	for u,v in combinations(V,2):
		for i,j in combinations(D,2):
			constraints.append([-var_v_maximal_i(u,j),-var_v_maximal_i(v,i)])

	if DEBUG: print("c\tsymmetry breaking for Schnyder woods") # TODO: stefan nochmal fragen
	for (x,y,z) in permutations(V,3):
		constraints.append([-var_colored_directed_edge(x,y,0),-var_colored_directed_edge(y,z,1),-var_colored_directed_edge(z,x,2)])


	# if 1: #!!!!Assumes lexicographically minimal and degree at least 1. 
	# 	print("c\tsymmetry breaking for the three suspending vertices")
	# 	constraints.append([+var_v_maximal_i(0,0)])
	# 	constraints.append([+var_v_maximal_i(n-1,2)])
	# 	for v in V:
	# 		for w in set(V)-{0,n-1}:
	# 			if w < v:
	# 				constraints.append([-var_v_maximal_i(v,1),-var_triangulation_edge(0,w),-var_triangulation_edge(w,n-1)])


def planar_encoding(V, var_edge, vpool, constraints, DEBUG=False, fixRotation = False, fixTriangle = False, fixFirstVertex = True, fixTriangleInTriangulation=False):
	D = range(3)
	all_variables = []
	all_variables += [('triangulation_edge',(u,v)) for u,v in combinations(V,2)] # u,v is an edge in the triangulation (which contains the graph)

	all_variables += [('u_smaller_v_i',(u,v,i)) for i in D for u,v in permutations(V,2)] # u < v in i-th linear u_smaller_v_i 
	all_variables += [('uv_smaller_w_i',(u,v,w,i)) for i in D for u,v,w in permutations(V,3)] # u < w and v < w in i-th linear u_smaller_v_i 

	all_variables_index = {}

	for v in all_variables:
		all_variables_index[v] = vpool.id()

	def var(L):	return all_variables_index[L]
	def var_u_smaller_v_i(*L): return var(('u_smaller_v_i',L))
	def var_uv_smaller_w_i(*L): return var(('uv_smaller_w_i',L))
	if DEBUG: print("c\tdefine linear orders")
	for i in range(3):
		# anti-symmetrie + connexity
		for u,v in permutations(V,2):
			constraints.append([+var_u_smaller_v_i(u,v,i),+var_u_smaller_v_i(v,u,i)])
			constraints.append([-var_u_smaller_v_i(u,v,i),-var_u_smaller_v_i(v,u,i)])

		# transitivity
		for u,v,w in permutations(V,3):
			constraints.append([-var_u_smaller_v_i(u,v,i),-var_u_smaller_v_i(v,w,i),+var_u_smaller_v_i(u,w,i)])

	if DEBUG: print("c\tassert uv_smaller_w_i variable")
	for i in range(3):
		for u,v,w in permutations(V,3):
			if u<v:
				constraints.append([-var_uv_smaller_w_i(u,v,w,i),+var_u_smaller_v_i(u,w,i)])
				constraints.append([-var_uv_smaller_w_i(u,v,w,i),+var_u_smaller_v_i(v,w,i)])
				constraints.append([+var_uv_smaller_w_i(u,v,w,i),-var_u_smaller_v_i(u,w,i),-var_u_smaller_v_i(v,w,i)])

	# if DEBUG: print("c\tantichain")
	# for u,v in permutations(V,2):
	# 	constraints.append([+var_u_smaller_v_i(u,v,i) for i in D])


	# # !!!!!!! w.l.o.g. first vertex the smallest in first order
	if fixFirstVertex:
		for v in V:
			if v == 0: continue
			constraints.append([+var_u_smaller_v_i(0,v,0)])
	if fixTriangle:
		max2 = [vpool.id() for _ in V]
		max3 = [vpool.id() for _ in V]
		# TODO 

		

	if fixTriangleInTriangulation:
		var_triangle = { e: vpool.id() for e in combinations(V, 2)}
		for u,v in combinations(V,2):
			constraints.append([-var_edge(u,v), var_triangle((u,v))])
			
		var_triangle_reason_not = { (e,v): vpool.id() for e in combinations(V, 2) for v in set(V) - {e[0], e[1]}}

		pass


	if fixRotation: 
		exit() # TODO probably false or not 
		# !!!!!!!!! w.l.o.g fix rotation of triangles.
		for u,v,w in permutations(V,3):
			constraints.append([-var_edge(u,v), -var_edge(v,w), -var_edge(u,w), 
				-var_u_smaller_v_i(u,v,0), +var_u_smaller_v_i(u,v,1), +var_u_smaller_v_i(u,v,2),
				+var_u_smaller_v_i(v,w,0), -var_u_smaller_v_i(v,w,1), +var_u_smaller_v_i(v,w,2), 
				+var_u_smaller_v_i(w,u,0), +var_u_smaller_v_i(w,u,1), -var_u_smaller_v_i(w,u,2)])

		
	if DEBUG: print("c\tplanarity criterion")
	# definition 1.1 from http://page.math.tu-berlin.de/~felsner/Paper/ppg-rev.pdf
	for (u,v) in combinations(V,2):
		for w in set(V)-{u,v}:
			constraints.append([-var_edge(u,v)]+[var_uv_smaller_w_i(u,v,w,i) for i in range(3)]) # in upper triangle
			constraints.append([-var_edge(v,u)]+[var_uv_smaller_w_i(u,v,w,i) for i in range(3)]) # in lower triangle

