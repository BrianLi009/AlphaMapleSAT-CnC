from z3 import *
from ast import literal_eval
from itertools import *
from collections import defaultdict
import sys
import random
from fractions import Fraction
# set_param(proof=True)

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--vertices", "-n", type=int, help="number of vertices")
parser.add_argument("--verbose", "-v", action="count", default=0, help="increase verbosity")
parser.add_argument("--minimize", "-m", action="store_true", help="try to extract unembeddable cores (subgraphs) from unembeddable graphs")
parser.add_argument("graph_file", help="file of graphs to process")
args = parser.parse_args()

n = args.vertices

result_map = {"unknown" : "timeout", "sat" : "embedding found", "unsat" : "unembeddable"} 

class Vector:
    def __init__(self, x = None, y = None, z = None):
        self.x = x
        self.y = y
        self.z = z
        self.xr = None
        self.yr = None
        self.zr = None

class CrossProductVector(Vector):
    # assigned as crossproduct of two others
    def __init__(self, v1 = None, v2 = None, x = None, y = None ,z = None):
        Vector.__init__(self,  x, y, z)
        self.v1 = v1
        self.v2 = v2

class StartingVector(Vector):
    # has to be guessed by the solver; (fix length to 1 and z must be non-negative)
    def __init__(self, x,y,z):
        Vector.__init__(self, x, y, z)

class OrthogonalBaseVector(Vector):
    # one of the three vectors: (0,0,1), (0,1,0), (1,0,0)
    def __init__(self, x,y,z):
        Vector.__init__(self, x, y, z)


def crossproduct(v1, v2):
    return ( v1.y * v2.z - v1.z * v2.y,  v1.z * v2.x - v1.x * v2.z,  v1.x * v2.y - v1.y * v2.x)



def solve(G, a, timeout = 10):
    # printInRedlog(G,a)
    # solve giving a vector assignment and the graph
    base = [i for i,v in enumerate(a) if isinstance(v, OrthogonalBaseVector)]
    if args.verbose >= 1:
        free_vertices_fixed = [i for i,v in enumerate(a) if isinstance(v, OrthogonalBaseVector)]
        free_vertices_basis = [i for i,v in enumerate(a) if isinstance(v, StartingVector)]
        bound_vertices = [i for i,v in enumerate(a) if isinstance(v, CrossProductVector)]
        print(f"Trying the cross-product cover (fixed {'edge    ' if len(free_vertices_fixed) == 2 else 'triangle'} | free | bound) [ ", end="")
        print(" ".join(map(str, free_vertices_fixed)), end=" | ")
        print(" ".join(map(str, free_vertices_basis)), end=" | ")
        print(" ".join(map(str, bound_vertices)), end=" ] ")
        sys.stdout.flush()

    # for i in [i for i,v in enumerate(a) if isinstance(v, CrossProductVector)]:
    #    print(i, "is given by", a[i].v1, a[i].v2)

    s = Solver() # new solver for the graph
    s.set("timeout", timeout * 1000) # timeout in miliseconds

    # s.set("timeout", 10)
    useDisjunctions = True
    ''' 
    for v in a:
        if isinstance(v, StartingVector):
            s.add(v.x * v.x + v.y * v.y + v.z * v.z == 1) # unit
            s.add(v.z >= 0) # z must be positive '''

        # if useDisjunctions:
        #     s.add(Or(v.x != 0, v.y != 0, v.z != 0)) # already handled in line where ensuring that they are not collinear

    # all orthogonal relations if not specified by crossproduct or start
    for u,v in G:
        # print(a[u].x, a[v].x, a[u].y, a[v].y, a[u].z, a[v].z)
        s.add(a[u].x * a[v].x + a[u].y * a[v].y + a[u].z * a[v].z == 0)
        
        # decided to skip this because for crossproduct both most be known for crossproduct only a sinlge one
        # if isinstance(u, CrossProductVector) and (a[u].v1 == v or a[u].v2 == v):
        #    continue

        #if isinstance(v, CrossProductVector) and (a[v].v1 == u or a[v].v2 == u):
        #    continue

    # pairwise distinct (i.e., not collinear)
    for u,v in combinations(range(n),2):
        # if 1 in [u,v]:
        #    continue
        x,y,z = crossproduct(a[u], a[v])
        if useDisjunctions:
            s.add(Or(x != 0, y != 0, z != 0)) # if crossproduct is zero then collinear
        else:
            s.add(x * x + y * y + z * z > 0) 

    res = s.check()
    if args.verbose >= 1:
        print(result_map[str(res)], end="") 
        if str(res) == "unknown":
            print(f" ({timeout}s)")
        else:
            print()

            
    if args.verbose >= 2:
        print(s.statistics())
        sys.stdout.flush()
    # print( s.proof())
    # m = s.model()
    # print(m[a[1].x], m[a[1].y], m[a[1].z])
    s.reset()
    # solve giving an assignment a and a graph G
    return res

def assignCrossproductDependencies(G, assignment, expand, assigned):
    # BFS search like; list of vertices two expand; i.e.; assign all adjacent crossproduct vertices which are not assigned yet the vector
    order = expand.copy()
    while len(expand) != 0:
        v = expand[0]
        # print("Expand:", v, expand)
        expand = expand[1:] # should use better datastructure but not that important
        for u in range(n):
            if assigned[u]:
                continue
            if not isinstance(assignment[u], CrossProductVector):
                continue

            if not (v,u) in G:
                continue

            if assignment[u].v1 == None:
                assignment[u].v1 = v
            elif assignment[u].v2 == None:
                assignment[u].v2 = v
                x,y,z = crossproduct(assignment[assignment[u].v1], assignment[assignment[u].v2])
                assignment[u].x = x
                assignment[u].y = y
                assignment[u].z = z
                expand.append(u)
                assigned[u] = True
                # print(u)
                order.append(u)
            else:
                print("Should not happen")
    # print("order", order)

def getAssignemntHeuristic(G, timeout):
    # get an assignemnt relatively quickly, which can be solved fast very likely
    # Idea:
    #   get vertex with highest degree
    #   get triangle with highest degree if possible, other adjacent vertex with highest degree
    #   add vertex such that common neighbors with already fixed vertices is maximized
    G2 = G + [(j,i) for i,j in G] 
    adjacencyList = [[j for j in range(n) if (i,j) in G] for i in range(n)]
    v1 = None
    maxNeighbors = -1
    for i,neighbors in enumerate(adjacencyList):
        if len(neighbors) > maxNeighbors:
            v1 = i
            maxNeighbors = len(neighbors)

    hasTriangle = False # prefer triangles
    v2 = None
    v3 = None
    maxNeighbors = -1
    for v2_t,v3_t in combinations(adjacencyList[v1], 2):
        if v2_t not in adjacencyList[v3_t]:
            continue
        hasTriangle = True
        if len(adjacencyList[v2_t])+ len(adjacencyList[v3_t]) > maxNeighbors:
            v2 = v2_t
            v3 = v3_t
            maxNeighbors = len(adjacencyList[v2_t])+ len(adjacencyList[v3_t])
    
    if not hasTriangle:
        for v2_t in adjacencyList[v1]:
            if len(adjacencyList[v2_t]) > maxNeighbors:
                v2 = v2_t
                maxNeighbors = len(adjacencyList[v2_t])

    
    assignment = [CrossProductVector() for _ in range(n)]
    expand = []
    assigned = [False for _ in range(n)]

    assignment[v1] = OrthogonalBaseVector(0,0,1)
    assignment[v2] = OrthogonalBaseVector(0,1,0)
    expand.extend([v1,v2])
    assigned[v1] = True
    assigned[v2] = True

    if hasTriangle:
        assignment[v3] = OrthogonalBaseVector(1,0,0)
        expand.append(v3)
        assigned[v3] = True

    assignCrossproductDependencies(G2, assignment, expand, assigned)
    
    while False in assigned:
        notAssigned = [i for i,a in enumerate(assigned) if not a]

        highest = -1
        new_i = None
        for i in notAssigned:
            heuristicValue = sum([ 10 for j in adjacencyList[i] if isinstance(assignment[j], CrossProductVector) and assignment[j].v2 == None and  assignment[j].v1 != None]) + \
                sum([ 1 for j in adjacencyList[i] if isinstance(assignment[j], CrossProductVector) and assignment[j].v1 == None]) # prefer the once leading to new assigned vectors

            if heuristicValue > highest:
                highest = heuristicValue
                new_i = i

        i = new_i# random unassigned vector
        # print("Start", i)
        assignment[i] = StartingVector(Real("x" + str(i)), Real("y" + str(i)), Real("z" + str(i)))
        assigned[i] = True
        assignCrossproductDependencies(G2, assignment, expand = [i], assigned=assigned)
    
    res = solve(G, assignment, timeout=timeout)
    if  str(res) != "unknown":
        return True, str(res) == "sat"

    return False, None



def getOneAssignemnt(G, timeout=2, increment=1):
    # randomized until solved
    G2 = G + [(j,i) for i,j in G] # both directions
    triangles = [(i,j,k) for i,j,k in combinations(range(n), 3) if (i,j) in G2 and (j,k) in G2 and (i,k) in G2]
    currentTimeout = timeout
    while True:
        #  gets one assignment
        assignment = [CrossProductVector() for _ in range(n)]
        expand = []
        assigned = [False for _ in range(n)]
        # assign first triangle (if present)
        if len(triangles) != 0:
            i,j,k = random.choice(triangles) # random triangle
            assignment[i] = OrthogonalBaseVector(0,0,1)
            assignment[j] = OrthogonalBaseVector(0,1,0)
            assignment[k] = OrthogonalBaseVector(1,0,0)
            expand.extend([i,j,k])
            assigned[i] = True
            assigned[j] = True
            assigned[k] = True
            triangleFound = True
        else:
            # if not a triangle at least fix an orthogonal pair
            i,j = random.choice(G) # random edge
            assignment[i] = OrthogonalBaseVector(0,0,1)
            assignment[j] = OrthogonalBaseVector(0,1,0)
            expand.extend([i,j])
            assigned[i] = True
            assigned[j] = True

        assignCrossproductDependencies(G2, assignment, expand, assigned)

        while False in assigned:
            notAssigned = [i for i,a in enumerate(assigned) if not a]
            i = random.choice(notAssigned) # random unassigned vector
            # print("Start", i)
            assignment[i] = StartingVector(Real("x" + str(i)), Real("y" + str(i)), Real("z" + str(i)))
            assigned[i] = True
            assignCrossproductDependencies(G2, assignment, expand = [i], assigned=assigned)
        
        res = solve(G, assignment, timeout=currentTimeout)
        if  str(res) != "unknown":
            return True, str(res) == "sat"
        currentTimeout += increment

def getAssignmentWithOneGuess(G, adjacentToTriangle = False, timeout=10):
    G2 = G + [(j,i) for i,j in G] # both directions
    triangles = [(i,j,k) for i,j,k in combinations(range(n), 3) if (i,j) in G2 and (j,k) in G2 and (i,k) in G2]

    for t in triangles:
        for v in range(n):
            if v in t: continue
            if adjacentToTriangle and not ((v,t[0]) in G2 or  (v,t[1]) in G2  or (v,t[2]) in G2 ): continue
            assignment = [CrossProductVector() for _ in range(n)]
            assigned = [False for _ in range(n)]

            assignment[t[0]] = OrthogonalBaseVector(0,0,1)
            assignment[t[1]] = OrthogonalBaseVector(0,1,0)
            assignment[t[2]] = OrthogonalBaseVector(1,0,0)
            assigned[t[0]] = True
            assigned[t[1]] = True
            assigned[t[2]] = True
            

            # print("Start", i)
            assignment[v] = StartingVector(Real("x" + str(v)), Real("y" + str(v)), Real("z" + str(v)))
            assigned[v] = True
            assignCrossproductDependencies(G2, assignment, expand= list(t) + [v], assigned=assigned)
            if False in assigned:
                continue
            
            '''
            print("Crossproducts:")
            for i,u in enumerate(assignment):
                if isinstance(u, CrossProductVector):
                    print("\t", i, "dependent from", u.v1, u.v2, u.x, u.y, u.z)

            print("Start:")
            for i,u in enumerate(assignment):
                if isinstance(u, StartingVector):
                    print("\t", i, u.x, u.y, u.z)

            print("orthognal:")
            for i,u in enumerate(assignment):
                if isinstance(u, OrthogonalBaseVector):
                    print("\t", i, u.x, u.y, u.z) '''
           
            res = solve(G, assignment, timeout=timeout)
            if str(res) != "unknown":
                return True, str(res) == "sat"
    return False, None

def getAssignmentWithTwoGuess(G, timeout=60):
    G2 = G + [(j,i) for i,j in G] # both directions
    triangles = [(i,j,k) for i,j,k in combinations(range(n), 3) if (i,j) in G2 and (j,k) in G2 and (i,k) in G2]
    vertexPairs = list(combinations(range(n),2))
    p = list(product(triangles, vertexPairs))
    random.shuffle(p)
    for t, vertexPair in p:
            v,u = vertexPair
            if v in t or u in t: continue
            assignment = [CrossProductVector() for _ in range(n)]
            assigned = [False for _ in range(n)]

            assignment[t[0]] = OrthogonalBaseVector(0,0,1)
            assignment[t[1]] = OrthogonalBaseVector(0,1,0)
            assignment[t[2]] = OrthogonalBaseVector(1,0,0)
            assigned[t[0]] = True
            assigned[t[1]] = True
            assigned[t[2]] = True
            

            # print("Start", i)
            assignment[v] = StartingVector(Real("x" + str(v)), Real("y" + str(v)), Real("z" + str(v)))
            assigned[v] = True
            assignment[u] = StartingVector(Real("x" + str(u)), Real("y" + str(u)), Real("z" + str(u)))
            assigned[u] = True
            assignCrossproductDependencies(G2, assignment, expand= list(t) + [v,u], assigned=assigned)
            #if t == (1, 6, 7) and v == 9 and u == 12:
            #    print(f"Unassigned: {[v for v in range(n) if assigned[v] == False]}")
            if False in assigned:
                continue
           
            res = solve(G, assignment, timeout=timeout)
            if str(res) != "unknown":
                return True, str(res) == "sat"
    return False, None


def solveGraphWrapper(G, t0=3, t1=3, t2=3):
    # if not G or maxvert(G) < 9:
    #    return True, True
    #print("attempting one-assignment")
    solved, sat = getAssignmentWithOneGuess(G, timeout=t0)
    # if not solved:
    #    solved, sat = special(G)
    if not solved:
        solved, sat = getAssignemntHeuristic(G, 5)
    if not solved:
        #print("attempting two-assignment")
        solved, sat = getAssignmentWithTwoGuess(G, timeout=t1)
    if not solved:
        #print("attempting two-assignment")
        solved, sat = getOneAssignemnt(G, timeout=t2)
    return solved, sat

def maxvert(G):
    return max(max(u, w) for u, w in G)

def remove_vertex(G, v):
    N = maxvert(G)
    return sorted(tuple(sorted((u, w if w < N else v))) for (u, w) in G if u != v and w != v)

# returns the graph and the sequence of low-degree deletions, starting with v
# TODO it would be great to detect connected components as well
#
# WARNING: leave d set to 1 or this may blow up
# even with d=1, if G-v contains a connected component isomorphic to a path
# this will blow up. Shouldn't happen on expected inputs, but who knows
def remove_vertex_and_low_degree(G, v, d=1):
    H = remove_vertex(G, v)
    seq = [v]
    #print(G)
    #print(f"rem {v}")
    #print(H)
    w = low_deg_vert(H, d)
    while w:
        H = remove_vertex(H, w)
        #print(f"rem {w}")
        #print(H)
        seq.append(w)
        w = low_deg_vert(H, d)
    return H, seq

def low_deg_vert(G, d):
    D = defaultdict(int)
    for u, v in G:
        D[u] += 1
        D[v] += 1
    D = [v for v in D.keys() if D[v] <= d]
    if not D:
        return None
    return max(D)

def deg(G, v):
    return sum(1 for u, w in G if v == u or v == w)
    
if args.verbose >= 1:
    print("Z3 version:", get_version_string ())
# set_param(proof=True)
for line in open(args.graph_file, "r"):

    if not line.strip().startswith("["):
        continue
    G = literal_eval(line)
    # G = [(0, 19), (0, 20), (0, 21), (0, 22), (1, 16), (1, 17), (1, 18), (1, 22), (2, 14), (2, 15), (2, 18), (2, 21), (3, 12), (3, 13), (3, 18), (3, 20), (4, 10), (4, 11), (4, 17), (4, 19), (5, 9), (5, 11), (5, 13), (5, 15), (6, 7), (6, 8), (6, 14), (6, 16), (7, 8), (7, 10), (7, 13), (8, 9), (8, 12), (9, 11), (9, 12), (10, 13), (10, 17), (11, 19), (12, 20), (13, 15), (14, 15), (14, 16), (16, 22), (17, 18), (18, 21), (19, 21), (20, 22)]
    # G = []
    # print("Yes" if getAssignmentWithOneGuess(G) else "No")
    # print("Yes" if getAssignmentWithTwoGuess(G) else "No")
    # print("-----------------")
    print(f"Checking embeddability of {G}")
    solved, sat = solveGraphWrapper(G)
    if not solved:
        print("Could not determine whether the graph is embeddable")
    elif args.minimize:
        print("Will now attempt to minimize the graph by removing vertices while preserving unembeddability")
        progress = True
        attempts = []
        verts_removed = []
        renaming = {v : v for v in range(n)}
        queries = 0
        t = n
        known_sat = set()
        while progress:
            progress = False
            candidates = sorted(set(range(n)) - known_sat)
            #candidates = sorted(set(range(n)) - known_sat, key=lambda v: (deg(G, v), v))
            #random.shuffle(candidates)
            q = n
            for v in candidates:
                if args.verbose >= 1:
                    print(f"  trying to remove the vertex {v}")
                #H = remove_vertex(G, v)
                H, seq = remove_vertex_and_low_degree(G, v)
                attempts.append(renaming[v])
                n = q - len(seq)
                solved, sat = solveGraphWrapper(H)
                queries += 1
                n = q
                if solved:
                    if not sat:
                        # must restore and decrement n by 1 to obtain a renaming
                        G = H
                        progress = True
                        print("    removing", end="")
                        for w in seq:
                            n -= 1
                            print(f" {renaming[w]}", end="")
                            verts_removed.append(renaming[w])
                            renaming[w] = renaming[n]
                        print()
                        break
                    else:
                        if args.verbose >= 1:
                            print(f"    cannot remove {v}")
                        known_sat.add(v)
        if verts_removed:
            print(f"Minimized unembeddable graph (-{len(verts_removed)} vertices after {queries} queries):")
            print(sorted(tuple(sorted((renaming[u], renaming[v]))) for u, v in G))
            print(f"The removed vertices: {verts_removed}")
        else:
            print(f"No vertices removed. Graph is minimal unembeddable ({queries} queries)")
        print(f"The sequence of vertex removal attempts: {attempts}")
        known_sat_orig = {renaming[v] for v in known_sat}
        if len(known_sat) + len(verts_removed) == t:
            print("Reason for stopping: all proper induced subgraphs are embeddable")
            print("Attempting to remove edges")
            i = 0
            edges_removed = 0
            # TODO do not just remove edges greedily, but explore all options
            while i < len(G):
                u, v = G[i]
                if deg(G, u) >= 3 and deg(G, v) >= 3:
                    if args.verbose >= 1:
                        print(f"trying to remove the edge ({renaming[u]} {renaming[v]})")
                    H = G[:i] + G[i+1:]
                    solved, sat = solveGraphWrapper(H, t1=60)
                    queries += 1
                    if solved:
                        if not sat:
                            print(f"removing the edge ({renaming[u]} {renaming[v]})")
                            G = H
                            i -= 1
                            edges_removed += 1
                        else:
                            if args.verbose >= 1:
                                print(f"cannot remove the edge ({renaming[u]} {renaming[v]})")

                i += 1
            if edges_removed > 0:
                print(f"Additionally removed {edges_removed} edges")
            else:
                print("No edges removed")
            print("--------------------------")
            print("The final minimized graph:")
            print(sorted(tuple(sorted((renaming[u], renaming[v]))) for u, v in G))
        else:
            print(f"Reason for stopping: could not determine embeddability for removals of {set(range(t)) - set(verts_removed) - set(known_sat_orig)}")
        n = t



# ----------- not used; can be used as input to solve it with redlog ---------------
def printInRedlog(G, a):
    # solve giving a vector assignment and the graph
    print("Base:", [i for i,v in enumerate(a) if isinstance(v, OrthogonalBaseVector)])
    print("Start:", [i for i,v in enumerate(a) if isinstance(v, StartingVector)])
    print("Crossproduct:", [i for i,v in enumerate(a) if isinstance(v, CrossProductVector)])
    sys.stdout.flush()

    # for i,v in enumerate(a):
    #    print(i, str(v.x), str(v.y), str(v.z))

    EQ = "="
    NEQ = "neq"
    OR = "or"
    AND = "and"

    useDisjunctions = True

    print("load_package redlog; rlset R;")

    print("phi := ")

    print("0 = 0 ")
    for v in a:
        if isinstance(v, StartingVector):
            print(AND, str(v.x * v.x + v.y * v.y + v.z * v.z), EQ, 1)
            print(v.z, ">= 0") # z must be positive

        # if useDisjunctions:
        #     s.add(Or(v.x != 0, v.y != 0, v.z != 0)) # already handled in line where ensuring that they are not collinear

    # all orthogonal relations if not specified by crossproduct or start
    for u,v in G:
        print(AND,  a[u].x * a[v].x + a[u].y * a[v].y + a[u].z * a[v].z, EQ, 0)
        
        # decided to skip this because for crossproduct both most be known for crossproduct only a sinlge one
        # if isinstance(u, CrossProductVector) and (a[u].v1 == v or a[u].v2 == v):
        #    continue

        #if isinstance(v, CrossProductVector) and (a[v].v1 == u or a[v].v2 == u):
        #    continue

    # pairwise distinct (i.e., not collinear)
    for u,v in combinations(range(n),2):
        if (u,v) in G:
            continue
        x,y,z = crossproduct(a[u], a[v])
        if useDisjunctions:
            print(AND, "(", x, NEQ, 0, OR, y, NEQ, 0, OR, z, NEQ, 0, ")" )
        else:
            pass # TODO
            # s.add(x*x + y * y + z * z > 0)
    print(";")

    print("rlqe(rlex(phi));")
