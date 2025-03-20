#!/home1/peitl/bin/sage

import sys
from sage.graphs.independent_sets import IndependentSets
for line in sys.stdin:
    if line.startswith("["):
        G = Graph(eval(line))
    else:
        G = Graph(line)
    edgelist = "[" + ",".join(f"({u},{v})" for (u,v,_) in G.edges()) + "]"
    I = IndependentSets(G, maximal=true)
    indep_list = "[" +  ",".join(("[" + ",".join(map(str, S)) + "]") for S in I) + "]"
    print(f"{edgelist};{indep_list}")
