import sys
from ast import literal_eval
i=0
out=sys.argv[1]
for line in sys.stdin:
    i += 1
    if line.startswith("["):
        G = Graph(literal_eval(line))
    else:
        G = Graph(line)
    n = G.num_verts()
    with open(f"{out}/{i}.lad", "w") as fh:
        print(n, file=fh)
        for v in sorted(G.vertices()):
            print(f"{G.degree(v)} {' '.join(map(lambda x: str(x), sorted(G.neighbors(v))))}", file=fh)
