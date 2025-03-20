'''
    Calculate all vertex degree partitions for k-uniform hyper edges with m edges and n vertices
'''

k = 5
m = 17
n = 11

allPartitions = []

def rec(mindegree, remainingdegree, v, partition):
    if v == n:
        if remainingdegree == 0:
            allPartitions.append(partition)
        return

    if mindegree * (n - v) > remainingdegree:
        return

    if m * (n - v) < remainingdegree:
        return

    for d in range(mindegree, m + 1): # m is maximum degree for vertex
        rec(d, remainingdegree - d, v + 1, [x for x in partition] + [d])


rec(3, m * k, 0, [])


print(len(allPartitions))

from collections import Counter

for p in allPartitions:

    c = Counter(p)
    # print(c)
    d = sorted(list(c.keys())) # sorted degrees
    o = [c[x] for x in d]
    print(" ".join(map(str, o)), "  with degrees  " ,  " ".join(map(str, d)))
