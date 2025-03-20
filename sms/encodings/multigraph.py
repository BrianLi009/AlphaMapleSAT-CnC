from pysms.graph_builder import *
from itertools import *


def getMultiParser():
    parser = getDefaultParser()
    parser.add_argument("--disjoint", action="store_true", help="Graphs must be pairwise disjoint")
    parser.add_argument("--earthmoon", type=int, help="Criteria related to earth moon with given chromatic number; doesn't check thickness 2")
    parser.add_argument("--earthmoonLight", action="store_true", help="Search for a biplanar graph with independence number < 3")

    parser.add_argument("--earthmoon_candidate1", action="store_true", help="Try to find decomposition of C5[4, 4, 4, 4, 3] into two planar graphs")
    parser.add_argument("--earthmoon_candidate2", action="store_true", help="Try to find decomposition of C7[4, 4, 4, 4, 4, 4, 4] into two planar graphs")
    parser.add_argument("--earthmoon_candidate3", action="store_true", help="Try to find decomposition of C7[4, 4, 4, 4, 4, 4, 3] into two planar graphs")
    parser.add_argument("--earthmoon_candidate4", action="store_true", help="Try to find decomposition of C7[1,8,1,8,1,8,1] into two planar graphs")
    parser.add_argument("--earthmoon_candidate5", action="store_true")
    parser.add_argument("--earthmoon_candidate6", action="store_true")

    return parser


class MultiGraphBuilder(GraphEncodingBuilder):
    def __init__(self, n, multiGraph=2, directed=False, staticInitialPartition=False, underlyingGraph=False):
        super().__init__(n, directed, multiGraph, staticInitialPartition, underlyingGraph)

    def disjoint(self):
        for i, j in combinations(range(args.multigraph), 2):
            for u, v in combinations(self.V, 2):
                self.append([-self.var_edge_multi(i, u, v), -self.var_edge_multi(j, u, v)])

    def add_constraints_by_arguments(self, args):
        super().add_constraints_by_arguments(args)

        if args.disjoint:
            self.disjoint()

        if args.earthmoon:
            self.paramsSMS["thickness2multi"] = 5
            self.paramsSMS["frequency"] = 30
            V = self.V

            # first graph is union of both
            G = self.varEdgeMultiTable[0]
            G1 = self.varEdgeMultiTable[1]
            G2 = self.varEdgeMultiTable[2]

            for v, u in combinations(V, 2):
                self.CNF_OR_APPEND([G1[u][v], G2[u][v]], G[u][v])  # lower triangle gives union of both
                self.append([-G1[u][v], -G2[u][v]])  # disjoint

            # explicitely ensure at least a certain minimum chromatic number
            chi = args.earthmoon
            from more_itertools import set_partitions

            for coloring in set_partitions(self.V, chi - 1):
                # print(x)
                clause = []  # monochromatic edge
                for color in coloring:
                    clause.extend([G[v][u] for v, u in combinations(color, 2)])
                self.append(clause)
            # self.paramsSMS["chi"] = args.earthmoon # set min chromatic number

            # w.l.o.g.: first graph is triangulation and second doesn't have more edges than triangulation
            if True:
                # euler and F=E*2/3 to get number of edges: V + F - E = 2; F = E * 2 / 3 -> V - 2= 1/3 E -> E = 3(V - 2)
                edgesInTriangulation = 3 * (len(V) - 2)
                self.counterFunction([G1[i][j] for i, j in combinations(V, 2)], countUpto=edgesInTriangulation, atMost=edgesInTriangulation, atLeast=edgesInTriangulation)
                self.counterFunction([G2[i][j] for i, j in combinations(V, 2)], countUpto=edgesInTriangulation, atMost=edgesInTriangulation)

            # graphs given by each direction (upper and lower triangular) must be planar
            if False:
                planar_encoding_schnyder(V, lambda v, u: G1[v][u], self, self)
                planar_encoding_schnyder(V, lambda v, u: G2[v][u], self, self)

            # mindegree on undirected version
            for i in V:
                self.counterFunction([G[i][j] for j in V if j != i], countUpto=args.earthmoon - 1, atLeast=args.earthmoon - 1)

            # forbid some trivial cases
            for A in combinations(V, 5):
                self.append([-G1[i][j] for i, j in combinations(A, 2)])
                self.append([-G2[i][j] for i, j in combinations(A, 2)])

            for A in combinations(V, 6):
                for B in combinations(A, 3):
                    if min(A) not in B:
                        continue
                    self.append([-G1[i][j] for i in set(A) - set(B) for j in B])
                    self.append([-G2[i][j] for i in set(A) - set(B) for j in B])

        if args.earthmoonLight:
            self.paramsSMS["thickness2multi"] = 5
            self.paramsSMS["cutoff"] = 30000
            self.paramsSMS["frequency"] = 30
            V = self.V

            # first graph is union of both
            G = self.varEdgeMultiTable[0]
            G1 = self.varEdgeMultiTable[1]
            G2 = self.varEdgeMultiTable[2]

            for v, u in combinations(V, 2):
                self.CNF_OR_APPEND([G1[u][v], G2[u][v]], G[u][v])  # lower triangle gives union of both
                self.append([-G1[u][v], -G2[u][v]])  # disjoint

            self.maxIndependentSet(2)

            # w.l.o.g.: first graph is triangulation and second doesn't have more edges than triangulation
            if True:
                # euler and F=E*2/3 to get number of edges: V + F - E = 2; F = E * 2 / 3 -> V - 2= 1/3 E -> E = 3(V - 2)
                edgesInTriangulation = 3 * (len(V) - 2)
                self.counterFunction([G1[i][j] for i, j in combinations(V, 2)], countUpto=edgesInTriangulation, atMost=edgesInTriangulation, atLeast=edgesInTriangulation)
                self.counterFunction([G2[i][j] for i, j in combinations(V, 2)], countUpto=edgesInTriangulation, atMost=edgesInTriangulation)

        if args.earthmoon_candidate1 or args.earthmoon_candidate2 or args.earthmoon_candidate3 or args.earthmoon_candidate4 or args.earthmoon_candidate5 or args.earthmoon_candidate6:
            partition = []
            noCycle = False
            E = []
            if args.earthmoon_candidate1:
                # C5[4, 4, 4, 4, 3]
                assert self.n == 19
                assert args.multigraph == 3
                partition = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18]]
                # partition = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16], [17, 18]]

            if args.earthmoon_candidate2:
                # C_7 where each vertex is replaced by K4s and joining neighborhoods
                assert self.n == 28
                assert args.multigraph == 3
                for i in range(7):
                    partition.append(list(range(4 * i, 4 * (i + 1))))

            if args.earthmoon_candidate3:
                # C_7 where each vertex is replaced by K4s and joining neighborhoods exept the last with a K_3
                assert self.n == 27
                assert args.multigraph == 3
                for i in range(6):
                    partition.append(list(range(4 * i, 4 * (i + 1))))
                partition.append([24, 25, 26])

            if args.earthmoon_candidate4:
                assert self.n == 28
                assert args.multigraph == 3
                partition = [[0], list(range(1, 1 + 8)), [9], list(range(10, 10 + 8)), [18], list(range(19, 19 + 8)), [27]]

            if args.earthmoon_candidate5:
                assert self.n == 24
                assert args.multigraph == 3
                partition = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]
                # for i in range(3):
                #    partition.append(list(range(5 * i, 5 * (i + 1))))
                noCycle = True

            if args.earthmoon_candidate6:
                assert self.n == 19
                assert args.multigraph == 3
                for i in range(5):
                    partition.append(list(range(3 * i, 3 * (i + 1))))
                partition.append([15, 16, 17, 18])

                for i in partition[0]:
                    for j in partition[-2]:
                        E.append((i, j))

            # E.extend([(0,16), (1,16), (2,16), (3,16)])

            if False:
                # inflated Groetsch graph
                G = [
                    (0, 1, None),
                    (0, 2, None),
                    (0, 3, None),
                    (0, 4, None),
                    (0, 5, None),
                    (1, 7, None),
                    (1, 10, None),
                    (2, 6, None),
                    (2, 8, None),
                    (3, 7, None),
                    (3, 9, None),
                    (4, 8, None),
                    (4, 10, None),
                    (5, 6, None),
                    (5, 9, None),
                    (6, 7, None),
                    (6, 10, None),
                    (7, 8, None),
                    (8, 9, None),
                    (9, 10, None)
                ]
                G = [(u, v) for u, v, _ in G]

                for i in range(11):
                    partition.append(list(range(3 * i, 3 * (i + 1))))

                E = []
                for u,v in G:
                    for i in partition[u]:
                        for j in partition[v]:
                            E.append((i,j))
                for P in partition:
                    for i, j in combinations(P, 2):
                        E.append((i, j))


            for P in partition:
                for i, j in combinations(P, 2):
                    E.append((i, j))
            for i in range(len(partition) - 1):
                for u in partition[i]:
                    for v in partition[i + 1]:
                        E.append((u, v))

            if not noCycle:  # cycle
                for u in partition[0]:
                    for v in partition[-1]:
                        E.append((u, v))

            self.paramsSMS["thickness2multi"] = 5
            self.paramsSMS["frequency"] = 30
            V = self.V

            # first graph is union of both
            G = self.varEdgeMultiTable[0]
            G1 = self.varEdgeMultiTable[1]
            G2 = self.varEdgeMultiTable[2]

            # print(E)
            for i, j in E:
                self.append([G[i][j]])

            for i, j in combinations(self.V, 2):
                if (i, j) in E:
                    self.append([G[i][j]])
                else:
                    self.append([-G[i][j]])

            for v, u in combinations(V, 2):
                self.CNF_OR_APPEND([G1[u][v], G2[u][v]], G[u][v])  # lower triangle gives union of both
                self.append([-G1[u][v], -G2[u][v]])  # disjoint

            if True:
                # euler and F=E*2/3 to get number of edges: V + F - E = 2; F = E * 2 / 3 -> V - 2= 1/3 E -> E = 3(V - 2)
                edgesInTriangulation = 3 * (len(V) - 2)
                self.counterFunction([G1[i][j] for i, j in combinations(V, 2)], countUpto=edgesInTriangulation, atMost=edgesInTriangulation)
                self.counterFunction([G2[i][j] for i, j in combinations(V, 2)], countUpto=edgesInTriangulation, atMost=edgesInTriangulation)

            if True:
                # forbid some trivial cases
                for A in combinations(V, 5):
                    self.append([-G1[i][j] for i, j in combinations(A, 2)])
                    self.append([-G2[i][j] for i, j in combinations(A, 2)])
            if False:
                for A in combinations(V, 6):
                    for B in combinations(A, 3):
                        if min(A) not in B:
                            continue
                        self.append([-G1[i][j] for i in set(A) - set(B) for j in B])
                        self.append([-G2[i][j] for i in set(A) - set(B) for j in B])

            self.paramsSMS["thickness2multi"] = "5"
            self.paramsSMS["initial-partition"] = " ".join(map(str, map(len, partition)))


args = getMultiParser().parse_args()
b = MultiGraphBuilder(args.vertices, multiGraph=args.multigraph, directed=args.directed, staticInitialPartition=args.static_partition, underlyingGraph=args.underlying_graph)
b.add_constraints_by_arguments(args)
b.solveArgs(args)
