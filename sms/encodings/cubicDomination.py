# Script for trying to glue graphs together trying to get a cubic graph with large domination number

from pysms.graph_builder import *

# add potential arguments here
parser = getDefaultParser()
parser.add_argument(
    "--interface",
    type=int,
    nargs="+",
    required=True,
    help="Description of the interface given by several integer pairs. The first gives the degree to the part to construct, whilst the second gives the degree within the interface. \
                    For example '1 1 1 1 2 0' gives an interface with 3 vertices. (Only for up to 3 vertices, otherwise not uniquely defined by this numbers)",
)

parser.add_argument(
    "--min-domination",
    type=int,
    nargs="+",
    help="The sequence encodes different options of domination and the minimum domination number for each case. \
        We use 0 to encode that a vertex in the interface is in the dominating set, 1 to encode that it most be dominated by the constructed part and 2 that it is dominated by the other part.\
            For example '0 1 2 10 2 1 1 9' encodes that if the three vertices in the interface are dominated like described, then the domination number must be >= 10 and >= 9 respectively."
)


class DominationEncodingBuilder(GraphEncodingBuilder):
    def __init__(
        self,
        n,
        interface,
        staticInitialPartition=False,
    ):
        super().__init__(n, staticInitialPartition=staticInitialPartition)

        assert len(interface) % 2 == 0

        self.interface = [tuple(interface[i : i + 2]) for i in range(0, len(interface), 2)]
        print(self.interface)
        i = len(self.interface)
        assert i <= 3
        for d in self.interface:
            assert d[0] >= 1 and d[0] <= 2
            assert d[1] >= 0 and d[1] <= 1
            assert d[0] + d[1] <= 2

        s = "1 " * i
        self.paramsSMS["initial-partition"] = s + str(self.n - i)  # TODO refine if interface has symmetries
        self.paramsSMS["domination-connectedness"] = f"1 {i}"  # first gives the number of interfaces, second the size of each interface (Already implemented for more general case)

        self.add_degree_constraints()

    def add_degree_constraints(self):
        # degrees of vertices not in interface
        cubicVertices = list(range(len(self.interface), self.n))
        self.degreeBounds(cubicVertices, lower=3, upper=3)

        # degrees of vertices between interface and other
        for v, d in enumerate(self.interface):
            self.counterFunction([self.var_edge(v, u) for u in cubicVertices], d[0], d[0], d[0])

        # fix edges within the interface
        neighboringVertices = [v for v, d in enumerate(self.interface) if d[1] == 1]  # vertices with one neighbor in the interface
        assert len(neighboringVertices) == 0 or len(neighboringVertices) == 2
        for u, v in combinations(list(range(len(self.interface))), 2):
            if u in neighboringVertices and v in neighboringVertices:
                self.append([self.var_edge(u, v)])  # edge between degree 1 vertices
            else:
                self.append([-self.var_edge(u, v)])  # no other edge in the interface


if __name__ == "__main__":
    args, forwarding_args = parser.parse_known_args()
    g = DominationEncodingBuilder(args.vertices, interface=args.interface, staticInitialPartition=args.static_partition)
    g.add_constraints_by_arguments(args)
    g.solveArgs(args, forwarding_args)
