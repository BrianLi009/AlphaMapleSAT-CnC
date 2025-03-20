from pysms.graph_builder import *
from itertools import *


class TriangleFreeSlimBuilder(GraphEncodingBuilder):
    def __init__(self, n, staticInitialPartition=False):
        super().__init__(n, staticInitialPartition=staticInitialPartition)
        self.paramsSMS["min-chromatic-number"] = 6
        self.paramsSMS["coloring-algo"] = 2
        self.paramsSMS["cutoff"] = 20000

        self.mtf()
        self.noSubsumingNeighborhoods()


    
    def add_constraints_by_arguments(self, args):
        super().add_constraints_by_arguments(args)
        
        if args.slim:
            import graph_28216
            self.slimSingleStep(40, args.slim, graph_28216.G, induced=False)


if __name__ == "__main__":
    parser = getDefaultParser()
    parser.add_argument("--slim", type=int, help="Use SLIM to try to improve the graph")
    args = parser.parse_args()
    if not args.deltaLow:
        args.deltaLow = 5
    g = TriangleFreeSlimBuilder(args.vertices, staticInitialPartition=args.static_partition)
    g.add_constraints_by_arguments(args)
    g.solveArgs(args, forwarding_args="")
