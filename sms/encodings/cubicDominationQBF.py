"""
    Script for trying to glue graphs together trying to get a cubic graph with large domination number.

    Now only for disjoint interfaces

    TODO: that would be my suggestion for a general interface, currently implementation only for 1 interface.
    Currently, we use a very simple way to describe the interface by a list of integers.
    The first gives the size of each interface (the all have the same number of vertices),
    then we have triples, where the first indicates the vertex, the second the degree to the constructed component and the last the degree within the interface.

    For example "3 x 2 0 y 1 1 z 1 1 x' 1 0 y' 1 0 z' 2 0" where x,y,z,x',y',z' \in [n], means that all interfaces have size 3, 
    x,y,z and x',y',z' are interface where x has degree 2 to the constructed part (excluding the interface x,y,z itself but to other interfaces),
    and x has zero neighbors in the interface but y and z have one (i.e., in this case they are adjacent).

    For the domination number one specifies how each vertex in the interface (in the order of the interface) is dominated
    using 0 for in the dominating set, 1 for dominated by the constructed part and 2 for dominated by the other part.
    And finally the minimum domination number for this case.
    For example if we have two interfaces with size 3 and '0 1 2 2 1 1 10 ' then if the first interface is dominated by  '0 1 2' and the second by '2 1 1' then the domination number must be at least 10.
    If we want a second case, we can just concatenate the arguments. 
"""


from pysms.qcir_graph_builder import *

# add potential arguments here
parser = getDefaultParser()
parser.add_argument(
    "--interface",
    type=int,
    nargs="+",
    help="Description of the interface given by several integer pairs. The first gives the degree to the part to construct, whilst the second gives the degree within the interface. \
                    For example '1 1 1 1 2 0' gives an interface with 3 vertices. (Only for up to 3 vertices, otherwise not uniquely defined by this numbers)",
)

parser.add_argument(
    "--num-interfaces",
    type=int,
    default=1,
    help="Number of interfaces",
)



# parser.add_argument(
#     "--interfaces",
#     type=int,
#     nargs="+",
#     required=True,
#     help="See top of file for description of the interface.",
# )

parser.add_argument(
    "--min-domination",
    type=int,
    nargs="+",
    help="The sequence encodes different options of domination and the minimum domination number for each case. \
        We use 0 to encode that a vertex in the interface is in the dominating set, 1 to encode that it most be dominated by the constructed part and 2 that it is dominated by the other part.\
            For example '0 1 2 10 2 1 1 9' encodes that if the three vertices in the interface are dominated like described, then the domination number must be >= 10 and >= 9 respectively.",
)


class DominationEncodingBuilder(GraphEncodingBuilder):
    def __init__(self, n, interface, min_domination, num_interfaces):
        super().__init__(n)

        self.num_interfaces = num_interfaces

        assert interface # at least one interface must be specified

        if interface:

            assert len(interface) % 2 == 0
            assert len(interface) % num_interfaces == 0

            # self.interface = [tuple(interface[i : i + 2]) for i in range(0, len(interface), 2)]
            # # print(self.interface)
            # interfaceSize = len(self.interface)
            # assert interfaceSize <= 3
            # for d in self.interface:
            #     assert d[0] >= 1 and d[0] <= 2
            #     assert d[1] >= 0 and d[1] <= 1
            #     assert d[0] + d[1] <= 2

            self.interfaces = []
            self.interfaceSize = len(interface) // num_interfaces // 2
            interfaceSize = self.interfaceSize
            for i in range(0, len(interface), self.interfaceSize * 2):
                self.interfaces.append([tuple(interface[i + j : i + j + 2]) for j in range(0, self.interfaceSize * 2, 2)])

            print(self.interfaces)



        # if interfaces:
        #     interfaceSize = interfaces[0]
        #     self.interfaces = []
        #     interfaces = interfaces[1:]
        #     assert len(interfaces) % (interfaceSize * 3) == 0
        #     for i in range(0, len(interfaces), interfaceSize * 3):
        #         self.interfaces.append([tuple(interfaces[i + j : i + j + 3]) for j in range(0, interfaceSize * 3, 3)])

        #     print(self.interfaces)

        #     interfaceVertices = set()
        #     for i in self.interfaces:
        #         for vertexDescription in i:
        #             interfaceVertices.add(vertexDescription[0])
        #     print(interfaceVertices)
        #     exit()


        s = "1 " * interfaceSize * num_interfaces
        self.paramsSMS["initial-partition"] = s + str(self.n - interfaceSize * num_interfaces)  # TODO refine if interface has symmetries
        self.paramsSMS["domination-connectedness"] = f"{num_interfaces} {interfaceSize}"  # first gives the number of interfaces, second the size of each interface (Already implemented for more general case)

        self.add_degree_constraints()

        if min_domination:
            verticesInInterface = self.interfaceSize * self.num_interfaces
            sizeOfconfig = verticesInInterface + 1
            assert len(min_domination) % sizeOfconfig == 0
            for i in range(len(min_domination) // sizeOfconfig):
                spec = min_domination[i * sizeOfconfig : (i + 1) * sizeOfconfig - 1]
                minDomination = min_domination[(i + 1) * sizeOfconfig - 1]
                # print(spec, minDomination)

                self.add_domination_constraints(spec, minDomination)

    def add_degree_constraints(self):
        # degrees of vertices not in interface
        numInterfaceVertices = self.interfaceSize * self.num_interfaces
        cubicVertices = list(range(numInterfaceVertices, self.n))

        outputGate = AndGate(self.id(), [])
        degree = 3
        for u in cubicVertices:
            seqCounter([self.var_edge(u, v) for v in self.V if v != u], degree, self, outputGate, atMost=degree, atLeast=degree)

        def getInterfaceVertex(x, v):
            # x is the index of the interface, v the index of the vertex in the interface
            return v + x * self.interfaceSize
        
        # degrees of vertices between interface and other
        for x, interface in enumerate(self.interfaces):
            for v, d in enumerate(interface):
                otherVertices = set(self.V) - set([getInterfaceVertex(x,u) for u in range(self.interfaceSize)])
                seqCounter([self.var_edge(getInterfaceVertex(x,v), u) for u in otherVertices], d[0], self, outputGate, atMost=d[0], atLeast=d[0])

        # fix edges within the interfaces
        for x, interface in enumerate(self.interfaces):
            neighboringVertices = [v for v, d in enumerate(interface) if d[1] == 1]  # vertices with one neighbor in the interface
            assert len(neighboringVertices) == 0 or len(neighboringVertices) == 2
            for u, v in combinations(list(range(self.interfaceSize)), 2):
                u = getInterfaceVertex(x,u)
                v = getInterfaceVertex(x,v)
                print(u,v)
                if u in neighboringVertices and v in neighboringVertices:
                    outputGate.appendToInput(AndGate(self.id(), [self.var_edge(u,v)]))
                else:
                    outputGate.appendToInput(AndGate(self.id(), [-self.var_edge(u,v)]))  # no other edge in the interface

        self.addExistentialGate(outputGate)

    def add_domination_constraints(self, dominationOfInteface, minDomination):
        dominationVars = [self.id() for _ in self.V]  # indicate whether a vertex is in the dominating set

        validDominatingSet = AndGate(self.id(), [])
        # fix vertices in interface which are present for sure
        for v, d in enumerate(dominationOfInteface):
            if d == 0:
                validDominatingSet.appendToInput(dominationVars[v])

        # ensure that dominating set
        for v in self.V:
            if v < self.interfaceSize * self.num_interfaces and dominationOfInteface[v] == 2:
                continue  # dominated from other side
            if v < self.interfaceSize * self.num_interfaces and dominationOfInteface[v] == 0:
                continue  # already in dominating set
            validDominatingSet.appendToInput(OrGate(self.id(), [dominationVars[v]] + [AndGate(self.id(), [self.var_edge(v, u), dominationVars[u]]) for u in self.V if u != v]))

        # limit dominiation number to minDomination - 1
        seqCounter(dominationVars, minDomination - 1, self, validDominatingSet, atMost=minDomination - 1)

        g = NegatedGate(validDominatingSet)
        self.addUniversalGate(g)


if __name__ == "__main__":
    args, forwarding_args = parser.parse_known_args()
    if forwarding_args:
        print("WARNING: Unknown arguments for python script which are forwarded to SMS:", forwarding_args, file=stderr)
    b = DominationEncodingBuilder(args.vertices, interface=args.interface, min_domination=args.min_domination, num_interfaces=args.num_interfaces)
    b.add_constraints_by_arguments(args)

    if args.print_cnf:
        with open(args.print_cnf, "w") as cnf_fh:
            b.print_dimacs(cnf_fh)
    if args.print_qcir:
        with open(args.print_qcir, "w") as qcir_fh:
            b.print_qcir(qcir_fh)
    else:
        b.solveArgs(args, forwarding_args)
