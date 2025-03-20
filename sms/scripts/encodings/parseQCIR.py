from argparse import *

parser = ArgumentParser()
parser.add_argument("--file", "-f", type=str, required=True, help="File containing the encoding")
parser.add_argument("--vertices", "-v", type=int, default=2, help="Number of vertices for SMS (default 2 means no SMS)")
parser.add_argument("--temp-file", type=str, default="./", help="Directory where all the temporary files are stored")
parser.add_argument("--args-SMS", type=str, default="", help="Arguments forwarded to SMS")
parser.add_argument("--timeout", "-t", type=str, help="Timeout passed to `timeout`")
parser.add_argument("--all-graphs", "-a", action="store_true", help="Run SMS with --all-graphs")

args = parser.parse_args()

file = args.file

encGate = []  # The encoding of the gate as a CNF using teitsin (without polarities)
with open(file) as f:

    seenGatesAndVariables = set()
    if not f.readline().startswith("#QCIR-G14"):
        print("ERROR")
        exit(1)

    existential = []
    # free variables part
    line = f.readline()
    if line.startswith("free("):
        line = line[len("free(") : -2]
        existential = [int(num) for num in line.split(",")]

        seenGatesAndVariables |= set(existential)

        line = f.readline()

    # existential part (optional)
    if line.startswith("exists("):
        line = line[len("exists(") : -2]
        existential += [int(num) for num in line.split(",")]

        seenGatesAndVariables |= set(existential)

        line = f.readline()

    # print("Existential:", existential)

    # universal part
    if line.startswith("forall("):
        line = line[len("forall(") : -2]
        forall = [int(num) for num in line.split(",")]
        # print("Forall: ", forall)

        seenGatesAndVariables |= set(forall)

        line = f.readline()

    # output gate
    if not line.startswith("output("):
        print("ERROR")
        exit(1)
    outputgate = int(line[len("output(") : -2])
    # print("Ouput:", outputgate)

    # parse gates
    for line in f:
        gate, inputs = line.split("=")
        gate = int(gate)
        inputs = inputs.strip()
        gateType = None
        if inputs.startswith("or"):
            gateType = "or"
            inputs = inputs[3:-1]
        elif inputs.startswith("and"):
            gateType = "and"
            inputs = inputs[4:-1]
        else:
            print("ERROR")
            exit(1)
        # print("Line:", inputs)
        if inputs == "":
            print("Warning: empty gate")
            if gateType == "or":
                encGate.append([-gate])
            if gateType == "and":
                encGate.append([+gate])
            continue

        # print("ASDF", inputs)
        inputs = [int(num) for num in inputs.split(",")]

        # TODO check whether they are already defined
        absInputs = set(map(abs, inputs))
        if not absInputs <= seenGatesAndVariables:
            raise Exception(f"The gates must be a topological ordering; gate {gate} uses undefined variable or gate {absInputs - seenGatesAndVariables}")
        assert gate > 0
        seenGatesAndVariables.add(gate)

        if gateType == "or":
            encGate.append([-gate] + inputs)
            for i in inputs:
                encGate.append([+gate, -i])

        if gateType == "and":
            encGate.append([+gate] + [-i for i in inputs])
            for i in inputs:
                encGate.append([-gate, +i])

        """
        inputs = [id2gatefun(int(num)) for num in inputs]
        print(inputs)
        if gateType == "or":
            id2gate[gate] = OrGate(gate, inputs)

        if gateType == "and":
            id2gate[gate] = AndGate(gate, inputs) """

import os

# write to DIMACS and solve
output = f"./encExistential{os.getpid()}.enc"
maxVariable = max(abs(value) for row in encGate for value in row)
with open(output, "w") as f:
    print(f"p cnf {maxVariable} {len(encGate) + 1}", file=f)
    print(outputgate, 0, file=f)
    for c in encGate:
        print(" ".join(str(x) for x in c), 0, file=f)

# for universal part it should be negated I think
outputgateUniversal = maxVariable + 1
outputUniversal = f"./encUniversal{os.getpid()}.enc"

with open(outputUniversal, "w") as f:
    print(f"p cnf {maxVariable + 1} {len(encGate) + 1}", file=f)
    print(outputgateUniversal, 0, file=f)
    print(outputgateUniversal, outputgate, 0, file=f)
    print(-outputgateUniversal, -outputgate, 0, file=f)
    for c in encGate:
        print(" ".join(str(x) for x in c), 0, file=f)

assumptionFile = f"./assumptions{os.getpid()}.enc"
with open(assumptionFile, "w") as f:
    print(" ".join(map(str, existential)), file=f)

sms_command = f"time smsg {args.args_SMS} -v {args.vertices} --dimacs {output} --forallQCIR {outputUniversal} --forallQCIRAssumptions {assumptionFile}  --print-stats"
if args.all_graphs:
    sms_command += " --all-graphs"
if args.timeout:
    sms_command = f"timeout {args.timeout} " + sms_command

os.system(sms_command)
os.system(f"rm {output}")
os.system(f"rm {outputUniversal}")
os.system(f"rm {assumptionFile}")
