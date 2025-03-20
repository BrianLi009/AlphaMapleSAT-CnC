#!/usr/bin/env python3
# takes one CNF file (must come first) and an arbitrary number of QCIR files, and conjoins them, sharing the free variables

def andgate(gout, gins):
    return f"{gout}=and({','.join(map(str, gins))})"

def orgate(gout, gins):
    return f"{gout}=or({','.join(map(str, gins))})"

def mkgate(gtype, gout, gins):
    return f"{gout}={gtype}({','.join(map(str, gins))})"

def parse_gate(gateline):
    head_end = line.index("=")
    head = gateline[:head_end].strip()

    gtype_end = line.index("(", head_end+1)
    gtype = gateline[line.index("=")+1:line.index("(")].strip()

    body_end = line.rindex(")")
    body = gateline[gtype_end+1:body_end].strip()

    return gtype, int(head), list(map(int, body.split(",")))

import sys

top_var = 0
name = {}

prefix = []
gates = []
output_constraints = []

occupied = set()

with open(sys.argv[1]) as fp:
    clauses = []
    for line in fp:
        if line[0] in ["p", "c"]:
            continue
        else:
            lits = list(map(int, line.split()))
            if lits[-1] == 0:
                lits.pop()
            occupied |= {abs(l) for l in lits}
            clauses.append(lits)
    top_var = max(occupied)
    prefix.append(sorted(occupied))

    # TODO reverse-engineer tseitin
    for c in clauses:
        top_var += 1
        gates.append(orgate(top_var, c))
        occupied.add(top_var)
        output_constraints.append(top_var)
    del clauses


prefix.append([]) # universal variables

for f in sys.argv[2:]:
    name = {}
    output = None
    property_type = None
    with open(f) as fp:
        for line in fp:
            if line[0] == "#":
                continue
            elif line.startswith("free"):
                F = set(map(int, line[line.index("(")+1:line.rindex(")")].split(",")))
            elif line.startswith("forall"):
                property_type = "forall"
                uvars = list(map(int, line[line.index("(")+1:line.rindex(")")].split(",")))
                for v in uvars:
                    top_var += 1
                    name[v] = top_var
                    occupied.add(top_var)
                    prefix[-1].append(top_var)
            elif line.startswith("exists"):
                property_type = "exists"
                uvars = list(map(int, line[line.index("(")+1:line.rindex(")")].split(",")))
                for v in uvars:
                    top_var += 1
                    name[v] = top_var
                    occupied.add(top_var)
                    prefix[0].append(top_var)
            elif line.startswith("output"):
                output = int(line[line.index("(")+1:line.rindex(")")])
            elif "=" in line:
                gtype, head, body = parse_gate(line)
                top_var += 1
                name[head] = top_var
                occupied.add(top_var)
                gates.append(mkgate(gtype, name.get(head, head), [name.get(x, x) if x > 0 else -name.get(-x, -x) for x in body]))
    #if property_type == "exists":
    #    output_constraints.append(name.get(output, output))
    #else:
    #    output_constraints.append(-name.get(output, output))
    output_constraints.append(name.get(output, output))

top_var += 1
gates.append(andgate(top_var, output_constraints))

prefix[0] = sorted(set(prefix[0]) - F)
X = ",".join(map(str, prefix[0]))
U = ",".join(map(str, sorted(prefix[1])))
F = ",".join(map(str, sorted(F)))

print ("#QCIR-G14")
if len(F) > 0:
    if len(prefix[0]) > 0:
        print(f"free({F})")
        print(f"exists({X})")
    else:
        print(f"exists({F})")
else:
        print(f"exists({X})")
print(f"forall({U})")
print(f"output({top_var})")
print("\n".join(gates))
