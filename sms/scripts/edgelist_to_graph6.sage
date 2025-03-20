#!/home1/peitl/bin/sage

import sys
from ast import literal_eval
for line in sys.stdin:
    print(Graph(literal_eval(line)).graph6_string())
