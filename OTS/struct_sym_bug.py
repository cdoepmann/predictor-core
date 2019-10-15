from casadi import *
from casadi.tools import struct_symSX, struct_SX, entry


a = struct_symSX([
    entry('a', shape=(3, 1)),
    entry('b', shape=(3, 3)),
    entry('c', shape=(3, 1))
])

b = struct_symSX([
    entry('a_struct', struct=a)
])
