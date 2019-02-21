import numpy as np
import matplotlib.pyplot as plt
from ns_f import *

dat = data()
ident = global_ident()

setup_dict = {}
setup_dict['v_max'] = 1000  # packets / s
setup_dict['s_max'] = 30  # packets
setup_dict['timeout'] = 1  # s

input_1 = server(setup_dict, ident, dat, name='input_1')
input_2 = server(setup_dict, ident, dat, name='input_2')
output_1 = server(setup_dict, ident, dat, name='output_1')
output_2 = server(setup_dict, ident, dat, name='output_2')
server_1 = server(setup_dict, ident, dat, name='server_1')
server_2 = server(setup_dict, ident, dat, name='server_2')
circuits = [
    {'route': [input_1, server_1, server_2, output_1]},
    {'route': [input_2, server_1, server_2, output_2]},
]

nw = network(data=dat)
nw.from_circuits(circuits)
input_1.add_2_buffer(buffer_ind=0, circuit=0, n_packets=10)
input_2.add_2_buffer(buffer_ind=0, circuit=1, n_packets=10)

for i in range(100):
    nw.simulate()
