import numpy as np
import matplotlib.pyplot as plt
from ns_f import ns_f
from optimal_traffic_scheduler import optimal_traffic_scheduler

dat = ns_f.data(1000)

setup_dict_server = {}
setup_dict_server['v_max'] = 20000  # packets / s
setup_dict_server['s_max'] = 200  # packets
setup_dict_server['timeout'] = 1  # s

input_1 = ns_f.server(setup_dict_server, dat, name='input_1')
input_2 = ns_f.server(setup_dict_server, dat, name='input_2')
output_1 = ns_f.server(setup_dict_server, dat, name='output_1')
output_2 = ns_f.server(setup_dict_server, dat, name='output_2')
server_1 = ns_f.server(setup_dict_server, dat, name='server_1')
server_2 = ns_f.server(setup_dict_server, dat, name='server_2')

circuits = [
    {'route': [input_1, server_1, server_2, output_1]},
    {'route': [input_2, server_1, server_2, output_2]},
]

nw = ns_f.network(data=dat)
nw.from_circuits(circuits)

# Each server node is assigned an ots object (or a client node)
dt_ots = 0.2
N_steps = 20
ots_weights = {'control_delta': 1, 'send': 1, 'store': 1, 'receive': 5}

input_1.set_ots_client(dt_ots, N_steps)
input_2.set_ots_client(dt_ots, N_steps)
output_1.set_ots_client(dt_ots, N_steps)
output_2.set_ots_client(dt_ots, N_steps)
server_1.set_ots(dt_ots, N_steps, ots_weights)
server_2.set_ots(dt_ots, N_steps, ots_weights)

# Based on the network the ots objects are configured and setup. The optimizer is now ready to run.
nw.setup_ots()
