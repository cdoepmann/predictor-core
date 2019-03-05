import numpy as np
import matplotlib.pyplot as plt
from ns_f import ns_f
from optimal_traffic_scheduler import optimal_traffic_scheduler

""" TODO:
- Implement multiple circuits per input.
    - s_buffer and s_circuit should be calculated correctly in ots_client

"""

dat = ns_f.data(5000)

setup_dict_server = {}
setup_dict_server['v_max'] = 2000  # packets / s
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
dt_ots = 0.1
N_steps = 20
ots_weights = {'control_delta': 0.1, 'send': 1, 'store': 1, 'receive': 5}

input_1.set_ots_client(dt_ots, N_steps)
input_2.set_ots_client(dt_ots, N_steps)
output_1.set_ots_client(dt_ots, N_steps)
output_2.set_ots_client(dt_ots, N_steps)
server_1.set_ots(dt_ots, N_steps, ots_weights)
server_2.set_ots(dt_ots, N_steps, ots_weights)

# Based on the network the ots objects are configured and setup. The optimizer is now ready to run.
nw.setup_ots(dt_ots, N_steps)

input_1.add_2_buffer(buffer_ind=0, circuit=0, n_packets=1000)
input_2.add_2_buffer(buffer_ind=0, circuit=1, n_packets=2000)

nw.run_ots()

s_list = []
win_size_list = []
t = []

n_steps = 100

for k in range(n_steps):
    s_k = nw.nodes.apply(lambda row: row['node'].s, axis=1).tolist()
    win_size = nw.connections.apply(lambda row: row['prop'].window_size, axis=1).tolist()

    s_list.append(s_k)
    win_size_list.append(win_size)
    t.append(nw.t)

    nw.simulate()

    if nw.t_next_iter <= nw.t:
        nw.run_ots()

    if np.mod(k, 10) == 0:
        print('{} % completed'.format(100*k/n_steps))


dat.packet_list[dat.packet_list['ttransit'] != np.inf].hist(by='circuit', column='ttransit', figsize=[13, 5], sharex=True, sharey=True)
plt.tight_layout()
plt.show()


def get_con_props(nw):
    # We query these attributes from the connections object ('prop') that is part of the dataframe
    props = ['window_size', 'window', 'transit', 'transit_reply']
    for p in props:
        nw.connections[p] = nw.connections.apply(lambda row: getattr(row.prop, p), axis=1)
    # And display only the relevant columns:
    return nw.connections[['source_name', 'target_name', 'circuit', 'window_size', 'window', 'transit', 'transit_reply']]


def get_node_props(nw):
    # For display purposes we add the output_buffer list to the table:
    nw.nodes['output_buffer'] = nw.nodes.apply(lambda row: row['node'].output_buffer, axis=1)
    # And display only the relevant columns:
    return nw.nodes[['name', 'n_in', 'n_out', 'output_circuits', 's_buffer', 'output_buffer']]
