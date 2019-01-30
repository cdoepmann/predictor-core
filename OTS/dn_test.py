import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb
from optimal_traffic_scheduler import optimal_traffic_scheduler
import distributed_network
np.random.seed(99)  # Luftballons


setup_dict = {}
setup_dict['n_in'] = 1
setup_dict['n_out'] = 1
setup_dict['v_max'] = 20  # mb/s
setup_dict['s_max'] = 30  # mb
setup_dict['dt'] = 1  # s
setup_dict['N_steps'] = 20
setup_dict['v_delta_penalty'] = 1

setup_dict.update({'n_out': 2})
ots_1 = optimal_traffic_scheduler(setup_dict)
setup_dict.update({'n_out': 1})
ots_2 = optimal_traffic_scheduler(setup_dict)
setup_dict.update({'n_in': 2})
ots_3 = optimal_traffic_scheduler(setup_dict)
# Input -> ots_1 -> out_2 -> out_3 -> Output

seq_length = 300
input_mode = 1

# Rows -> Outputs
# colums -> Inputs
# Each row sum must be 1 (otherwise data is lost or created.)
c1 = [np.array([[0.5], [0.5]])]*setup_dict['N_steps']
c2 = [np.array([[1]])]*setup_dict['N_steps']
c3 = [np.array([[1, 1]])]*setup_dict['N_steps']


v_in_traj = np.convolve(19*np.random.rand(seq_length), np.ones(10)/(10), mode='same').reshape(-1, 1)
v_in_traj = [v_in_traj[i].reshape(-1, 1) for i in range(v_in_traj.shape[0])]


# Output fo server 3
bandwidth_traj = [np.array([[0]])]*seq_length
memory_traj = [np.array([[0]])]*seq_length

input_node_1 = distributed_network.input_node(v_in_traj)
output_node_1 = distributed_network.output_node(bandwidth_traj, memory_traj)
input_node_1.get_input(0, setup_dict['N_steps'])
output_node_1.get_output(0, setup_dict['N_steps'])


nodes = [input_node_1, ots_1, ots_2, ots_3, output_node_1]
nodes = pd.DataFrame({'node': nodes})

connections = [
    {'source': input_node_1, 'target': ots_1},
    {'source': ots_1, 'target': ots_2},
    {'source': ots_1, 'target': ots_3},
    {'source': ots_2, 'target': ots_3},
    {'source': ots_3, 'target': output_node_1},
]
connections = pd.DataFrame(connections)

connections['output_ind'] = None
connections['v'] = None
connections['bandwidth_load'] = None
connections['memory_load'] = None

nodes['con_in'] = None
nodes['n_in'] = None
nodes['con_out'] = None
nodes['n_out'] = None

# Run only once to get information:
for k, node_k in nodes.iterrows():
    node_k['con_out'] = (connections['source'] == node_k['node']).values
    node_k['n_out'] = sum(node_k['con_out'])
    connections.loc[node_k['con_out'], 'output_ind'] = np.arange(node_k['n_out']).tolist()

    node_k['con_in'] = (connections['target'] == node_k['node']).values
    node_k['n_in'] = sum(node_k['con_in'])


# In every simulation:
for i, connection_i in connections.iterrows():
    connection_i['v'] = np.stack(connection_i['source'].predict['v_out'])[:, [connection_i['output_ind']], :]
    connection_i['bandwidth_load'] = np.stack(connection_i['target'].predict['bandwidth_load'])
    connection_i['memory_load'] = np.stack(connection_i['target'].predict['bandwidth_load'])

for k, node_k in nodes.iterrows():
    if type(node_k.node) is optimal_traffic_scheduler:
        v_in = np.concatenate(connections.loc[node_k['con_in'], 'v'].values, axis=1)
        bandwidth_load = np.concatenate(connections.loc[node_k['con_out'], 'bandwidth_load'].values, axis=1)
        memory_load = np.concatenate(connections.loc[node_k['con_out'], 'memory_load'].values, axis=1)

node_k.node
connections.loc[nodes.loc[0]['con_in'], 'memory_load']


e

ind = 3
nodes['v_in'] = None

v_in = []
for k, node_k in nodes.iterrows():
    v_in.append([])
    for i, con_in in connections[node_k['con_in']].iterrows():
        v_in[k].append(np.stack(con_in['source'].predict['v_out'])[:, [con_in['output_ind']], :])
    if v_in[k]:
        node_k['v_in'] = np.concatenate(v_in[k], axis=1)

nodes.loc[4]['v_in'].flags
v_in_k[0].flags
