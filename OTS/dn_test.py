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


v_in_traj = np.convolve(7*np.random.rand(seq_length), np.ones(10)/(10), mode='same').reshape(-1, 1)
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
    # Boolean array that indicates in which connections node_k is the source.
    # This determines how many outputs the current node.
    node_k['con_out'] = (connections['source'] == node_k['node']).values
    node_k['n_out'] = sum(node_k['con_out'])
    # The output of each node is a vector with n_out elements. 'output_ind' marks which
    # of its elements refers to which connection:
    connections.loc[node_k['con_out'], 'output_ind'] = np.arange(node_k['n_out']).tolist()
    # Boolean array that indicates in which connections node_k is the target. This determines the
    # number of inputs.
    node_k['con_in'] = (connections['target'] == node_k['node']).values
    node_k['n_in'] = sum(node_k['con_in'])


# In every simulation:
# a) Determine associated properties for every connection that are determined by the source and target:
for i, connection_i in connections.iterrows():
    # Package stream is depending on the source. Create [N_timesteps x n_outputs x 1] array (with np.stack())
    # and access the element that is stored in 'output_ind' for each connection.
    connection_i['v'] = np.stack(connection_i['source'].predict['v_out'])[:, [connection_i['output_ind']], :]
    # Bandwidth and memory load are depending on the target. Create [N_timesteps x 1 x1] array.
    connection_i['bandwidth_load'] = np.stack(connection_i['target'].predict['bandwidth_load'])
    connection_i['memory_load'] = np.stack(connection_i['target'].predict['bandwidth_load'])

# b) Iterate over all nodes, query respective I/O data from connections and simulate node
for k, node_k in nodes.iterrows():
    # Simulate only if the node is an optimal_traffic_scheduler.
    if type(node_k.node) is optimal_traffic_scheduler:
        # Concatenate package streams for all inputs:
        v_in = np.concatenate(connections.loc[node_k['con_in'], 'v'].values, axis=1)
        # Concatenate bandwidth and memory load for all outputs:
        bandwidth_load = np.concatenate(connections.loc[node_k['con_out'], 'bandwidth_load'].values, axis=1)
        memory_load = np.concatenate(connections.loc[node_k['con_out'], 'memory_load'].values, axis=1)


circuits = [
    {'route': [input_node_1, ots_1, ots_2, ots_3, output_node_1], 'v_traj':v_in_traj},
    {'route': [input_node_1, ots_1, ots_3, output_node_1], 'v_traj':v_in_traj}
]


def circ_2_network(circuits):
    """
    Create nodes and connections DataFrame from circuit dict.
    Each connection has a new attribute 'circuit' that lists all the circuits that are active.
    """
    connections = []
    nodes = {'node': []}
    for i, circuit_i in enumerate(circuits):
        connections.extend([{'source': circuit_i['route'][k], 'target': circuit_i['route'][k+1], 'circuit':[i]}
                            for k in range(len(circuit_i['route'])-1)])
        nodes['node'].extend(circuit_i['route'])
    con_pd = pd.DataFrame(connections)
    nodes_pd = pd.DataFrame(nodes).drop_duplicates().reset_index().drop('index', axis=1)
    # Identify duplicate connections (same source and target) and add which circuit is active.
    for i, dupl_con in con_pd[con_pd.duplicated(subset=['source', 'target'])].reset_index().iterrows():
        con_pd[(con_pd.source == dupl_con.source) & (con_pd.target == dupl_con.target)].circuit.values[0].extend(dupl_con['circuit'])
    con_pd = con_pd.drop_duplicates(subset=['source', 'target'])
    return con_pd, nodes_pd


con_pd, nodes_pd = circ_2_network(circuits)

nodes_pd
