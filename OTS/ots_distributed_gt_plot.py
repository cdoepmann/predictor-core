import graph_tool.all as gt
import graph_tool
import numpy as np
import pandas as pd
# We need some Gtk and gobject functions
from gi.repository import Gtk, Gdk, GdkPixbuf, GObject
import numpy as np
import pdb
from optimal_traffic_scheduler import optimal_traffic_scheduler
import distributed_network
import matplotlib.pyplot as plt
np.random.seed(99)  # Luftballons


setup_dict = {}
setup_dict['n_in'] = 1
setup_dict['n_out'] = 1
setup_dict['v_max'] = 20  # mb/s
setup_dict['s_max'] = 30  # mb
setup_dict['dt'] = 1  # s
setup_dict['N_steps'] = 20
setup_dict['v_delta_penalty'] = 10

setup_dict.update({'n_out': 2})
ots_1 = optimal_traffic_scheduler(setup_dict)
setup_dict.update({'n_out': 1})
ots_2 = optimal_traffic_scheduler(setup_dict)
setup_dict.update({'n_in': 2})
ots_3 = optimal_traffic_scheduler(setup_dict)
# Input -> ots_1 -> out_2 -> out_3 -> Output


seq_length = 100
input_mode = 1

# Rows -> Outputs
# colums -> Inputs
# Each row sum must be 1 (otherwise data is lost or created.)
c1 = [np.array([[0.5], [0.5]])]*setup_dict['N_steps']
c2 = [np.array([[1]])]*setup_dict['N_steps']
c3 = [np.array([[1, 1]])]*setup_dict['N_steps']

if input_mode == 1:  # smoothed random values
    v_in_traj = np.convolve(18*np.random.rand(seq_length), np.ones(seq_length//10)/(seq_length/10), mode='same').reshape(-1, 1)
    v_in_traj = [v_in_traj[i].reshape(-1, 1) for i in range(v_in_traj.shape[0])]
if input_mode == 2:  # constant input
    v_in_traj = [np.array([[8]])]*seq_length

# Output fo server 3
bandwidth_traj = [np.array([[0]])]*seq_length
memory_traj = [np.array([[0]])]*seq_length

input_node_1 = distributed_network.input_node(v_in_traj)
output_node_1 = distributed_network.output_node(bandwidth_traj, memory_traj)

connections = [
    {'source': [input_node_1], 'node': ots_1, 'target': [ots_2, ots_3]},
    {'source': [ots_1],      'node': ots_2, 'target': [ots_3]},
    {'source': [ots_2, ots_1],    'node': ots_3, 'target': [output_node_1]},
]
input_nodes = [input_node_1]
output_nodes = [output_node_1]


def get_edges_nodes(connections):
    """
    Input: Connections must be of the format:
    connections = [
        {'source': [input], 'node': ots_1, 'target': [ots_2, ots_3]},
        ...
        }
    where each element of the list is a dict with the keywords 'source', 'node' and 'target'.
    'source' and 'target' must be lists of input, output or ots objects. 'node' must be an 'ots' object.
    Returns:
    - pandas.DataFrame with columns 'source' and 'target'. That define edges of a network.
    There are no duplicate edges.
    - pandas.DataFrame with columns 'nodes'. That defines the nodes of a network
    """
    # Initialize empty list. Each list item will contain a dict with 'source' and 'target'.
    # Note that each element in the "connections" list contains exactly one node but may contain multiple sources and targets.
    edges = []
    nodes = {'nodes': []}
    for connection_i in connections:
        nodes['nodes'].append(connection_i['node'])
        for source_k in connection_i['source']:
            edges.append({'source': source_k, 'target': connection_i['node']})
            nodes['nodes'].append(source_k)
        for target_k in connection_i['target']:
            edges.append({'source': connection_i['node'], 'target': target_k})
            nodes['nodes'].append(target_k)
    edges_df = pd.DataFrame(edges)
    nodes_df = pd.DataFrame(nodes)

    # Drop the duplicates, reset the index and delete the column (axis=1) that contains the old indices.
    edges_df_rmv_duplicate = edges_df.drop_duplicates().reset_index().drop('index', axis=1)
    nodes_df_rmv_duplicate = nodes_df.drop_duplicates().reset_index().drop('index', axis=1)

    return edges_df_rmv_duplicate, nodes_df_rmv_duplicate


dn = distributed_network.distributed_network(input_nodes, output_nodes, connections, setup_dict['N_steps'])

for i in range(3):
    dn.simulate(c_list=[c1, c2, c3])

color = plt.rcParams['axes.prop_cycle'].by_key()['color']

g = gt.Graph()

input_vert = [{'object': vert, 'vert': g.add_vertex()} for vert in input_nodes]
ots_vert = [{'object': vert['node'], 'vert': g.add_vertex()} for vert in connections]
output_vert = [{'object': vert, 'vert': g.add_vertex()} for vert in output_nodes]

vert_dt = pd.DataFrame(input_vert+ots_vert+output_vert)
edge_list, node_list = get_edges_nodes(connections)
edge_list['edge'] = None
for i, edge_i in edge_list.iterrows():
    source_i = vert_dt[vert_dt['object'] == edge_i['source']]
    target_i = vert_dt[vert_dt['object'] == edge_i['target']]
    edge_list['edge'][i] = g.add_edge(source_i.vert.values[0], target_i.vert.values[0])


vert_size = g.new_vertex_property('double')
vert_size.a = 20*np.random.rand(g.num_vertices())+5


vert_shape = g.new_vertex_property('string')
vert_comp = g.new_vertex_property('vector<double>')

comp = np.random.rand(g.num_vertices(), 3)
comp /= np.sum(comp, axis=0, keepdims=True)

halo = g.new_vertex_property('bool')
halo_size = g.new_vertex_property('double')
halo_color = g.new_vertex_property('vector<double>')

for i in range(g.num_vertices()):
    halo[i] = True
    halo_size[i] = 1.5
    vert_shape[i] = 'pie'
    vert_comp[i] = comp[i].tolist()
    halo_color[i] = [1.0, 0.16, 0.27, 0.98]


e_width = g.new_edge_property('double')
for i, edge_i in edge_list.iterrows():
    pdb.set_trace()
    source_i =
    e_width[edge_i['edge']] = 5

# for i, edge_i in enumerate(edge_list):
#     vert_dt[vert_dt['object'] == edge_i['source']]
#     e_width[i] = edge_i['source'].predict['v_out'][0]


vprops = {'size': vert_size, 'shape': vert_shape,
          'pie_fractions': vert_comp, 'halo': halo, 'halo_size': halo_size,
          'halo_color': halo_color}
eprops = {'pen_width': e_width}
pos = gt.sfdp_layout(g, K=0.2)

gt.graph_draw(g, pos=pos, vprops=vprops, eprops=eprops, output_size=(800, 400))
