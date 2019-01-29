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
setup_dict['v_max'] = 20  # mb/s
setup_dict['s_max'] = 30  # mb
setup_dict['dt'] = 1  # s
setup_dict['N_steps'] = 10
setup_dict['v_delta_penalty'] = 30

setup_dict['n_in'] = 3
setup_dict['n_out'] = 2
ots_1_in = optimal_traffic_scheduler(setup_dict)
ots_2_in = optimal_traffic_scheduler(setup_dict)
setup_dict['n_in'] = 2
setup_dict['n_out'] = 3
ots_1_out = optimal_traffic_scheduler(setup_dict)
ots_2_out = optimal_traffic_scheduler(setup_dict)

seq_length = 100
v_in_traj_1 = np.convolve(15*np.random.rand(seq_length), np.ones(seq_length//10)/(seq_length/10), mode='same').reshape(-1, 1)
v_in_traj_2 = np.convolve(15*np.random.rand(seq_length), np.ones(seq_length//10)/(seq_length/10), mode='same').reshape(-1, 1)
bandwidth_traj = [np.array([[0]])]*seq_length
memory_traj = [np.array([[0]])]*seq_length

input_node_1 = distributed_network.input_node(v_in_traj_1)
input_node_2 = distributed_network.input_node(v_in_traj_2)
output_node_1 = distributed_network.output_node(bandwidth_traj, memory_traj)
output_node_2 = distributed_network.output_node(bandwidth_traj, memory_traj)

connections = [
    {'source': [input_node_1, ots_1_out, ots_2_in], 'node': ots_1_in, 'target': [ots_1_out, ots_2_in]},
    {'source': [input_node_2, ots_1_in, ots_2_out],  'node': ots_2_in, 'target': [ots_1_in, ots_2_out]},
    {'source': [ots_1_in, ots_2_out],    'node': ots_1_out, 'target': [output_node_1, ots_1_in, ots_2_out]},
    {'source': [ots_2_in, ots_1_out],    'node': ots_2_out, 'target': [output_node_2, ots_2_in, ots_1_out]},
]
input_nodes = [input_node_1]
output_nodes = [output_node_1]


# Rows -> Outputs
# colums -> Inputs
# Each row sum must be 1 (otherwise data is lost or created.)
c_list = []
for connection_i in connections:
    n_in, n_out = connection_i['node'].n_in, connection_i['node'].n_out
    comp = np.random.rand(n_out, n_in)
    comp /= np.sum(comp, axis=0, keepdims=True)
    c_list.append([comp]*setup_dict['N_steps'])

# pdb.set_trace()

dn = distributed_network.distributed_network(input_nodes, output_nodes, connections, setup_dict['N_steps'])

for i in range(3):
    dn.simulate(c_list=c_list)

# dn_plot = distributed_network.ots_gt_plot(dn, connections)
# # dn_plot.show_gt()
#
# dn_plot.anim_gt(c_list=c)