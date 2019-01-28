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
setup_dict['N_steps'] = 10
setup_dict['v_delta_penalty'] = 0

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

if input_mode == 1:  # smoothed random values
    v_in_traj = np.convolve(15*np.random.rand(seq_length), np.ones(seq_length//10)/(seq_length/10), mode='same').reshape(-1, 1)
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


dn = distributed_network.distributed_network(input_nodes, output_nodes, connections, setup_dict['N_steps'])

# for i in range(3):
#     dn.simulate(c_list=[c1, c2, c3])

dn_plot = distributed_network.ots_gt_plot(dn, connections)
# dn_plot.show_gt()

dn_plot.anim_gt(c_list=[c1, c2, c3])
