import numpy as np
import pandas as pd
from optimal_traffic_scheduler_02 import optimal_traffic_scheduler
from distributed_network_02 import distributed_network, client_node
from ots_visu_02 import ots_gt_plot
import pdb

# Same configuration for all nodes:
setup_dict = {}
setup_dict['v_max'] = 30  # mb/s
setup_dict['s_max'] = 30  # mb
setup_dict['dt'] = 1  # s
setup_dict['N_steps'] = 20
setup_dict['v_delta_penalty'] = 10

ots_1 = optimal_traffic_scheduler(setup_dict, name='ots_1')
ots_2 = optimal_traffic_scheduler(setup_dict, name='ots_2')
ots_3 = optimal_traffic_scheduler(setup_dict, name='ots_3')
ots_4 = optimal_traffic_scheduler(setup_dict, name='ots_4')

source_fun = []

a = np.array([1, 3, 5])
data = a*np.ones((500, a.shape[0]))


def inputs(k, i):
    return [data[[k+m], i].reshape(1, 1) for m in range(setup_dict['N_steps'])]


source_fun = [lambda k: inputs(k, 0), lambda k: inputs(k, 1), lambda k: inputs(k, 2)]


def target_fun(k): return (setup_dict['N_steps']*[np.array([[0]])], setup_dict['N_steps']*[np.array([[0]])])


input_node_1 = client_node(setup_dict['N_steps'], name='input_node_1', source_fun=source_fun[0])
input_node_2 = client_node(setup_dict['N_steps'], name='input_node_2', source_fun=source_fun[1])
input_node_3 = client_node(setup_dict['N_steps'], name='input_node_3', source_fun=source_fun[2])
output_node_1 = client_node(setup_dict['N_steps'], name='output_node_1', target_fun=target_fun)
output_node_2 = client_node(setup_dict['N_steps'], name='output_node_2', target_fun=target_fun)
output_node_3 = client_node(setup_dict['N_steps'], name='output_node_3', target_fun=target_fun)

circuits = [
    {'route': [input_node_1, ots_1, ots_2, ots_3, ots_4, output_node_1]},
    {'route': [input_node_2, ots_1, ots_3, ots_4, output_node_2]},
    {'route': [input_node_3, ots_1, ots_2, ots_3, ots_4, output_node_3]},
]


dn = distributed_network(circuits)

gt_anim = ots_gt_plot(dn)
gt_anim.anim_gt()
# #
# for i in range(30):
#     dn.simulate()
#
# gt_anim.show_gt()
