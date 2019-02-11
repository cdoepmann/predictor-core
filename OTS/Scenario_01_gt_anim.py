import numpy as np
import pandas as pd
from optimal_traffic_scheduler import optimal_traffic_scheduler
from distributed_network import distributed_network, client_node
from ots_visu import ots_gt_plot
import pdb

# Same configuration for all nodes:
setup_dict = {}
setup_dict['v_max'] = 20  # mb/s
setup_dict['s_max'] = 30  # mb
setup_dict['dt'] = 1  # s
setup_dict['N_steps'] = 20
setup_dict['v_delta_penalty'] = 1

ots_1 = optimal_traffic_scheduler(setup_dict, name='ots_1')
ots_2 = optimal_traffic_scheduler(setup_dict, name='ots_2')
ots_3 = optimal_traffic_scheduler(setup_dict, name='ots_3')
ots_4 = optimal_traffic_scheduler(setup_dict, name='ots_4')
ots_5 = optimal_traffic_scheduler(setup_dict, name='ots_5')


source_fun = []

data = np.convolve((6*np.random.rand(100, 3)).ravel(), np.ones(5)/5, mode='same').reshape(100, 3)
# a = 1*np.array([1, 3, 5])
# data = a*np.ones((500, 3))


def inputs(k, i):
    return [data[[k+m], i].reshape(1, 1) for m in range(setup_dict['N_steps'])]


source_fun = [lambda k: inputs(k, 0), lambda k: inputs(k, 1), lambda k: inputs(k, 2)]


def target_fun(k): return (setup_dict['N_steps']*[np.array([[0]])], setup_dict['N_steps']*[np.array([[0]])])


input_node_1 = client_node(setup_dict['N_steps'], name='input_node_1', source_fun=source_fun[0])
input_node_2 = client_node(setup_dict['N_steps'], name='input_node_2', source_fun=source_fun[1])
input_node_3 = client_node(setup_dict['N_steps'], name='input_node_3', source_fun=source_fun[2])
input_node_4 = client_node(setup_dict['N_steps'], name='input_node_4', source_fun=source_fun[2])
output_node_1 = client_node(setup_dict['N_steps'], name='output_node_1', target_fun=target_fun)
output_node_2 = client_node(setup_dict['N_steps'], name='output_node_2', target_fun=target_fun)
output_node_3 = client_node(setup_dict['N_steps'], name='output_node_3', target_fun=target_fun)
output_node_4 = client_node(setup_dict['N_steps'], name='output_node_4', target_fun=target_fun)

circuits = [
    {'route': [input_node_1, ots_1, ots_2, ots_3, ots_4, output_node_1]},
    {'route': [input_node_2, ots_3, ots_5, ots_4, ots_1, output_node_2]},
    {'route': [input_node_3, ots_1, ots_2, ots_3, ots_4, output_node_3]},
    #    {'route': [input_node_4, ots_2, output_node_4]},
]


dn = distributed_network(circuits, delay=0.1)

gt_anim = ots_gt_plot(dn)
gt_anim.anim_gt()
