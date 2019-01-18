import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from casadi import *
import pdb
from optimal_traffic_scheduler import optimal_traffic_scheduler, ots_plotter
import distributed_network
np.random.seed(99)  # Luftballons


setup_dict = {}
setup_dict['n_in'] = 1
setup_dict['n_out'] = 1
setup_dict['v_max'] = 20  # mb/s
setup_dict['s_max'] = 20  # mb
setup_dict['dt'] = 1  # s
setup_dict['N_steps'] = 20
setup_dict['v_delta_penalty'] = 1

ots_1 = optimal_traffic_scheduler(setup_dict)
ots_2 = optimal_traffic_scheduler(setup_dict)
ots_3 = optimal_traffic_scheduler(setup_dict)

# Input -> ots_1 -> out_2 -> out_3 -> Output


seq_length = 100

# Input for server 1
c_traj = [np.array([[1]])]*setup_dict['N_steps']
v_in_traj = np.convolve(16*np.random.rand(seq_length), np.ones(seq_length//10)/(seq_length/10), mode='same').reshape(-1, 1)

v_in_traj = [v_in_traj[i].reshape(-1, 1) for i in range(v_in_traj.shape[0])]

v_in_traj = [np.array([[8]])]*seq_length

# Output fo server 3
bandwidth_traj = [np.array([[0]])]*seq_length
memory_traj = [np.array([[0]])]*seq_length

input_node = distributed_network.input_node(v_in_traj)
output_node = distributed_network.output_node(bandwidth_traj, memory_traj)

connections = [
    {'source': input_node, 'node': ots_1, 'target': ots_2},
    {'source': ots_1,      'node': ots_2, 'target': ots_3},
    {'source': ots_2,      'node': ots_3, 'target': output_node},
]

dn = distributed_network.distributed_network([input_node], [output_node], connections, setup_dict['N_steps'])


fig, ax = plt.subplots(3, 1)

N = range(setup_dict['N_steps'])


def update(t):
    dn.simulate(c_list=[c_traj]*len(connections))

    for ax_i in ax:
        ax_i.cla()
        ax_i.set_ylim([0, 15])

    line_obj = []
    line_obj.append(ax[0].step(N, np.concatenate(ots_1.predict['v_out']).reshape(-1, 1)))
    line_obj.append(ax[1].step(N, np.concatenate(ots_2.predict['v_out']).reshape(-1, 1)))
    line_obj.append(ax[2].step(N, np.concatenate(ots_3.predict['v_out']).reshape(-1, 1)))
    return line_obj


anim = FuncAnimation(fig, update, frames=range(5), repeat=False)
plt.show()

ots_1_plot = ots_plotter(ots_1)
ots_1_plot.update(6)
plt.show()
