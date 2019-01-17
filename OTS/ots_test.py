import numpy as np
import matplotlib.pyplot as plt
from casadi import *
import pdb
from optimal_traffic_scheduler import optimal_traffic_scheduler


setup_dict = {}
setup_dict['n_in'] = 2
setup_dict['n_out'] = 2
setup_dict['v_max'] = 20  # mb/s
setup_dict['s_max'] = 200  # mb
setup_dict['dt'] = 1  # s
setup_dict['N_steps'] = 40
setup_dict['v_delta_penalty'] = 0.1


ots = optimal_traffic_scheduler(setup_dict=setup_dict)


# Simple scenario for test problem:
c_traj = [np.array([[1, 0], [0.5, 0.5]]) for i in range(setup_dict['N_steps'])]

bandwidth_traj = []
memory_traj = []
v_in_traj = []
for k in range(setup_dict['N_steps']):
    if k < 10:
        v_in_traj += [np.array([[4], [2]])]
        bandwidth_traj += [np.array([[1], [0]])]
        memory_traj += [np.array([[1], [0]])]
    elif k < 20:
        v_in_traj += [5*np.ones((setup_dict['n_in'], 1))]
        bandwidth_traj += [np.array([[0], [0]])]
        memory_traj += [np.array([[0], [0]])]
    else:
        v_in_traj += [np.array([[1], [2]])]
        bandwidth_traj += [np.array([[0], [0]])]
        memory_traj += [np.array([[0], [0]])]


s0 = 1*np.ones((setup_dict['n_out'], 1))

scheduler_result = ots.solve(s0, v_in_traj, c_traj, bandwidth_traj, memory_traj)


"""

Graphical Representation:

"""
plt.figure(figsize=(16, 9))
in_stream = plt.subplot2grid((4, 3), (2, 0), rowspan=2)
in_comp = [plt.subplot2grid((4, 3), (0, 0), sharex=in_stream),
           plt.subplot2grid((4, 3), (1, 0), sharex=in_stream)]

server_buffer = plt.subplot2grid((4, 3), (0, 1), rowspan=2, sharex=in_stream)
server_load = [plt.subplot2grid((4, 3), (2, 1), sharex=in_stream),
               plt.subplot2grid((4, 3), (3, 1), sharex=in_stream)]

out_stream = plt.subplot2grid((4, 3), (0, 2), rowspan=2, sharex=in_stream)
out_load = [plt.subplot2grid((4, 3), (2, 2), sharex=in_stream),
            plt.subplot2grid((4, 3), (3, 2), sharex=in_stream)]

in_comp[0].set_title('Incoming Server')
server_buffer.set_title('Current Server')
out_stream.set_title('Outgoing Server')

N = range(setup_dict['N_steps'])

# Incoming server plots:
lines = in_comp[0].step(N, np.stack(c_traj, axis=2)[0, :, :].T)
in_comp[0].set_ylabel('In 1: Composition')
in_comp[0].legend(iter(lines), ('Out 1', 'Out 2'))
lines = in_comp[1].step(N, np.stack(c_traj, axis=2)[1, :, :].T)
in_comp[1].legend(iter(lines), ('Out 1', 'Out 2'))
in_comp[1].set_ylabel('In 2: Composition')
lines = in_stream.step(N, np.concatenate(v_in_traj, axis=1).T)
in_stream.legend(iter(lines), ('In 1', 'In 2'))
in_stream.set_ylabel('Package stream')

# Current server plots:
lines = server_buffer.step(N, np.concatenate(scheduler_result['s_traj'], axis=1).T)
server_buffer.legend(iter(lines), ('Out 1', 'Out 2'))
server_buffer.set_ylabel('buffer memory')
server_load[0].step(N, np.concatenate(scheduler_result['bandwidth_traj']))
server_load[0].set_ylabel('Bandwidth load')
server_load[1].step(N, np.concatenate(scheduler_result['memory_traj']))
server_load[1].set_ylabel('Memory load')

# Outgoing server:
lines = out_stream.step(N, np.concatenate(scheduler_result['v_out_traj'], axis=1).T)
out_stream.set_ylabel('Package stream')
out_stream.legend(iter(lines), ('Out 1', 'Out 2'))
lines = out_load[0].step(N, np.concatenate(bandwidth_traj, axis=1).T)
out_load[0].legend(iter(lines), ('Out 1', 'Out 2'))
out_load[0].set_ylabel('Bandwidth load')
lines = out_load[1].step(N, np.concatenate(memory_traj, axis=1).T)
out_load[1].legend(iter(lines), ('Out 1', 'Out 2'))
out_load[1].set_ylabel('Memory load')

set_x_range = [ax.set_ylim([-0.05, 1.05]) for ax in in_comp+server_load+out_load]


plt.tight_layout()
plt.show()
