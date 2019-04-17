import numpy as np
import matplotlib.pyplot as plt
from optimal_traffic_scheduler import optimal_traffic_scheduler
import pdb

setup_dict = {}
setup_dict['v_in_max_total'] = 20  # packets / s
setup_dict['v_out_max_total'] = 20  # packets / s
setup_dict['s_max'] = 200  # packets
setup_dict['dt'] = 0.1  # s
setup_dict['N_steps'] = 20
setup_dict['weights'] = {'control_delta': 0.1, 'send': 1, 'store': 0, 'receive': 1}

ots = optimal_traffic_scheduler(setup_dict)


# Lets assume the following:
circuits_in = [[0, 1, 2], [3, 4]]
circuits_out = [[1, 3], [0], [2, 4]]

output_delay = np.array([0.2, 0.2, 0.2])

n_in = len(circuits_in)
n_out = len(circuits_out)

n_circuit_in = [len(c_i) for c_i in circuits_in]

n_circuit_out = [len(c_i) for c_i in circuits_out]

ots.setup(n_in, n_out, circuits_in, circuits_out, output_delay=output_delay)


# Create some dummy data:
s_buffer_0 = np.zeros((n_out, 1))
s_buffer_0[0] = 50
s_transit_0 = np.zeros((n_out, 1))
s_circuit_0 = np.zeros((np.sum(n_circuit_in), 1))

v_in_req = [np.array([[10, 3]]).T]*ots.N_steps

cv_in = [[np.array([[0.5, 0.25, 0.25]]).T, np.array([[0.5, 0.5]]).T]]*ots.N_steps


v_out_max = [np.array([[5, 5, 5]]).T]*ots.N_steps

bandwidth_load_target = [np.array([[0, 0, 0]]).T]*ots.N_steps
memory_load_target = [np.array([[0, 0, 0]]).T]*ots.N_steps

bandwidth_load_source = [np.array([[0, 0]]).T]*ots.N_steps
memory_load_source = [np.array([[0, 0]]).T]*ots.N_steps


# Call the solver:
ots.solve(s_buffer_0, s_circuit_0, s_transit_0, v_in_req, cv_in, v_out_max, bandwidth_load_target, memory_load_target, bandwidth_load_source, memory_load_source, output_delay)


fig, ax = plt.subplots(2, 3, sharex=True, figsize=[16, 9])

ax[0, 0].step(range(20), np.sum(np.concatenate(ots.predict[-1]['v_out'], axis=1), axis=0))
ax[0, 1].step(range(20), np.sum(np.concatenate(ots.predict[-1]['v_in_req'], axis=1), axis=0), label='v_in_req')
ax[0, 1].step(range(20), np.sum(np.concatenate(ots.predict[-1]['v_in_max'], axis=1), axis=0), label='v_max')
ax[0, 1].step(range(20), np.sum(np.concatenate(ots.predict[-1]['v_in'], axis=1), axis=0), label='v_in')
ax[0, 1].legend()
ax[0, 2].step(range(20), np.sum(np.concatenate(ots.predict[-1]['s_buffer'], axis=1), axis=0))
ax[1, 0].step(range(20), np.concatenate(ots.predict[-1]['v_out'], axis=1).T)
ax[1, 1].step(range(20), np.concatenate(ots.predict[-1]['v_in'], axis=1).T)
ax[1, 2].step(range(20), np.concatenate(ots.predict[-1]['s_buffer'], axis=1).T)

ax[0, 0].set_ylim(bottom=-0.1, top=setup_dict['v_out_max_total'])
ax[0, 1].set_ylim(bottom=-0.1, top=setup_dict['v_in_max_total'])
ax[0, 2].set_ylim(bottom=-1, top=setup_dict['s_max'])
ax[1, 0].set_ylim(bottom=-0.1, top=setup_dict['v_out_max_total'])
ax[1, 1].set_ylim(bottom=-0.1, top=setup_dict['v_in_max_total'])
ax[1, 2].set_ylim(bottom=-1, top=setup_dict['s_max'])

ax[0, 0].set_title('Outgoing packets')
ax[0, 1].set_title('Incoming packets')
ax[0, 2].set_title('Buffer memory')
ax[0, 0].set_ylabel('cumulated packets / second')
ax[0, 1].set_ylabel('cumulated packets / second')
ax[0, 2].set_ylabel('cumulated packets')
ax[1, 0].set_ylabel('packets / second')
ax[1, 1].set_ylabel('packets / second')
ax[1, 2].set_ylabel('packets')


plt.tight_layout()
plt.show()
#np.sum(np.concatenate(ots.predict[-1]['s_transit'], axis=1), axis=0)
