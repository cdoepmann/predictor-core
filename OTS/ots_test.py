import numpy as np
import matplotlib.pyplot as plt
from optimal_traffic_scheduler import optimal_traffic_scheduler
import pdb

setup_dict = {}
setup_dict['v_in_max_total'] = 10  # packets / s
setup_dict['v_out_max_total'] = 15  # packets / s
setup_dict['dt'] = 1  # s
setup_dict['N_steps'] = 20
setup_dict['weights'] = {'control_delta': 0., 'send': 1, 'store': 0, 'receive': 1}

ots = optimal_traffic_scheduler(setup_dict)

# Lets assume the following:
circuits_in = [[0, 1], [2, 3, 4]]
circuits_out = [[0, 3], [1, 4], [2]]

n_in = len(circuits_in)
n_out = len(circuits_out)

n_circuit_in = [len(c_i) for c_i in circuits_in]

n_circuit_out = [len(c_i) for c_i in circuits_out]

ots.setup(n_in, n_out, circuits_in, circuits_out)

# Create some dummy data:
s_circuit_0 = np.array([50, 10, 70, 10, 10]).reshape(-1, 1)
s_buffer_0 = np.array([60, 80, 10]).reshape(-1, 1)

v_in_req = [np.array([[20, 6]]).T]*ots.N_steps

cv_in = [[np.array([[0.5, 0.5]]).T, np.array([[0.3, 0.5, 0.2]]).T]]*ots.N_steps


v_out_max = [np.array([[5, 10, 12]]).T]*ots.N_steps

bandwidth_load_target = [np.array([[0, 0, 0]]).T]*ots.N_steps
s_buffer_target = [np.array([[10, 10, 10]]).T]*ots.N_steps

bandwidth_load_source = [np.array([[0, 0]]).T]*ots.N_steps
s_buffer_source = [np.array([[50, 50]]).T]*ots.N_steps


# Call the solver:
ots.solve(s_buffer_0, s_circuit_0, v_in_req, cv_in, v_out_max, bandwidth_load_target, s_buffer_target, bandwidth_load_source, s_buffer_source)


fig, ax = plt.subplots(2, 3, sharex=True, figsize=[16, 9])

ax[0, 0].step(range(ots.N_steps), np.sum(np.concatenate(ots.predict['v_out'], axis=1), axis=0))
ax[0, 1].step(range(ots.N_steps), np.sum(np.concatenate(ots.predict['v_in_req'], axis=1), axis=0), label='v_in_req')
ax[0, 1].step(range(ots.N_steps), np.sum(np.concatenate(ots.predict['v_in_max'], axis=1), axis=0), label='v_max')
ax[0, 1].step(range(ots.N_steps), np.sum(np.concatenate(ots.predict['v_in'], axis=1), axis=0), label='v_in')
ax[0, 1].legend()
ax[0, 2].step(range(ots.N_steps), np.sum(np.concatenate(ots.predict['s_buffer'], axis=1), axis=0))
ax[1, 0].step(range(ots.N_steps), np.concatenate(ots.predict['v_out'], axis=1).T)
ax[1, 1].step(range(ots.N_steps), np.concatenate(ots.predict['v_in'], axis=1).T)
ax[1, 2].step(range(ots.N_steps), np.concatenate(ots.predict['s_buffer'], axis=1).T)

# ax[0, 0].set_ylim(bottom=-0.1, top=1.2*setup_dict['v_out_max_total'])
# ax[0, 1].set_ylim(bottom=-0.1, top=1.2*setup_dict['v_in_max_total'])
# ax[1, 0].set_ylim(bottom=-0.1, top=1.2*setup_dict['v_out_max_total'])
# ax[1, 1].set_ylim(bottom=-0.1, top=1.2*setup_dict['v_in_max_total'])

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
#np.sum(np.concatenate(ots.predict['s_transit'], axis=1), axis=0)
