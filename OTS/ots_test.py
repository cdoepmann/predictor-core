import numpy as np
import matplotlib.pyplot as plt
from optimal_traffic_scheduler import optimal_traffic_scheduler
import pdb

setup_dict = {}
setup_dict['v_in_max_total'] = 15  # packets / s
setup_dict['v_out_max_total'] = 15  # packets / s
setup_dict['dt'] = 1  # s
setup_dict['N_steps'] = 20
setup_dict['weights'] = {'control_delta': 0., 'send': 1, 'store': 0, 'receive': 1}

ots = optimal_traffic_scheduler(setup_dict)

# Lets assume the following:
circuits_in = [[0], [1]]
circuits_out = [[0], [1]]

n_in = len(circuits_in)
n_out = len(circuits_out)

n_circuit_in = [len(c_i) for c_i in circuits_in]

n_circuit_out = [len(c_i) for c_i in circuits_out]

ots.setup(n_in, n_out, circuits_in, circuits_out)

# Create some dummy data:
s_circuit_0 = np.array([50, 30]).reshape(-1, 1)
s_buffer_0 = np.array([50, 30]).reshape(-1, 1)

v_in_req = [np.array([[10, 10]]).T]*ots.N_steps

cv_in = [[np.array([[1.]]).T, np.array([[1.]]).T]]*ots.N_steps


v_out_max = [np.array([[10, 5]]).T]*ots.N_steps

bandwidth_load_target = [np.array([[0, 0]]).T]*ots.N_steps
s_buffer_target = [np.array([[10, 10]]).T]*ots.N_steps

bandwidth_load_source = [np.array([[0, 0]]).T]*ots.N_steps
s_buffer_source = [np.array([[25, 50]]).T]*ots.N_steps


# Call the solver:
ots.solve(s_buffer_0, s_circuit_0, v_in_req, cv_in, v_out_max, bandwidth_load_target, s_buffer_target, bandwidth_load_source, s_buffer_source)


fig, ax = plt.subplots(2, 3, sharex=True, figsize=[16, 9])

ax[0, 0].step(range(len(ots.predict['v_in_req'])), np.sum(np.concatenate(ots.predict['v_out'], axis=1), axis=0))
ax[0, 1].step(range(len(ots.predict['v_in_req'])), np.sum(np.concatenate(ots.predict['v_in_req'], axis=1), axis=0), label='v_in_req')
ax[0, 1].step(range(len(ots.predict['v_in_max'])), np.sum(np.concatenate(ots.predict['v_in_max'], axis=1), axis=0), label='v_max')
ax[0, 1].step(range(len(ots.predict['v_in'])), np.sum(np.concatenate(ots.predict['v_in'], axis=1), axis=0), label='v_in')
ax[0, 1].legend()
ax[0, 2].step(range(len(ots.predict['s_buffer'])), np.sum(np.concatenate(ots.predict['s_buffer'], axis=1), axis=0))
ax[1, 0].step(range(len(ots.predict['v_out'])), np.concatenate(ots.predict['v_out'], axis=1).T)
ax[1, 1].step(range(len(ots.predict['v_in'])), np.concatenate(ots.predict['v_in'], axis=1).T)
ax[1, 2].step(range(len(ots.predict['s_buffer'])), np.concatenate(ots.predict['s_buffer'], axis=1).T)

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
