import numpy as np
import matplotlib.pyplot as plt
from optimal_traffic_scheduler import optimal_traffic_scheduler
import pdb

setup_dict = {}
setup_dict['v_in_max_total'] = 300  # packets / s
setup_dict['v_out_max_total'] = 300  # packets / s
setup_dict['s_c_max_total'] = 200  # packets
setup_dict['dt'] = 0.04  # s
setup_dict['N_steps'] = 20
setup_dict['weights'] = {'control_delta': 1e4, 'send': 0, 'store': 0, 'receive': 0}

ots = optimal_traffic_scheduler(setup_dict)

# Lets assume the following:
circuits_in = [[1], [2], [3]]
circuits_out = [[1], [2], [3]]

n_in = len(circuits_in)
n_out = len(circuits_out)

n_circuit_in = [len(c_i) for c_i in circuits_in]

n_circuit_out = [len(c_i) for c_i in circuits_out]

ots.setup(n_in, n_out, circuits_in, circuits_out)

# Create some dummy data:
s_circuit_0 = np.array([100.0, 100.0, 100.0]).reshape(-1, 1)
s_buffer_0 = np.array([100.0, 100.0, 100.0]).reshape(-1, 1)

v_in_req = [np.array([[200., 100., 100.]]).T]*ots.N_steps

cv_in = [[np.array([[1.]]).T, np.array([[1]]).T, np.array([[1.]]).T]]*ots.N_steps


v_out_max = [np.array([[50., 100., 0.]]).T]*ots.N_steps

s_buffer_source = [np.array([[0.0, 0.0, 0.0]]).T]*ots.N_steps


# Call the solver:
ots.solve(s_buffer_0, s_circuit_0, v_in_req, cv_in, v_out_max, s_buffer_source)


fig, ax = plt.subplots(2, 3, sharex=True, figsize=[16, 9])

ax[0, 0].step(range(len(ots.predict['v_in_req'])), np.sum(np.concatenate(ots.predict['v_out'], axis=1), axis=0), linewidth=4, alpha=0.5)
ax[0, 1].step(range(len(ots.predict['v_in_req'])), np.sum(np.concatenate(ots.predict['v_in_req'], axis=1), axis=0), linewidth=4, alpha=0.5, label='v_in_req')
ax[0, 1].step(range(len(ots.predict['v_in_max'])), np.sum(np.concatenate(ots.predict['v_in_max'], axis=1), axis=0), linewidth=4, alpha=0.5, label='v_max')
ax[0, 1].step(range(len(ots.predict['v_in'])), np.sum(np.concatenate(ots.predict['v_in'], axis=1), axis=0), linewidth=4, alpha=0.5, label='v_in')
ax[0, 1].legend()
ax[0, 2].step(range(len(ots.predict['s_buffer'])), np.sum(np.concatenate(ots.predict['s_buffer'], axis=1), axis=0), linewidth=4, alpha=0.5)
lines = ax[1, 0].step(range(len(ots.predict['v_out'])), np.concatenate(ots.predict['v_out'], axis=1).T, linewidth=4, alpha=0.5)
ax[1, 0].legend(lines, np.arange(n_out), title='Connection #')
lines = ax[1, 1].step(range(len(ots.predict['v_in'])), np.concatenate(ots.predict['v_in'], axis=1).T, linewidth=4, alpha=0.5)
ax[1, 1].legend(lines, np.arange(n_in), title='Connection #')
lines = ax[1, 2].step(range(len(ots.predict['s_buffer'])), np.concatenate(ots.predict['s_buffer'], axis=1).T, linewidth=4, alpha=0.5)
ax[1, 2].legend(lines, np.arange(n_out), title='Connection #')

ax[0, 0].get_shared_y_axes().join(ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1])

for ax_i in ax.flatten():
    ax_i.grid(which='both', linestyle='--')
    ax_i.ticklabel_format(useOffset=False)
    # ax_i.set_ylim(bottom=-1, top=1.2*ax_i.get_ylim()[1])

# ax[0, 0].set_ylim(bottom=-0.1, top=1.2*setup_dict['v_in_max_total'])
# ax[0, 1].set_ylim(bottom=-0.1, top=1.2*setup_dict['v_in_max_total'])
# ax[1, 0].set_ylim(bottom=-0.1, top=1.2*setup_dict['v_out_max_total'])
# ax[1, 1].set_ylim(bottom=-0.1, top=1.2*setup_dict['v_in_max_total'])
# ax[0, 2].set_ylim(bottom=-0.1, top=1.2*ax[0, 2].get_ylim()[1])

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
# np.sum(np.concatenate(ots.predict['s_transit'], axis=1), axis=0)
