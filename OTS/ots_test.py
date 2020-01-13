import numpy as np
import matplotlib.pyplot as plt
from optimal_traffic_scheduler import optimal_traffic_scheduler
import pdb

setup_dict = {}
setup_dict['v_in_max_total'] = 800  # packets / s
setup_dict['v_out_max_total'] = 800  # packets / s
setup_dict['s_c_max_total'] = 100  # packets
setup_dict['scaling'] = 800
setup_dict['dt'] = 0.04  # s
setup_dict['N_steps'] = 10
control_delta = 0

ots = optimal_traffic_scheduler(setup_dict)

# Lets assume the following:
circuits_in = [[1], [2], [3]]
circuits_out = [[1], [2], [3]]

n_in = len(circuits_in)
n_out = len(circuits_out)
input_type = ['node', 'node', 'node']

ots.setup(n_in, n_out, circuits_in, circuits_out, input_type)

# Create some dummy data:
s_buffer_0 = np.array([0, 0, 0]).reshape(-1, 1)

v_out_max = [np.array([[400., 0, 400.]]).T]*ots.N_steps

s_buffer_source = [np.array([[100,5.0, 100.0]]).T]*ots.N_steps
v_out_source = [np.array([[400,0.0, 400.0]]).T]*ots.N_steps


# Call the solver:
ots.solve(s_buffer_0, v_out_max, s_buffer_source, v_out_source, control_delta)

fig, ax = plt.subplots(2, 3, sharex=True, figsize=[10, 6])

x_state = range(len(ots.predict['s_buffer']))
x_input = range(len(ots.predict['v_out']))

ax[0, 0].step(range(len(ots.predict['v_out'])), np.sum(np.concatenate(ots.predict['v_out'], axis=1), axis=0), linewidth=4, alpha=0.5, label='cumulated')
ax[0, 0].legend()
ax[0, 1].step(range(len(ots.predict['v_in'])), np.sum(np.concatenate(ots.predict['v_in'], axis=1), axis=0), linewidth=4, alpha=0.5, label='cumulated')
ax[0, 1].legend()
ax[0, 2].step(x_state, np.sum(np.concatenate(ots.predict['s_buffer'], axis=1), axis=0), linewidth=4, alpha=0.5)
ax[0, 2].step(x_input, np.concatenate(ots.predict['s_buffer_source_corr'], axis=1).T, linewidth=4, alpha=0.5, linestyle='--')
lines = ax[1, 0].step(range(len(ots.predict['v_out'])), np.concatenate(ots.predict['v_out'], axis=1).T, linewidth=4, alpha=0.5)
ax[1, 0].legend(lines, np.arange(n_out), title='Connection #')
lines = ax[1, 1].step(range(len(ots.predict['v_in'])), np.concatenate(ots.predict['v_in'], axis=1).T, linewidth=4, alpha=0.5)
plt.sca(ax[1, 1])
ax[1, 1].add_artist(plt.legend(lines, np.arange(n_in), title='v_in con #', loc=2))
ax[1, 1].set_prop_cycle(None)
lines = ax[1, 2].step(range(len(ots.predict['s_buffer'])), np.concatenate(ots.predict['s_buffer'], axis=1).T, linewidth=4, alpha=0.5)
ax[1, 2].legend(lines, np.arange(n_out), title='Connection #')

ax[0, 0].get_shared_y_axes().join(ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1])

for ax_i in ax.flatten():
    ax_i.grid(which='both', linestyle='--')
    ax_i.ticklabel_format(useOffset=False)
    #ax_i.set_ylim(bottom=-1, top=np.maximum(1.2*ax_i.get_ylim()[1], 1))
    #ax_i.set_ylim(bottom=-1)

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
