import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from optimal_traffic_scheduler import optimal_traffic_scheduler
import pdb

mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.unicode_minus'] = False
#mpl.rcParams['svg.fonttype'] = 'none'
#mpl.rcParams['text.usetex'] = True
mpl.rcParams['mathtext.fontset'] = 'cm'
#mpl.rcParams['text.latex.preamble'] = "\\usepackage{amsmath}\\usepackage{amssymb}"

setup_dict = {}
setup_dict['v_in_max_total'] = 60  # packets / s
setup_dict['v_out_max_total'] = 60  # packets / s
setup_dict['s_c_max_total'] = 15  # packets
setup_dict['scaling'] = 50
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
s_buffer_0 = np.array([2, 4, 4]).reshape(-1, 1)

v_out_max = [np.array([[40., 30, 25.]]).T]*ots.N_steps

s_buffer_source = [np.array([[3,2.0, 0.0]]).T]*ots.N_steps
v_out_source = [np.array([[0.0,50.0, 10.0]]).T]*ots.N_steps


# Call the solver:
ots.solve(s_buffer_0, v_out_max, s_buffer_source, v_out_source, control_delta)

widths = [1,1,1,1]
heights = [2, 3]
gs = {'width_ratios' : widths, 'height_ratios': heights}
fig, ax = plt.subplots(2, 4, figsize=(14,4), sharex=True, gridspec_kw = gs)

x_state = range(len(ots.predict['s_buffer']))
x_input = range(len(ots.predict['v_out']))

ax[0,0].set_title('source ($n_i$) circuit queue')
ax[0,0].set_ylim(0, 10)

ax[1,0].step(x_input, np.concatenate(s_buffer_source, axis=1).T, linestyle='--')
ax[1, 0].set_prop_cycle(None)
ax[1,0].step(x_input, np.concatenate(ots.predict['s_buffer_source_corr'], axis=1).T, linewidth=4, alpha=0.5)
legend_lines = [Line2D([0], [0], color='#838383', lw=4, alpha=0.5),
                Line2D([0], [0], color='#838383', linestyle='--')]
ax[1,0].legend(legend_lines, ['$s_i$', '$\~{s}_{j|i}$'], loc='upper left', bbox_to_anchor=(0,1.2))


ax[0,0].step(x_input, np.sum(np.concatenate(s_buffer_source, axis=1), axis=0), linestyle='--', label='$\sum s_i$')
ax[0, 0].set_prop_cycle(None)
ax[0,0].step(x_input, np.sum(np.concatenate(ots.predict['s_buffer_source_corr'], axis=1), axis=0), linewidth=4, alpha=0.5, label='$\sum  \~{s}_{j|i}$')
ax[0,0].legend(loc='upper right')
##########

ax[0, 3].set_title('Outgoing packets')
ax[0, 3].axhline(setup_dict['v_out_max_total'], linestyle = '--', label='$C_{n_j}^{\mathrm{out}}$')
ax[0, 3].step(x_input, np.sum(np.concatenate(ots.predict['v_out'], axis=1), axis=0), linewidth=4, alpha=0.5, label='$\sum_{m\in P_{n_j}} r_{\mathrm{out},j,m}$')
ax[0, 3].legend(loc='lower left')
ax[0, 3].set_ylim(40, 62)

lines = ax[1, 3].step(x_input, np.concatenate(ots.predict['v_out'], axis=1).T, linewidth=4, alpha=0.5)
ax[1, 3].legend(lines, np.arange(n_out), title='Connection #',loc='center right', bbox_to_anchor=(1.5,0.5))
ax[1, 3].set_prop_cycle(None)
ax[1,3].step(x_input, np.concatenate(ots.predict['v_out_max'], axis=1)[0,:],linestyle='--')
ax[1,3].step(x_input, np.concatenate(ots.predict['v_out_max'], axis=1)[1,:],linestyle='--')
ax[1,3].step(x_input, np.concatenate(ots.predict['v_out_max'], axis=1)[2,:],linestyle='--')

legend_lines = [Line2D([0], [0], color='#838383', lw=4, alpha=0.5),
                Line2D([0], [0], color='#838383', linestyle='--')]
ax[1,3].add_artist(plt.legend(legend_lines, ['$r_{\mathrm{out},j}$', '$r_{\mathrm{out},j}^{\mathrm{max}}$'], loc='upper left', bbox_to_anchor=(0,1.2)))

###########

ax[0, 1].set_title('Incoming packets')
ax[0, 1].axhline(setup_dict['v_in_max_total'], linestyle = '--', label='$C_{n_j}^{\mathrm{in}}$')
ax[0, 1].step(x_input, np.sum(np.concatenate(ots.predict['v_in'], axis=1), axis=0), linewidth=4, alpha=0.5, label='$\sum_{m\in P_{n_j}} r_{\mathrm{in},j,m}$')
ax[0, 1].legend(loc='lower left')
ax[0, 1].set_ylim(40, 62)

lines = ax[1, 1].step(x_input, np.concatenate(ots.predict['v_in'], axis=1).T, linewidth=4, alpha=0.5)
plt.sca(ax[1, 1])
#ax[1, 1].add_artist(plt.legend(lines, np.arange(n_in), title='v_in con #', loc=1))
ax[1, 1].set_prop_cycle(None)
lines = ax[1, 1].step(x_input, np.concatenate(v_out_source, axis=1).T, linestyle='--')
legend_lines = [Line2D([0], [0], color='#838383', lw=4, alpha=0.5),
                Line2D([0], [0], color='#838383', linestyle='--')]
ax[1,1].legend(legend_lines, ['$r_{\mathrm{in},j}$', '$r_{\mathrm{out},i}$'], loc='center right')

##########

ax[0, 2].set_title('node ($n_j$) circuit queue')
ax[0, 2].step(x_state, np.sum(np.concatenate(ots.predict['s_buffer'], axis=1), axis=0), linewidth=4, alpha=0.5)

ax[1, 2].step(x_state, np.concatenate(ots.predict['s_buffer'], axis=1).T, linewidth=4, alpha=0.5)
ax[1, 2].axhline(setup_dict['s_c_max_total'], linestyle='--', color='#838383')
legend_lines = [Line2D([0], [0], color='#838383', lw=4, alpha=0.5),
                Line2D([0], [0], color='#838383', linestyle='--')]
ax[1,2].legend(legend_lines, ['$s_{j}$', '$s_{j}^{\mathrm{max}}$'], loc='upper left', bbox_to_anchor=(0,1.2))
#ax[1, 2].legend(lines, np.arange(n_out), title='Connection #')

##########


ax[0, 0].get_shared_y_axes().join(ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1])

for ax_i in ax.flatten():
    ax_i.grid(which='both', linestyle='--')
    ax_i.ticklabel_format(useOffset=False)
    #ax_i.set_ylim(bottom=-1, top=np.maximum(1.2*ax_i.get_ylim()[1], 1))
    #ax_i.set_ylim(bottom=-1)

ax[0, 0].set_ylabel('pack.')
ax[0, 1].set_ylabel('pack. / s')
ax[0, 2].set_ylabel('pack.')
ax[0, 3].set_ylabel('pack. / s')

ax[1, 0].set_ylabel('pack.')
ax[1, 1].set_ylabel('pack. / s')
ax[1, 2].set_ylabel('pack.')
ax[1, 3].set_ylabel('pack. / s')
ax[1, 0].set_xlabel('time [s]')
ax[1, 1].set_xlabel('time [s]')
ax[1, 2].set_xlabel('time [s]')
ax[1, 3].set_xlabel('time [s]')


fig.tight_layout(pad=0.2)
fig.align_ylabels()
plt.show()
