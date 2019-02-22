import numpy as np
import matplotlib.pyplot as plt
from ns_f import *

dat = data()

setup_dict = {}
setup_dict['v_max'] = 2000  # packets / s
setup_dict['s_max'] = 20  # packets
setup_dict['timeout'] = 0.5  # s

input_1 = server(setup_dict, dat, name='input_1')
input_2 = server(setup_dict, dat, name='input_2')
output_1 = server(setup_dict, dat, name='output_1')
output_2 = server(setup_dict, dat, name='output_2')
server_1 = server(setup_dict, dat, name='server_1')
server_2 = server(setup_dict, dat, name='server_2')
circuits = [
    {'route': [input_1, server_1, server_2, output_1]},
    {'route': [input_2, server_1, server_2, output_2]},
]


nw = network(data=dat)
nw.from_circuits(circuits)
input_1.add_2_buffer(buffer_ind=0, circuit=0, n_packets=10)
input_2.add_2_buffer(buffer_ind=0, circuit=1, n_packets=10)


s_list = []
win_size_list = []
t = []

n_steps = 500

for k in range(n_steps):
    if k < 400 and input_1.s_max-input_1.s >= 3:
        input_1.add_2_buffer(buffer_ind=0, circuit=0, n_packets=3, tnow=nw.t)
    if k < 400 and input_2.s_max-input_2.s >= 3:
        input_2.add_2_buffer(buffer_ind=0, circuit=1, n_packets=3, tnow=nw.t)

    s_k = nw.nodes.apply(lambda row: row['node'].s, axis=1).tolist()
    win_size = nw.connections['window_size'].tolist()

    s_list.append(s_k)
    win_size_list.append(win_size)
    t.append(nw.t)

    nw.simulate()

    if np.mod(k, 10) == 0:
        print('{} % completed'.format(100*k/n_steps))


win_size_list = np.array(win_size_list)
s_list = np.array(s_list)
t = np.array(t)


fig, ax = plt.subplots(1, 2, figsize=[13, 5], sharex=True)
lines = ax[0].plot(t, s_list)
ax[0].set_title('Buffer storage')
ax[0].set_ylabel('# packets')
ax[0].set_xlabel('time [s]')
ax[0].legend((lines), (nw.nodes.name.tolist()))
lines = ax[1].plot(t, win_size_list)
ax[1].legend((lines), (nw.connections.index.tolist()), title='Con #:')
ax[1].set_title('window size')
ax[1].set_xlabel('time [s]')
plt.tight_layout()
plt.show()
