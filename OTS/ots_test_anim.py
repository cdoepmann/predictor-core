import numpy as np
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
# from casadi import *
import scipy.interpolate
import pdb
from optimal_traffic_scheduler import optimal_traffic_scheduler
import time

# ff_path = '/usr/bin/ffmpeg'
# plt.rcParams['animation.ffmpeg_path'] = ff_path
# if ff_path not in sys.path:
#     sys.path.append(ff_path)

output_format = '.'

setup_dict = {}
setup_dict['n_in'] = 2
setup_dict['n_out'] = 2
setup_dict['v_max'] = 20  # mb/s
setup_dict['s_max'] = 200  # mb
setup_dict['dt'] = 1  # s
setup_dict['N_steps'] = 40
setup_dict['v_delta_penalty'] = 0.1

interp = 5


ots = optimal_traffic_scheduler(setup_dict=setup_dict)


# Simple scenario for test problem:
c_traj = [np.array([[.4, .6], [.3, .7]]) for i in range(setup_dict['N_steps'])]

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
        bandwidth_traj += [np.array([[0.], [0.2]])]
        memory_traj += [np.array([[0], [0]])]
    else:
        v_in_traj += [np.array([[1], [2]])]
        bandwidth_traj += [np.array([[0.2], [0]])]
        memory_traj += [np.array([[0], [.3]])]


s0 = 1*np.ones((setup_dict['n_out'], 1))

scheduler_result = ots.solve(s0, v_in_traj, c_traj, bandwidth_traj, memory_traj)


# Interpolate for better visualization:
N = np.arange(setup_dict['N_steps'], dtype='float')
N_interp = np.linspace(0, setup_dict['N_steps'], num=interp*setup_dict['N_steps'])


c_traj_interp = scipy.interpolate.interp1d(N, np.stack(c_traj, axis=2), fill_value='extrapolate')(N_interp).T
v_in_traj_interp = scipy.interpolate.interp1d(N, np.concatenate(v_in_traj, axis=1), fill_value='extrapolate')(N_interp).T
bandwidth_traj_interp = scipy.interpolate.interp1d(N, np.concatenate(bandwidth_traj, axis=1), fill_value='extrapolate')(N_interp).T
memory_traj_interp = scipy.interpolate.interp1d(N, np.concatenate(memory_traj, axis=1), fill_value='extrapolate')(N_interp).T

scheduler_result_interp = {key: scipy.interpolate.interp1d(N, np.concatenate(
    value, axis=1), fill_value='extrapolate')(N_interp).T for key, value in scheduler_result.items()}


"""

Graphical Representation:

"""
fig = plt.figure(figsize=(16, 9))
in_stream = plt.subplot2grid((4, 4), (2, 0), rowspan=2)
in_comp = [plt.subplot2grid((4, 4), (0, 0), sharex=in_stream),
           plt.subplot2grid((4, 4), (1, 0), sharex=in_stream)]

server_buffer = plt.subplot2grid((4, 4), (0, 1), rowspan=2, sharex=in_stream)
server_load = [plt.subplot2grid((4, 4), (2, 1), sharex=in_stream),
               plt.subplot2grid((4, 4), (3, 1), sharex=in_stream)]

out_stream = plt.subplot2grid((4, 4), (0, 2), rowspan=2, sharex=in_stream)
out_load = [plt.subplot2grid((4, 4), (2, 2), sharex=in_stream),
            plt.subplot2grid((4, 4), (3, 2), sharex=in_stream)]

buffer1_anim = plt.subplot2grid((4, 4), (0, 3), rowspan=2)
buffer2_anim = plt.subplot2grid((4, 4), (2, 3), rowspan=2)

in_comp[0].set_title('Incoming Server')
server_buffer.set_title('Current Server')
out_stream.set_title('Outgoing Server')


def update(k):
    ind = np.argmin(np.abs(k-N_interp))

    line_obj = []
    # Incoming server plots:
    in_comp[0].cla()
    in_comp[1].cla()
    in_stream.cla()
    line_obj.append(in_comp[0].axvline(k, linestyle='--', color='#000000'))

    line_obj.append(in_comp[0].step(N, np.stack(c_traj, axis=2)[0, :, :].T))
    in_comp[0].set_ylabel('In 1: Composition')
    in_comp[0].legend(iter(line_obj[-1]), ('Out 1', 'Out 2'), loc='upper right')
    lines = in_comp[1].step(N, np.stack(c_traj, axis=2)[1, :, :].T)
    line_obj.append(in_comp[1].axvline(k, linestyle='--', color='#000000'))
    in_comp[1].legend(iter(lines), ('Out 1', 'Out 2'), loc='upper right')
    in_comp[1].set_ylabel('In 2: Composition')
    lines = in_stream.step(N, np.concatenate(v_in_traj, axis=1).T)
    line_obj.append(in_stream.axvline(k, linestyle='--', color='#000000'))
    in_stream.legend(iter(lines), ('In 1', 'In 2'), loc='upper right')
    in_stream.set_ylabel('Package stream')

    # Current server plots:
    server_buffer.cla()
    server_load[0].cla()
    server_load[1].cla()
    line_obj.append(server_load[0].axvline(k, linestyle='--', color='#000000'))
    line_obj.append(server_load[1].axvline(k, linestyle='--', color='#000000'))
    line_obj.append(server_buffer.axvline(k, linestyle='--', color='#000000'))
    lines = server_buffer.step(N, np.concatenate(scheduler_result['s_traj'], axis=1).T)
    server_buffer.legend(iter(lines), ('Out 1', 'Out 2'), loc='upper right')
    server_buffer.set_ylabel('buffer memory')
    server_load[0].step(N, np.concatenate(scheduler_result['bandwidth_traj']))
    server_load[0].set_ylabel('Bandwidth load')
    server_load[1].step(N, np.concatenate(scheduler_result['memory_traj']))
    server_load[1].set_ylabel('Memory load')

    # Outgoing server:
    out_stream.cla()
    out_load[0].cla()
    out_load[1].cla()
    line_obj.append(out_load[0].axvline(k, linestyle='--', color='#000000'))
    line_obj.append(out_load[1].axvline(k, linestyle='--', color='#000000'))
    line_obj.append(out_stream.axvline(k, linestyle='--', color='#000000'))
    line_obj.append(out_stream.step(N, np.concatenate(scheduler_result['v_out_traj'], axis=1).T))
    out_stream.set_ylabel('Package stream')
    out_stream.legend(iter(line_obj[-1]), ('Out 1', 'Out 2'), loc='upper right')
    lines = out_load[0].step(N, np.concatenate(bandwidth_traj, axis=1).T)
    out_load[0].legend(iter(lines), ('Out 1', 'Out 2'), loc='upper right')
    out_load[0].set_ylabel('Bandwidth load')
    lines = out_load[1].step(N, np.concatenate(memory_traj, axis=1).T)
    out_load[1].legend(iter(lines), ('Out 1', 'Out 2'), loc='upper right')
    out_load[1].set_ylabel('Memory load')

    # Sankey Diagrams:
    buffer1_anim.cla()
    buffer2_anim.cla()

    set_x_range = [ax.set_ylim([-0.05, 1.05]) for ax in in_comp+server_load+out_load]

    in_mat = (c_traj_interp[ind, :, :]*v_in_traj_interp[[ind]]).T
    out_mat = np.copy(scheduler_result_interp['v_out_traj'][ind])

    in11 = in_mat[0, 0]
    in12 = in_mat[0, 1]
    in21 = in_mat[1, 0]
    in22 = in_mat[1, 1]

    out1 = out_mat[0]
    out2 = out_mat[1]

    mem1 = scheduler_result_interp['s_traj'][ind][0]/20
    mem2 = scheduler_result_interp['s_traj'][ind][1]/20

    bw_cap1 = bandwidth_traj_interp[ind][0]
    mem_cap1 = memory_traj_interp[ind][0]

    bw_cap2 = bandwidth_traj_interp[ind][1]
    mem_cap2 = memory_traj_interp[ind][1]

    sankey = Sankey(ax=buffer1_anim, shoulder=0, scale=0.25)
    sankey.add(flows=[in11, in21, -out1],
               labels=['', '', 'Out: 1'],
               orientations=[1, -1, 0],
               pathlengths=[0.5, 0.5, 1])
    buffer1_anim.set_prop_cycle(None)
    sankey.add(flows=[in21, -in21, ],
               labels=['In: 2', ''],
               orientations=[0, 1],
               pathlengths=[0.5, 0.5],
               prior=0,
               connect=(1, 1))
    buffer1_anim.set_prop_cycle(None)
    sankey.add(flows=[in11, -in11, ],
               labels=['In: 1', ''],
               orientations=[0, -1],
               pathlengths=[0.5, 0.5],
               prior=0,
               connect=(0, 1))
    sankey.finish()
    line_obj.append(buffer1_anim.bar(0.5, mem1, width=0.5))
    line_obj.append(buffer1_anim.bar(2.1, bw_cap1, width=0.2))
    line_obj.append(buffer1_anim.bar(2.1, -mem_cap1, width=0.2))
    line_obj.append(buffer1_anim.bar(2.1, 1, width=0.4, fill=False, edgecolor='k', align='center'))
    line_obj.append(buffer1_anim.bar(2.1, -1, width=0.4, fill=False, edgecolor='k', align='center'))
    line_obj.append(buffer1_anim.get_xaxis().set_ticks([]))
    line_obj.append(buffer1_anim.get_yaxis().set_ticks([]))
    line_obj.append(buffer1_anim.set_xlim([-3.3, 3.3]))
    line_obj.append(buffer1_anim.set_ylim([-3.3, 3.3]))

    sankey = Sankey(ax=buffer2_anim, shoulder=0, scale=0.25)
    sankey.add(flows=[in12, in22, -out2],
               labels=['', '', 'Out: 2'],
               orientations=[1, -1, 0],
               pathlengths=[0.5, 0.5, 1])
    buffer2_anim.set_prop_cycle(None)
    sankey.add(flows=[in22, -in22, ],
               labels=['In: 2', ''],
               orientations=[0, 1],
               pathlengths=[0.5, 0.5],
               prior=0,
               connect=(1, 1))
    buffer2_anim.set_prop_cycle(None)
    sankey.add(flows=[in12, -in12, ],
               labels=['In: 1', ''],
               orientations=[0, -1],
               pathlengths=[0.5, 0.5],
               prior=0,
               connect=(0, 1))
    sankey.finish()
    buffer2_anim.bar(0.5, mem2, label='buffer memory', width=0.5)
    buffer2_anim.bar(2.1, bw_cap2, label='bandwidth load', width=0.2, align='center')
    buffer2_anim.bar(2.1, -mem_cap2, label='memory load', width=0.2, align='center')
    buffer2_anim.bar(2.1, 1, width=0.4, fill=False, edgecolor='k', align='center')
    buffer2_anim.bar(2.1, -1, width=0.4, fill=False, edgecolor='k', align='center')
    buffer2_anim.legend(loc='lower right')
    buffer2_anim.set_xlim([-3.3, 3.3])
    buffer2_anim.set_ylim([-3.3, 3.3])
    buffer2_anim.get_xaxis().set_ticks([])
    buffer2_anim.get_yaxis().set_ticks([])

    in_comp[0].set_title('Incoming Server')
    server_buffer.set_title('Current Server')
    out_stream.set_title('Outgoing Server')
    buffer1_anim.set_title('Flow Animation')

    return line_obj


plt.tight_layout(pad=4, w_pad=1.7, h_pad=1.2)
anim = FuncAnimation(fig, update, frames=N_interp, repeat=False)


if 'mp4' in output_format:
    FFWriter = FFMpegWriter(fps=10)
    anim.save('test_interp.mp4', writer=FFWriter)
elif 'html' in output_format:
    anim.to_html5_video(embed_limit=250)
elif 'gif' in output_format:
    gif_writer = ImageMagickWriter(fps=1)
    anim.save('test', writer=gif_writer)
else:
    plt.show()
