import numpy as np
import matplotlib.pyplot as plt
from optimal_traffic_scheduler import optimal_traffic_scheduler
import pdb

setup_dict = {}
setup_dict['v_max'] = 20  # packets / s
setup_dict['s_max'] = 30  # packets
setup_dict['dt'] = 1  # s
setup_dict['N_steps'] = 20
setup_dict['v_delta_penalty'] = 3


ots = optimal_traffic_scheduler(setup_dict)


# Lets assume the following:
circuits_in = [[3, 4], [0, 1, 2]]
circuits_out = [[1, 2], [0], [3, 4]]

n_in = len(circuits_in)
n_out = len(circuits_out)

n_circuit_in = [len(c_i) for c_i in circuits_in]

n_circuit_out = [len(c_i) for c_i in circuits_out]

ots.setup(n_in, n_out, n_circuit_in, n_circuit_out)


def Pb_fun(c_in, c_out):
    P = [np.zeros((len(c_out), len(c_in_i))) for c_in_i in c_in]

    for i, c_in_i in enumerate(c_in):
        for j, c_in_ij in enumerate(c_in_i):
            k = [ind for ind, b_i in enumerate(c_out) if c_in_ij in b_i][0]
            P[i][k, j] = 1
    return P


def Pc_fun(c_in, c_out):
    c_in = [j for i in c_in for j in i]
    c_out = [j for i in c_out for j in i]
    Pc = np.zeros((len(c_out), len(c_in)))
    for i, c_in_i in enumerate(c_in):
        j = np.argwhere(c_in_i == np.array(c_out)).flatten()
        Pc[i, j] = 1
    return Pc


Pb = Pb_fun(circuits_in, circuits_out)
Pb = np.concatenate(Pb, axis=1)

Pc = Pc_fun(circuits_in, circuits_out)

# Create some dummy data:
s_buffer_0 = np.zeros((n_in, 1))
s_circuit_0 = np.zeros((np.sum(n_circuit_in), 1))

v_in_req = [np.array([[5, 3]]).T]*ots.N_steps

cv_in = [[np.array([[0.5, 0.5]]).T, np.array([[0.3, 0.3, 0.4]]).T]]*ots.N_steps

v_out_prev = [np.array([[0, 0, 0]]).T]*ots.N_steps

bandwidth_load_target = [np.array([[0]])]*ots.N_steps
memory_load_target = [np.array([[0]])]*ots.N_steps

bandwidth_load_source = [np.array([[0]])]*ots.N_steps
memory_load_source = [np.array([[0]])]*ots.N_steps


# Call the solver:
ots.solve(s_buffer_0, s_circuit_0, v_in_req, cv_in, Pb, Pc, bandwidth_load_target, memory_load_target, bandwidth_load_source, memory_load_source)
