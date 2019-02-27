import numpy as np
import matplotlib.pyplot as plt
from optimal_traffic_scheduler import optimal_traffic_scheduler
import pdb

setup_dict = {}
setup_dict['v_max'] = 20  # packets / s
setup_dict['s_max'] = 30  # packets
setup_dict['dt'] = 1  # s
setup_dict['N_steps'] = 20
setup_dict['v_delta_penalty'] = 2


ots = optimal_traffic_scheduler(setup_dict)


# Lets assume the following:
circuits_in = [[0, 1, 2], [3, 4]]
circuits_out = [[1, 3], [0], [2, 4]]

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
        Pc[j, i] = 1
    return Pc


Pb = Pb_fun(circuits_in, circuits_out)
Pb = np.concatenate(Pb, axis=1)

Pc = Pc_fun(circuits_in, circuits_out)
# Create some dummy data:
s_buffer_0 = np.zeros((n_out, 1))
s_circuit_0 = np.zeros((np.sum(n_circuit_in), 1))

v_in_req = [np.array([[12, 4]]).T]*ots.N_steps

cv_in = [[np.array([[0.5, 0.25, 0.25]]).T, np.array([[0.5, 0.5]]).T]]*ots.N_steps

v_out_prev = [np.array([[0, 0, 0]]).T]*ots.N_steps

v_out_max = [np.array([[6, 6, 6]]).T]*ots.N_steps

bandwidth_load_target = [np.array([[0, 0, 0]]).T]*ots.N_steps
memory_load_target = [np.array([[0, 0, 0]]).T]*ots.N_steps

bandwidth_load_source = [np.array([[0, 0]]).T]*ots.N_steps
memory_load_source = [np.array([[0, 0]]).T]*ots.N_steps


# Call the solver:
ots.solve(s_buffer_0, s_circuit_0, v_in_req, cv_in, v_out_max, Pb, Pc, bandwidth_load_target, memory_load_target, bandwidth_load_source, memory_load_source)


v_in_max = [np.array([[12, 4]]).T]*ots.N_steps
v_out = [np.array([[5, 6, 5]]).T]*ots.N_steps
model_fun = ots.mpc_problem['model']

s_buffer = [s_buffer_0]
s_circuit = [s_circuit_0]
cv_out = []
for k in range(ots.N_steps):
    model_out = model_fun(s_buffer[k], s_circuit[k], v_in_req[k], v_in_max[k], *cv_in[k], v_out[k], Pb, Pc)
    model_out = [model_out_i.full() for model_out_i in model_out]
    s_buffer.append(model_out[0])
    s_circuit.append(model_out[1])
    cv_out.append(model_out[2:])


s_buffer = np.concatenate(s_buffer, axis=1).T
s_circuit = np.concatenate(s_circuit, axis=1).T

plt.plot(s_buffer)
plt.show()


cv_out = [np.concatenate(i) for i in cv_out]
np.concatenate(cv_out, axis=1).T
