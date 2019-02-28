import numpy as np
import matplotlib.pyplot as plt
from optimal_traffic_scheduler import optimal_traffic_scheduler
import pdb

setup_dict = {}
setup_dict['v_max'] = 20  # packets / s
setup_dict['s_max'] = 30  # packets
setup_dict['dt'] = 1  # s
setup_dict['N_steps'] = 20
setup_dict['weights'] = {'control_delta': 1.41, 'send': 1, 'store': 1, 'receive': 10}

ots = optimal_traffic_scheduler(setup_dict)


# Lets assume the following:
circuits_in = [[0, 1, 2], [3, 4]]
circuits_out = [[1, 3], [0], [2, 4]]

n_in = len(circuits_in)
n_out = len(circuits_out)

n_circuit_in = [len(c_i) for c_i in circuits_in]

n_circuit_out = [len(c_i) for c_i in circuits_out]

ots.setup(n_in, n_out, n_circuit_in, n_circuit_out)


Pb = ots.Pb_fun(circuits_in, circuits_out)

Pc = ots.Pc_fun(circuits_in, circuits_out)

# Create some dummy data:
s_buffer_0 = np.zeros((n_out, 1))
s_circuit_0 = np.zeros((np.sum(n_circuit_in), 1))

v_in_req = [np.array([[12, 4]]).T]*ots.N_steps

cv_in = [[np.array([[0.5, 0.25, 0.25]]).T, np.array([[0.5, 0.5]]).T]]*ots.N_steps


v_out_max = [np.array([[6, 6, 6]]).T]*ots.N_steps

bandwidth_load_target = [np.array([[0, 0, 0]]).T]*ots.N_steps
memory_load_target = [np.array([[0, 0, 0]]).T]*ots.N_steps

bandwidth_load_source = [np.array([[0, 0]]).T]*ots.N_steps
memory_load_source = [np.array([[0, 0]]).T]*ots.N_steps


# Call the solver:
ots.solve(s_buffer_0, s_circuit_0, v_in_req, cv_in, v_out_max, Pb, Pc, bandwidth_load_target, memory_load_target, bandwidth_load_source, memory_load_source)
