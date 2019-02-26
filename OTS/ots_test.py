import numpy as np
import matplotlib.pyplot as plt
from optimal_traffic_scheduler import optimal_traffic_scheduler


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


circuits_in.append([5, 6])

circuits_in
