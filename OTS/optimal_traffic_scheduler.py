import numpy as np
import matplotlib.pyplot as plt
from casadi import *
import pdb


class optimal_traffic_scheduler:
    def __init__(self, setup_dict):
        self.n_in = setup_dict['n_in']
        self.n_out = setup_dict['n_out']
        self.v_max = setup_dict['v_max']
        self.s_max = setup_dict['s_max']
        self.dt = setup_dict['dt']
        self.N_steps = setup_dict['N_steps']
        self.v_delta_penalty = setup_dict['v_delta_penalty']

        self.initialize_prediction()
        self.problem_formulation()
        self.create_optim()

    def initialize_prediction(self):
        v_out_traj = [np.zeros((self.n_out, 1))]*self.N_steps
        s_traj = [np.zeros((self.n_out, 1))]*self.N_steps
        bandwidth_traj = [np.zeros((1, 1))]*self.N_steps
        memory_traj = [np.zeros((1, 1))]*self.N_steps

        self.state = {}
        self.state['v_out_traj'] = v_out_traj
        self.state['s_traj'] = s_traj
        self.state['bandwidth_traj'] = bandwidth_traj
        self.state['memory_traj'] = memory_traj

    # def state(self, keyword):
    #     return self.state[keyword]

    def problem_formulation(self):

        # Problem dictionaries:
        param_dict = {}
        state_dict = {}
        cntrl_dict = {}

        # Composition Matrix
        c = SX.sym('c', self.n_out, self.n_in)

        # Buffer memory
        s = SX.sym('s', self.n_out, 1)

        # Incoming packet stream
        v_in = SX.sym('v_in', self.n_in, 1)
        # Outgoing packet stream
        v_out = SX.sym('v_out', self.n_out, 1)

        # bandwidth / memory info outgoing servers
        bandwidth_load = SX.sym('bandwidth_load', self.n_out, 1)
        memory_load = SX.sym('memory_load', self.n_out, 1)

        # system dynamics, constraints and objective definition:
        s_next = s + self.dt*(c@v_in-v_out)
        cons = [
            # maximum bandwidth cant be exceeded
            sum1(v_in)+sum1(v_out) - self.v_max,
            sum1(s)-self.s_max,  # maximum (buffer)memory capacity cant be exceeded
            -s,  # buffer memory cant be <0 (for each output buffer)
            -v_out,  # outgoing package stream cant be negative
            -v_out*(0.99-bandwidth_load),
            -v_out*(0.99-memory_load),
        ]
        cons = vertcat(*cons)

        # Maximize bandwidth  and maximize buffer:(under consideration of outgoing server load)
        obj = sum1((1-bandwidth_load)*(1-memory_load)*(-v_out/self.v_max+s/self.s_max))

        mpc_problem = {}
        mpc_problem['cons'] = Function(
            'cons', [v_in, v_out, s, bandwidth_load, memory_load], [cons])
        mpc_problem['obj'] = Function('obj', [v_in, v_out, s, bandwidth_load, memory_load], [obj])
        mpc_problem['model'] = Function(
            'model', [v_in, v_out, s, c], [s_next])

        self.mpc_problem = mpc_problem

    def create_optim(self):
        # Initialize trajectory lists:
        s_k = []
        v_in_k = []
        c_k = []
        v_out_k = []
        v_out_k_delta = []
        cons_k = []
        bandwidth_load_k = []
        memory_load_k = []
        # Initialize objective value:
        obj_k = 0

        # Initial condition:
        s0 = SX.sym('s0', self.n_out, 1)

        for k in range(self.N_steps):
            # Composition Matrix
            c_k.append(SX.sym('c', self.n_out, self.n_in))
            # Incoming packet stream
            v_in_k.append(SX.sym('v_in', self.n_in, 1))
            # Outgoing packet stream
            v_out_k.append(SX.sym('v_out', self.n_out, 1))
            if k < self.N_steps-1:
                v_out_k_delta = self.v_delta_penalty*sum1((v_out_k[k]-self.state['v_out_traj'][k+1])**2)/self.v_max
            else:
                v_out_k_delta = self.v_delta_penalty*sum1((v_out_k[k]-v_out_k[k-1])**2)/self.v_max

            obj_k += v_out_k_delta

            # bandwidth / memory info outgoing servers
            bandwidth_load_k.append(SX.sym('bandwidth_load', self.n_out, 1))
            memory_load_k.append(SX.sym('memory_load', self.n_out, 1))

            # For the first step use s0
            if k == 0:
                s_k.append(self.mpc_problem['model'](v_in_k[k], v_out_k[k], s0, c_k[k]))
            else:  # In suceeding steps use the previous s
                s_k.append(self.mpc_problem['model'](v_in_k[k], v_out_k[k], s_k[k-1], c_k[k]))

            # Add the "stage cost" to the objective
            obj_k += self.mpc_problem['obj'](v_in_k[k], v_out_k[k],
                                             s_k[k], bandwidth_load_k[k], memory_load_k[k])
            # Constraints for the current step
            cons_k.append(self.mpc_problem['cons'](
                v_in_k[k], v_out_k[k], s_k[k], bandwidth_load_k[k], memory_load_k[k]))

        # Parameters for the optimization problem must be a single nx1 vector.
        # Therefore the composition matrix of each step is reshaped.
        c_k = [c_i.reshape((-1, 1)) for c_i in c_k]

        optim_dict = {'x': vertcat(*v_out_k),  # Optimization variable
                      'f': obj_k,  # objective
                      'g': vertcat(*cons_k),  # constraints (Note: cons<=0)
                      'p': vertcat(s0, *v_in_k,  *c_k, *bandwidth_load_k, *memory_load_k)}  # parameters

        # Create casadi optimization object:
        self.optim = nlpsol('optim', 'ipopt', optim_dict)
        # Create function to calculate buffer memory from parameter and optimization variable trajectories
        self.s_traj = Function('s_traj', v_in_k+v_out_k+[s0]+c_k, s_k)

    def solve(self, s0, v_in_traj, c_traj, bandwidth_traj, memory_traj):
        # Reshape composition matrix for each time step:
        c_traj = [c_i.reshape(-1, 1) for c_i in c_traj]
        # Create concatented parameter vector:
        p = np.concatenate([s0]+v_in_traj+c_traj+bandwidth_traj+memory_traj)
        # Solve optimization problem for given conditions:
        sol = self.optim(ubg=0, p=p)  # Note: constraints were formulated, such that cons<=0.

        # Retrieve trajectory of outgoing package streams:
        x = sol['x'].full().reshape(self.N_steps, -1)
        v_out_traj = [x[[k]].T for k in range(self.N_steps)]

        # Calculate trajectory of buffer memory usage:
        s_traj = np.concatenate(self.s_traj(*v_in_traj+v_out_traj+[s0]+c_traj), axis=1).T
        s_traj = [s_traj[[k]].T for k in range(self.N_steps)]

        # Calculate trajectory for bandwidth and memory:
        bandwidth_traj = [(np.sum(v_out_k, keepdims=True)+np.sum(v_in_k, keepdims=True))/self.v_max for v_out_k,
                          v_in_k in zip(v_out_traj, v_in_traj)]

        memory_traj = [np.sum(s_k, keepdims=True)/self.s_max for s_k in s_traj]

        self.state['v_out_traj'] = v_out_traj
        self.state['s_traj'] = s_traj
        self.state['bandwidth_traj'] = bandwidth_traj
        self.state['memory_traj'] = memory_traj

        return self.state


# setup_dict = {}
# setup_dict['n_in'] = 2
# setup_dict['n_out'] = 2
# setup_dict['v_max'] = 20  # mb/s
# setup_dict['s_max'] = 200  # mb
# setup_dict['dt'] = 1  # s
# setup_dict['N_steps'] = 20
#
#
# ots = optimal_traffic_scheduler(setup_dict=setup_dict)
#
#
# # Test for sample problem:
# v_in_traj = [np.ones((setup_dict['n_in'], 1)) for i in range(setup_dict['N_steps'])]
#
# c_traj = [np.random.rand(setup_dict['n_out'], setup_dict['n_in'])
#           for i in range(setup_dict['N_steps'])]
# c_traj = [c_traj[i]/np.sum(c_traj[i], axis=0) for i in range(setup_dict['N_steps'])]
#
# bandwidth_traj = [np.random.rand(setup_dict['n_out'], 1) for i in range(setup_dict['N_steps'])]
# memory_traj = [np.random.rand(setup_dict['n_out'], 1) for i in range(setup_dict['N_steps'])]
#
# s0 = np.ones((setup_dict['n_out'], 1))
#
# ots.solve(s0, v_in_traj, c_traj, bandwidth_traj, memory_traj)
