import numpy as np
import matplotlib.pyplot as plt
from casadi import *
import pdb


class optimal_traffic_scheduler:
    def __init__(self, setup_dict, record_values=True):
        self.n_in = setup_dict['n_in']
        self.n_out = setup_dict['n_out']
        self.v_max = setup_dict['v_max']
        self.s_max = setup_dict['s_max']
        self.dt = setup_dict['dt']
        self.N_steps = setup_dict['N_steps']
        self.v_delta_penalty = setup_dict['v_delta_penalty']
        self.record_values = record_values

        self.initialize_prediction()
        if self.record_values:
            self.initialize_record()
        self.problem_formulation()
        self.create_optim()

    def initialize_prediction(self):
        v_out_traj = [np.zeros((self.n_out, 1))]*self.N_steps
        s_traj = [np.zeros((self.n_out, 1))]*self.N_steps
        bandwidth_traj = [np.zeros((1, 1))]*self.N_steps
        memory_traj = [np.zeros((1, 1))]*self.N_steps

        self.predict = {}
        self.predict['v_out'] = v_out_traj
        self.predict['s'] = s_traj
        self.predict['bandwidth_load'] = bandwidth_traj
        self.predict['memory_load'] = memory_traj

    def initialize_record(self):
        # TODO : Save initial condition (especially for s)
        self.record = {'v_in': [], 'v_out': [], 'v_in_buffer': [], 's': [], 'c': [], 'bandwidth_load': [], 'memory_load': []}

    def problem_formulation(self):

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
        # Note: cons <= 0
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
        s_k = []  # [s_1, s_2, ..., s_N]
        v_in_k = []  # [v_in_0, v_in_1 , ... , v_in_N-1]
        c_k = []
        v_out_k = []
        v_out_prev_k = []
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
                v_out_prev_k.append(SX.sym('v_out_prev', self.n_out, 1))
                v_out_k_delta = self.v_delta_penalty*sum1((v_out_k[k]-v_out_prev_k[k])**2)/self.v_max
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
                      'p': vertcat(s0, *v_in_k,  *c_k, *v_out_prev_k, *bandwidth_load_k, *memory_load_k)}  # parameters

        # Create casadi optimization object:
        self.optim = nlpsol('optim', 'ipopt', optim_dict)
        # Create function to calculate buffer memory from parameter and optimization variable trajectories
        self.s_traj = Function('s_traj', v_in_k+v_out_k+[s0]+c_k, s_k)

    def solve(self, s0, v_in_traj, c_traj, bandwidth_traj, memory_traj):
        # Reshape composition matrix for each time step:
        c_traj_reshape = [c_i.reshape(-1, 1) for c_i in c_traj]
        # Get previous solution:
        v_out_prev_traj = self.predict['v_out'][1:]
        # Create concatented parameter vector:
        p = np.concatenate([s0]+v_in_traj+c_traj_reshape+v_out_prev_traj+bandwidth_traj+memory_traj)
        # Solve optimization problem for given conditions:
        sol = self.optim(ubg=0, p=p)  # Note: constraints were formulated, such that cons<=0.

        # Retrieve trajectory of outgoing package streams:
        x = sol['x'].full().reshape(self.N_steps, -1)
        v_out_traj = [x[[k]].T for k in range(self.N_steps)]

        # Calculate trajectory of buffer memory usage:
        s_traj = np.concatenate(self.s_traj(*v_in_traj+v_out_traj+[s0]+c_traj_reshape), axis=1).T
        s_traj = [s_traj[[k]].T for k in range(self.N_steps)]

        # Calculate trajectory for bandwidth and memory:
        bandwidth_traj = [(np.sum(v_out_k, keepdims=True)+np.sum(v_in_k, keepdims=True))/self.v_max for v_out_k,
                          v_in_k in zip(v_out_traj, v_in_traj)]

        memory_traj = [np.sum(s_k, keepdims=True)/self.s_max for s_k in s_traj]

        self.predict['v_in'] = v_in_traj
        self.predict['v_out'] = v_out_traj
        self.predict['c'] = c_traj
        self.predict['s'] = s_traj
        self.predict['bandwidth_load'] = bandwidth_traj
        self.predict['memory_load'] = memory_traj

        if self.record_values:
            self.record['v_in'].append(v_in_traj[0])
            self.record['v_in_buffer'].append(c_traj[0]@v_in_traj[0])
            self.record['v_out'].append(v_out_traj[0])
            self.record['c'].append(v_out_traj[0])
            self.record['s'].append(c_traj[0])
            self.record['bandwidth_load'].append(bandwidth_traj[0])
            self.record['memory_load'].append(memory_traj[0])
        return self.predict


class ots_plotter:
    def __init__(self, ots):
        self.ots = ots

        self.fig, self.ax = plt.subplots(nrows=self.ots.n_out, ncols=3, figsize=(16, 9), sharex=True)
        self.ax = np.atleast_2d(self.ax)  # Otherwise indexing fails, when nrows=1

        color = plt.rcParams['axes.prop_cycle'].by_key()['color']

        self.lines_record = {
            'v_in': [],
            'v_out': [],
            's': [],
            'bandwidth_load': [],
            'memory_load': []
        }
        self.lines_pred = {
            'v_in': [],
            'v_out': [],
            's': [],
            'bandwidth_load': [],
            'memory_load': []
        }

        # Initialize all line objects without data:
        for out_k in range(self.ots.n_out):
            self.lines_record['v_in'].append(self.ax[out_k, 0].step([], [], color=color[0], animated=True))
            self.lines_record['v_out'].append(self.ax[out_k, 0].step([], [], color=color[1], animated=True))
            self.lines_record['s'].append(self.ax[out_k, 1].step([], [], color=color[0], animated=True))
            self.lines_record['bandwidth_load'].append(self.ax[out_k, 2].step([], [], color=color[0], animated=True))
            self.lines_record['memory_load'].append(self.ax[out_k, 2].step([], [], color=color[0], animated=True))

            self.lines_pred['v_in'].append(self.ax[out_k, 0].step([], [], color=color[0], linestyle='--', animated=True))
            self.lines_pred['v_out'].append(self.ax[out_k, 0].step([], [], color=color[1], linestyle='--', animated=True))
            self.lines_pred['s'].append(self.ax[out_k, 1].step([], [], color=color[0], linestyle='--', animated=True))
            self.lines_pred['bandwidth_load'].append(self.ax[out_k, 2].step([], [], color=color[0], linestyle='--', animated=True))
            self.lines_pred['memory_load'].append(self.ax[out_k, 2].step([], [], color=color[0], linestyle='--', animated=True))

    def init(self):
        self.ax[0, 0].set_ylabel('test')
        # Set axis limit, title etc in here.

    def update(self, k):
        time = np.arange(k)
        pred_time = np.arange(k, k+self.ots.N_steps)

        v_in_buffer = [c@v for c, v in zip(self.ots.predict['c'], self.ots.predict['v_in'])]

        for out_k in range(self.ots.n_out):
            pdb.set_trace()
            self.lines_record['v_in'][out_k][0].set_data(time, np.concatenate(self.ots.record['v_in_buffer'], axis=1).T)
            self.lines_record['v_out'][out_k][0].set_data(time, np.concatenate(self.ots.record['v_in'], axis=1).T)
            self.lines_record['s'][out_k][0].set_data(time, np.concatenate(self.ots.record['s'], axis=1).T)
            self.lines_record['bandwidth_load'][out_k][0].set_data(time, np.concatenate(self.ots.record['bandwidth_load'], axis=1).T)
            self.lines_record['memory_load'][out_k][0].set_data(time, np.concatenate(self.ots.record['memory_load'], axis=1).T)

            self.lines_pred['v_in'][out_k][0].set_data(pred_time, np.concatenate(v_in_buffer, axis=1).T)
            self.lines_pred['v_out'][out_k][0].set_data(pred_time, np.concatenate(self.ots.predict['v_in'], axis=1).T)
            self.lines_pred['s'][out_k][0].set_data(pred_time, np.concatenate(self.ots.predict['s'], axis=1).T)
            self.lines_pred['bandwidth_load'][out_k][0].set_data(pred_time, np.concatenate(self.ots.predict['bandwidth_load'], axis=1).T)
            self.lines_pred['memory_load'][out_k][0].set_data(pred_time, np.concatenate(self.ots.predict['memory_load'], axis=1).T)
        # Return all line objects as simple list:
        return np.array([*self.lines_record.values()]+[*self.lines_pred.values()]).ravel().tolist()
