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
        self.time = np.array([[0]])  # 1,1 array for consistency.

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
        self.record = {'time': [], 'v_in': [], 'v_out': [], 'v_in_buffer': [],
                       's': [], 'c': [], 'bandwidth_load': [], 'memory_load': [],
                       'bandwidth_load_target': [], 'memory_load_target': []}

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
            -s,  # buffer memory cant be <0 (for each output buffer)
            -v_out,  # outgoing package stream cant be negative
        ]
        cons = vertcat(*cons)

        # Maximize bandwidth  and maximize buffer:(under consideration of outgoing server load)
        # Note that 0<bandwidth_load<1 and memory_load is normalized by s_max but can exceed 1.
        obj = sum1(((1-bandwidth_load)/fmax(memory_load, 1))*(-v_out/self.v_max+s/self.s_max))

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

    def solve(self, s0, v_in_traj, c_traj, bandwidth_target_traj, memory_target_traj):
        """
        Solves the optimal control problem defined in optimal_traffic_scheduler.problem_formulation().
        Inputs:
        - s0            : current memory for each buffer (must be n_out x 1 vector)
        Predicted trajectories (as lists with N_horizon elments):
        - v_in_traj             : Incoming package stream (n_in x 1 vector)
        - c_traj                : Composition matrix of incoming package stream (n_out x n_in matrix)
        - bandwidth_target_traj : Bandwidth load of target server(s) (n_out x 1 vector)
        - memory_target_traj    : Memory load of target server(s) (n_out x 1 vector)

        Populates the "predict" and "record" dictonaries of the class.
        - Predict: Constantly overwritten variables that save the current optimized state and control trajectories of the node
        - Record:  Lists with items appended for each call of solve, recording the past states of the node.

        Returns the predict dictionary. Note that the predict dictionary contains some of the inputs trajectories as well as the
        calculated optimal trajectories.
        """

        # Check if input complies with requirements:
        if not all([c_traj_k.shape == (self.n_out, self.n_in) for c_traj_k in c_traj]):
            raise Exception('Composition Matrix is not consistent with the I/O of the node.')
        if not s0.shape == (self.n_out, 1):
            raise Exception('Initial buffer storage is not consistent with the I/O of the node.')
        if not all([v_in_traj_k.shape == (self.n_in, 1) for v_in_traj_k in v_in_traj]):
            raise Exception('Incoming package stream is not consistent with the I/O of the node.')
        if not all([bandwidth_target_traj_k.shape == (self.n_out, 1) for bandwidth_target_traj_k in bandwidth_target_traj]):
            raise Exception('Bandwidth information of connected target server is not consistent with the I/O of the node.')
        if not all([memory_target_traj_k.shape == (self.n_out, 1) for memory_target_traj_k in memory_target_traj]):
            raise Exception('Memory information of connected target server is not consistent with the I/O of the node.')

        # Reshape composition matrix for each time step:
        c_traj_reshape = [c_i.reshape(-1, 1) for c_i in c_traj]
        # Get previous solution:
        v_out_prev_traj = self.predict['v_out']
        # Create concatented parameter vector:
        param = np.concatenate((s0, *v_in_traj, *c_traj_reshape, *v_out_prev_traj[1:], *bandwidth_target_traj, *memory_target_traj), axis=0)
        # Get initial condition:
        x0 = np.concatenate(v_out_prev_traj, axis=0)
        # Solve optimization problem for given conditions:
        sol = self.optim(ubg=0, p=param, x0=x0)  # Note: constraints were formulated, such that cons<=0.

        # Retrieve trajectory of outgoing package streams:
        x = sol['x'].full().reshape(self.N_steps, -1)
        v_out_traj = [x[[k]].T for k in range(self.N_steps)]

        # Calculate trajectory of buffer memory usage:
        s_traj = np.concatenate(self.s_traj(*v_in_traj, *v_out_traj, s0, *c_traj_reshape), axis=1).T
        s_traj = [s_traj[[k]].T for k in range(self.N_steps)]

        # Calculate trajectory for bandwidth and memory:
        bandwidth_node_traj = [(np.sum(v_out_k, keepdims=True)+np.sum(v_in_k, keepdims=True))/self.v_max for v_out_k,
                               v_in_k in zip(v_out_traj, v_in_traj)]

        memory_node_traj = [np.sum(s_k, keepdims=True)/self.s_max for s_k in s_traj]

        self.predict['v_in'] = v_in_traj
        self.predict['v_out'] = v_out_traj
        self.predict['c'] = c_traj
        self.predict['s'] = s_traj
        self.predict['bandwidth_load'] = bandwidth_node_traj
        self.predict['memory_load'] = memory_node_traj
        self.predict['bandwidth_load_target'] = np.copy(bandwidth_target_traj)
        self.predict['memory_load_target'] = np.copy(memory_target_traj)

        self.time += self.dt

        if self.record_values:
            self.record['time'].append(np.copy(self.time))
            self.record['v_in'].append(v_in_traj[0])
            self.record['v_in_buffer'].append(c_traj[0]@v_in_traj[0])
            self.record['v_out'].append(v_out_traj[0])
            self.record['c'].append(v_out_traj[0])
            self.record['s'].append(s_traj[0])
            self.record['bandwidth_load'].append(bandwidth_node_traj[0])
            self.record['memory_load'].append(memory_node_traj[0])
            self.record['bandwidth_load_target'].append(bandwidth_target_traj[0])
            self.record['memory_load_target'].append(memory_target_traj[0])
        return self.predict


class ots_plotter:
    def __init__(self, ots):
        # ots must be list object to iterate over its entries.
        if type(ots) == list:
            self.ots = ots
        else:
            self.ots = [ots]

        nrows = 0
        for ots_i in self.ots:
            nrows += ots_i.n_out

        self.fig, self.ax = plt.subplots(nrows=nrows, ncols=3, figsize=(16, 9), sharex=True)
        self.ax = np.atleast_2d(self.ax)  # Otherwise indexing fails, when nrows=1

        self.color = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def plot(self, k):
        for ax_i in self.ax.ravel():
            ax_i.cla()

        lines = []
        offset = 0

        for server_i, ots_i in enumerate(self.ots):
            # Get time instances of prediction:
            pred_time = np.arange(start=ots_i.time, stop=ots_i.time+(ots_i.N_steps+1)*ots_i.dt, step=ots_i.dt)
            # Calculate from v__in and composition matrix the individual incoming package streams for each buffer:
            v_in_buffer = np.concatenate([ots_i.record['v_in_buffer'][-1]]+[c@v for c, v in zip(ots_i.predict['c'], ots_i.predict['v_in'])], axis=1)
            # Concatenate records and predictions (create numpy n-d-array) to comply with matplotlib API
            record = {name: np.concatenate(val, axis=1) for name, val in ots_i.record.items()}
            # For the predictions: Add the last element of the record to the sequence to avoid discontinuities in lines.
            predict = {name: np.concatenate((ots_i.record[name][-1], *val), axis=1) for name, val in ots_i.predict.items()}

            for out_k in range(ots_i.n_out):
                """Diagram 01: Incoming and Outgoing packages. """
                lines.append(self.ax[out_k+offset, 0].plot([], [], linewidth=0))  # Dummy to get legend entry
                lines.append(self.ax[out_k+offset, 0].step(record['time'][0], record['v_in_buffer'][out_k], color=self.color[0]))
                lines.append(self.ax[out_k+offset, 0].step(record['time'][0], record['v_out'][out_k], color=self.color[1]))
                lines.append(self.ax[out_k+offset, 0].plot([], [], linewidth=0))  # Dummy to get legend entry
                lines.append(self.ax[out_k+offset, 0].step(pred_time, v_in_buffer[out_k], color=self.color[0], linestyle='--'))
                lines.append(self.ax[out_k+offset, 0].step(pred_time, predict['v_out'][out_k], color=self.color[1], linestyle='--'))
                self.ax[out_k+offset, 0].legend([line[0] for line in lines[-6:]], ['Recorded', 'Incoming', 'Outoing', 'Predicted', 'Incoming', 'Outgoing'],
                                                loc='upper left', ncol=2, title='Package Streams')
                self.ax[out_k+offset, 0].set_ylim([0, ots_i.v_max*1.1])

                """Diagram 02: Buffer Memory. """
                self.ax[out_k+offset, 1].set_title('Server {0}, Output buffer {1}'.format(server_i+1, out_k+1))
                lines.append(self.ax[out_k+offset, 1].plot([], [], linewidth=0))  # Dummy to get legend entry
                lines.append(self.ax[out_k+offset, 1].step(record['time'][0], record['s'][out_k], color=self.color[0]))
                lines.append(self.ax[out_k+offset, 1].step(pred_time, predict['s'][out_k], color=self.color[0], linestyle='--'))
                self.ax[out_k+offset, 1].legend([line[0] for line in lines[-3:]], ['Buffer Memory', 'Recorded', 'Predicted'], loc='upper left')
                self.ax[out_k+offset, 1].set_ylim([0, ots_i.s_max*1.1])

                """Diagram 03: Load. """
                lines.append(self.ax[out_k+offset, 2].plot([], [], linewidth=0))  # Dummy to get legend entry
                lines.append(self.ax[out_k+offset, 2].step(record['time'][0], record['bandwidth_load_target'][out_k], color=self.color[0]))
                lines.append(self.ax[out_k+offset, 2].step(record['time'][0], record['memory_load_target'][out_k], color=self.color[1]))
                lines.append(self.ax[out_k+offset, 2].plot([], [], linewidth=0))  # Dummy to get legend entry
                lines.append(self.ax[out_k+offset, 2].step(pred_time, predict['bandwidth_load_target'][out_k], color=self.color[0], linestyle='--'))
                lines.append(self.ax[out_k+offset, 2].step(pred_time, predict['memory_load_target'][out_k], color=self.color[1], linestyle='--'))
                self.ax[out_k+offset, 2].legend([line[0] for line in lines[-6:]], ['Recorded', 'Bandwidth', 'Memory', 'Predicted', 'Bandwidth', 'Memory'],
                                                loc='upper left', ncol=2, title='Server Load')
                self.ax[out_k+offset, 2].set_ylim([-0.1, 1.1])

            offset += ots_i.n_out

        # Return all line objects
        return lines


class ots_graph_tool:
    def __init__(self):
        None
