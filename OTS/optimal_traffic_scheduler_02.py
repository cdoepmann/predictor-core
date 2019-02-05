import numpy as np
import matplotlib.pyplot as plt
from casadi import *
import pdb


class optimal_traffic_scheduler:
    def __init__(self, setup_dict, name='ots', record_values=True):
        self.obj_name = name
        self.v_max = setup_dict['v_max']
        self.s_max = setup_dict['s_max']
        self.dt = setup_dict['dt']
        self.N_steps = setup_dict['N_steps']
        self.v_delta_penalty = setup_dict['v_delta_penalty']
        self.record_values = record_values
        self.time = np.array([[0]])  # 1,1 array for consistency.

        # Call setup, if n_in and n_out are part of the setup_dict.
        if 'n_in' and 'n_out_buffer' and 'n_out_circuit' in setup_dict.keys():
            self.n_in = setup_dict['n_in']
            self.n_out_buffer = setup_dict['n_out_buffer']
            self.n_out_circuit = setup_dict['n_out_circuit']
            self.setup()

    def setup(self, n_in=None, n_out_buffer=None, n_out_circuit=None):
        """
        Setup is part of the __init__(), if 'n_in' and 'n_out' are already defined.
        This is not the case, if the ots objects are created as part of a distributed network
        in which the I/O structure is created automatically.
        """
        if n_in and n_out_buffer and n_out_circuit:
            self.n_in = n_in
            self.n_out_buffer = n_out_buffer
            self.n_out_circuit = n_out_circuit

        self.initialize_prediction()
        if self.record_values:
            self.initialize_record()
        self.problem_formulation()
        self.create_optim()

    def initialize_prediction(self):
        # Initial conditions: all zeros.
        v_out_buffer = [np.zeros((self.n_out_buffer, 1))]*self.N_steps
        v_out_circuit = [np.zeros((self.n_out_circuit, 1))]*self.N_steps
        v_in_circuit = [np.zeros((self.n_in, 1))]*self.N_steps
        s_buffer = [np.zeros((self.n_out_buffer, 1))]*self.N_steps
        s_circuit = [np.zeros((self.n_out_circuit, 1))]*self.N_steps
        bandwidth_load = [np.zeros((1, 1))]*self.N_steps
        memory_load = [np.zeros((1, 1))]*self.N_steps

        self.predict = {}
        self.predict['v_out_buffer'] = v_out_buffer
        self.predict['v_out_circuit'] = v_out_circuit
        self.predict['v_in_circuit'] = v_in_circuit
        self.predict['s_buffer'] = s_buffer
        self.predict['s_circuit'] = s_circuit
        self.predict['bandwidth_load'] = bandwidth_load
        self.predict['memory_load'] = memory_load

    def initialize_record(self):
        # TODO : Save initial condition (especially for s)
        # NOTE: We are distinguishing between 'circuit' and 'buffer' for the memory, the input and the output.
        # Why? Each connected server node may carry multiple circuits. We keep track of them individually, to forward relevant predictions to
        # downstream nodes. This allows these nodes to predict where future packages will be directed. However, it is neiter feasible nor necessary
        # to regard circuits individually within a node. First of all, circuits do not persists, while connections do and second of all the number of circuits
        # is by orders of magnitude greater than the number of connections. The first reason would require to reformulate the optimization problem repeatedly,
        # while the second would increase the computation time. Lastly, it is not in our hand to influence the composition (with regards to circuits) of the output buffers
        # for the node. We can only manipulate the number of packages sent from each individual buffer and depending on the composition this will result in different amounts
        # of packages for each circuit.
        self.record = {'time': [], 'v_in_circuit': [], 'v_out_circuit': [], 'v_in_buffer': [], 'v_out_buffer': [],
                       's_circuit': [], 's_buffer': [], 'bandwidth_load': [], 'memory_load': [],
                       'bandwidth_load_target': [], 'memory_load_target': []}

        self.record['s_circuit'] = self.predict['s_circuit'][0]

    def problem_formulation(self):
        # Buffer memory
        s_buffer = SX.sym('s_buffer', self.n_out_buffer, 1)

        # Incoming package stream for each buffer:
        v_in_buffer = SX.sym('v_in_buffer', self.n_out_buffer, 1)

        # Outgoing packet stream
        v_out_buffer = SX.sym('v_out_buffer', self.n_out_buffer, 1)

        # bandwidth / memory info outgoing servers
        bandwidth_load = SX.sym('bandwidth_load', self.n_out_buffer, 1)
        memory_load = SX.sym('memory_load', self.n_out_buffer, 1)

        # system dynamics, constraints and objective definition:
        s_next = s_buffer + self.dt*(v_in_buffer-v_out_buffer)
        # Note: cons <= 0
        cons = [
            # maximum bandwidth cant be exceeded
            sum1(v_in_buffer)+sum1(v_out_buffer) - self.v_max,
            -s_buffer,  # buffer memory cant be <0 (for each output buffer)
            -v_out_buffer,  # outgoing package stream cant be negative
        ]
        cons = vertcat(*cons)

        # Maximize bandwidth  and maximize buffer:(under consideration of outgoing server load)
        # Note that 0<bandwidth_load<1 and memory_load is normalized by s_max but can exceed 1.
        obj = sum1(((1-bandwidth_load)/fmax(memory_load, 1))*(-v_out_buffer/self.v_max+s_buffer/self.s_max))

        mpc_problem = {}
        mpc_problem['cons'] = Function(
            'cons', [v_in_buffer, v_out_buffer, s_buffer, bandwidth_load, memory_load], [cons])
        mpc_problem['obj'] = Function('obj', [v_in_buffer, v_out_buffer, s_buffer, bandwidth_load, memory_load], [obj])
        mpc_problem['model'] = Function(
            'model', [v_in_buffer, v_out_buffer, s_buffer], [s_next])

        self.mpc_problem = mpc_problem

    def create_optim(self):
        # Initialize trajectory lists (each list item, one time-step):
        s_buffer_k = []  # [s_1, s_2, ..., s_N]
        v_buffer_in_k = []  # [v_in_0, v_in_1 , ... , v_in_N-1]
        v_buffer_out_k = []
        v_buffer_out_prev_k = []
        v_buffer_out_delta_k = []
        cons_k = []
        bandwidth_load_k = []
        memory_load_k = []
        # Initialize objective value:
        obj_k = 0

        # Initial condition:
        s_buffer_0 = SX.sym('s_buffer_0', self.n_out_buffer, 1)

        # Recursively evaluate system equation and add stage cost and stage constraints:
        for k in range(self.N_steps):

            # Incoming packet stream
            v_buffer_in_k.append(SX.sym('v_buffer_in', self.n_out_buffer, 1))
            # Outgoing packet stream
            v_buffer_out_k.append(SX.sym('v_buffer_out', self.n_out_buffer, 1))

            # For all but the last step: Penalize changes of v_buffer_out in comparison to the previous solution:
            if k < self.N_steps-1:
                v_buffer_out_prev_k.append(SX.sym('v_buffer_out_prev', self.n_out_buffer, 1))
                v_buffer_out_delta_k = self.v_delta_penalty*sum1((v_buffer_out_k[k]-v_buffer_out_prev_k[k])**2)/self.v_max
            else:
                # For the last step: Penalize change of v_buffer_out in comparison to the second to last step.
                v_buffer_out_delta_k = self.v_delta_penalty*sum1((v_buffer_out_k[k]-v_buffer_out_k[k-1])**2)/self.v_max

            obj_k += v_buffer_out_delta_k

            # bandwidth / memory info outgoing servers
            bandwidth_load_k.append(SX.sym('bandwidth_load', self.n_out_buffer, 1))
            memory_load_k.append(SX.sym('memory_load', self.n_out_buffer, 1))

            # For the first step use s_buffer_0
            if k == 0:
                s_buffer_k.append(self.mpc_problem['model'](v_buffer_in_k[k], v_buffer_out_k[k], s_buffer_0))
            else:  # In suceeding steps use the previous s
                s_buffer_k.append(self.mpc_problem['model'](v_buffer_in_k[k], v_buffer_out_k[k], s_buffer_k[k-1]))

            # Add the "stage cost" to the objective
            obj_k += self.mpc_problem['obj'](v_buffer_in_k[k], v_buffer_out_k[k],
                                             s_buffer_k[k], bandwidth_load_k[k], memory_load_k[k])
            # Constraints for the current step
            cons_k.append(self.mpc_problem['cons'](
                v_buffer_in_k[k], v_buffer_out_k[k], s_buffer_k[k], bandwidth_load_k[k], memory_load_k[k]))

        optim_dict = {'x': vertcat(*v_buffer_out_k),    # Optimization variable
                      'f': obj_k,                       # objective
                      'g': vertcat(*cons_k),            # constraints (Note: cons<=0)
                      'p': vertcat(s_buffer_0, *v_buffer_in_k, *v_buffer_out_prev_k, *bandwidth_load_k, *memory_load_k)}  # parameters

        # Create casadi optimization object:
        self.optim = nlpsol('optim', 'ipopt', optim_dict)
        # Create function to calculate buffer memory from parameter and optimization variable trajectories
        self.s_buffer_fun = Function('s_buffer_fun', v_buffer_in_k+v_buffer_out_k+[s_buffer_0], s_buffer_k)

    def solve(self, s_buffer_0, v_in_buffer, bandwidth_load_target, memory_load_target):
        """
        Solves the optimal control problem defined in optimal_traffic_scheduler.problem_formulation().
        Inputs:
        - s_buffer_0            : initial memory for each buffer (must be n_out_buffer x 1 vector)
        Predicted trajectories (as lists with N_horizon elments):
        - v_in_buffer           : Incoming package stream for each buffer (n_out_buffer x 1 vector)
        - bandwidth_load_target : Bandwidth load of target server(s) (n_out_buffer x 1 vector)
        - memory_load_target    : Memory load of target server(s) (n_out_buffer x 1 vector)

        Populates the "predict" and "record" dictonaries of the class.
        - Predict: Constantly overwritten variables that save the current optimized state and control trajectories of the node
        - Record:  Lists with items appended for each call of solve, recording the past states of the node.

        Returns the predict dictionary. Note that the predict dictionary contains some of the inputs trajectories as well as the
        calculated optimal trajectories.

        "Solve" also advances the time of the node by one time_step.
        """
        # Get previous solution:
        v_buffer_out_prev = self.predict['v_out_buffer']
        # Create concatented parameter vector:
        param = np.concatenate((s_buffer_0, *v_in_buffer, *v_buffer_out_prev[1:], *bandwidth_load_target, *memory_load_target), axis=0)
        # Get initial condition:
        x0 = np.concatenate(v_buffer_out_prev, axis=0)
        # Solve optimization problem for given conditions:
        sol = self.optim(ubg=0, p=param, x0=x0)  # Note: constraints were formulated, such that cons<=0.

        # Retrieve trajectory of outgoing package streams:
        x = sol['x'].full().reshape(self.N_steps, -1)
        v_out_buffer = [x[[k]].T for k in range(self.N_steps)]

        # Calculate trajectory of buffer memory usage:
        s_buffer = np.concatenate(self.s_buffer_fun(*v_in_buffer, *v_out_buffer, s_buffer_0), axis=1).T
        s_buffer = [s_buffer[[k]].T for k in range(self.N_steps)]

        # Calculate trajectory for bandwidth and memory:
        bandwidth_load_node = [(np.sum(v_out_k, keepdims=True)+np.sum(v_in_k, keepdims=True))/self.v_max for v_out_k,
                               v_in_k in zip(v_out_buffer, v_in_buffer)]

        memory_load_node = [np.sum(s_k, keepdims=True)/self.s_max for s_k in s_buffer]

        self.predict['v_in_buffer'] = v_in_buffer
        self.predict['v_out_buffer'] = v_out_buffer
        self.predict['s_buffer'] = s_buffer
        self.predict['bandwidth_load'] = bandwidth_load_node
        self.predict['memory_load'] = memory_load_node
        self.predict['bandwidth_load_target'] = np.copy(bandwidth_load_target)
        self.predict['memory_load_target'] = np.copy(memory_load_target)

        self.time += self.dt

        if self.record_values:
            self.record['time'].append(np.copy(self.time))
            self.record['v_in_buffer'].append(v_in_buffer[0])
            self.record['v_out_buffer'].append(v_out_buffer[0])
            self.record['s_buffer'].append(s_buffer[0])
            self.record['bandwidth_load'].append(bandwidth_load_node[0])
            self.record['memory_load'].append(memory_load_node[0])
            self.record['bandwidth_load_target'].append(bandwidth_load_target[0])
            self.record['memory_load_target'].append(memory_load_target[0])
        return self.predict

    def simulate_circuits(self, output_partition):
        v_in_circuit = self.predict['v_in_circuit']
        v_out_buffer = self.predict['v_out_buffer']
        s_circuit_0 = self.predict['s_circuit'][0]
        v_out_circuit = []
        s_circuit = []
        for k in range(self.N_steps):
            if k == 0:
                s_circuit_k = s_circuit_0
            else:
                s_circuit_k = s_circuit[k-1]
            v_out_circuit_k = np.maximum(s_circuit_k*(output_partition.T@(v_out_buffer[k]/np.maximum(output_partition@s_circuit_k, 1e-12))), 0)
            v_out_circuit.append(v_out_circuit_k)
            s_circuit_k_new = np.maximum(s_circuit_k+v_in_circuit[k]-v_out_circuit[k], 0)
            s_circuit.append(s_circuit_k_new)

        self.predict['v_out_circuit'] = v_out_circuit
        self.predict['s_circuit'] = s_circuit

        if self.record_values:
            self.record['v_out_circuit'] = v_out_circuit[0]
            self.record['v_in_circuit'] = v_in_circuit[0]
            self.record['s_circuit'] = s_circuit[0]
