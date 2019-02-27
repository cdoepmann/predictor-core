import numpy as np
import matplotlib.pyplot as plt
from casadi import *
import pdb
from scipy.interpolate import interp1d


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

    def setup(self, n_in=None, n_out=None, n_circuit_in=None, n_circuit_out=None):
        """
        n_in: Number of Inputs
        n_out: Number of outputs
        n_circuit_in: List with number of circuits per input (len(n_circuit_in)=n_in)
        n_circuit_out: List with number of circuits per input (len(n_circuit_out)=n_out)
        """

        self.n_in = n_in
        self.n_out = n_out
        self.n_circuit_in = n_circuit_in
        self.n_circuit_out = n_circuit_out

        assert len(self.n_circuit_in) == self.n_in
        assert len(self.n_circuit_out) == self.n_out
        assert np.sum(self.n_circuit_in) == np.sum(self.n_circuit_out)

        self.initialize_prediction()
        if self.record_values:
            self.initialize_record()
        self.problem_formulation()
        self.create_optim()

    def initialize_prediction(self):
        # Initial conditions: all zeros.
        v_out = [np.zeros((self.n_out, 1))]*self.N_steps
        v_out_max = [np.zeros((self.n_out, 1))]*self.N_steps
        cv_out = [[np.zeros((n_circuit_out_i, 1)) for n_circuit_out_i in self.n_circuit_out]]*self.N_steps
        v_in = [np.zeros((self.n_in, 1))]*self.N_steps
        v_in_req = [np.zeros((self.n_in, 1))]*self.N_steps
        v_in_max = [np.zeros((self.n_in, 1))]*self.N_steps
        cv_in = [[np.zeros((n_circuit_in_i, 1)) for n_circuit_in_i in self.n_circuit_in]]*self.N_steps
        s_buffer = [np.zeros((self.n_out, 1))]*self.N_steps
        s_circuit = [np.zeros((np.sum(self.n_circuit_in), 1))]*self.N_steps
        bandwidth_load = [np.zeros((1, 1))]*self.N_steps
        memory_load = [np.zeros((1, 1))]*self.N_steps
        bandwidth_load_target = [np.zeros((self.n_out, 1))]*self.N_steps
        memory_load_target = [np.zeros((self.n_out, 1))]*self.N_steps
        bandwidth_load_source = [np.zeros((self.n_in, 1))]*self.N_steps
        memory_load_source = [np.zeros((self.n_in, 1))]*self.N_steps

        self.predict = [{}]
        self.predict[0]['v_out'] = v_out
        self.predict[0]['v_out_max'] = v_out_max
        self.predict[0]['cv_out'] = cv_out
        self.predict[0]['v_in'] = v_in
        self.predict[0]['v_in_req'] = v_in_req
        self.predict[0]['v_in_max'] = v_in_max
        self.predict[0]['cv_in'] = cv_in
        self.predict[0]['s_buffer'] = s_buffer
        self.predict[0]['s_circuit'] = s_circuit
        self.predict[0]['bandwidth_load'] = bandwidth_load
        self.predict[0]['memory_load'] = memory_load
        self.predict[0]['bandwidth_load_target'] = bandwidth_load_target
        self.predict[0]['memory_load_target'] = memory_load_target
        self.predict[0]['bandwidth_load_source'] = bandwidth_load_source
        self.predict[0]['memory_load_source'] = memory_load_source

    def initialize_record(self):
        self.record = {}
        self.record['time'] = []
        # Set the first element of the predicted values as the first element of the recorded values.
        for predict_key in self.predict[0].keys():
            self.record[predict_key] = []
            self.record[predict_key].append(self.predict[0][predict_key][0])

    def problem_formulation(self):
        """ Memory """
        # Buffer memory
        s_buffer = SX.sym('s_buffer', self.n_out, 1)
        # Circuit memory:
        s_circuit = SX.sym('s_circuit', np.sum(self.n_circuit_in), 1)

        """ Incoming packet stream """
        # Incoming package stream for each buffer:
        v_in_req = SX.sym('v_in_req', self.n_in, 1)
        # Allowed incoming packet stream:
        v_in_max = SX.sym('v_in_max', self.n_in, 1)
        # v_in_max from previous solution
        v_in_max_prev = SX.sym('v_in_max_prev', self.n_in, 1)
        # Resulting incoming packet stream:
        v_in_list = vertsplit(fmin(v_in_req, v_in_max))
        # Concatenate v_in:
        v_in = vertcat(*v_in_list)
        # Composition of incoming stream:
        cv_in = [SX.sym('cv_in_'+str(i), self.n_circuit_in[i], 1) for i in range(self.n_in)]
        # Incoming packet stream (on circuit level)
        vc_in = vertcat(*[v_in_i*cv_in_i for v_in_i, cv_in_i in zip(v_in_list, cv_in)])

        """ Outgoing packet stream """
        # Outgoing packet stream:
        v_out = SX.sym('v_out', self.n_out, 1)
        # Maximum value for v_out:
        v_out_max = SX.sym('v_out', self.n_out, 1)
        # Outgoing packet stream (as list)
        v_out_list = vertsplit(v_out)
        # v_out from the previous solution:
        v_out_prev = SX.sym('v_out_prev', self.n_out, 1)

        """ Load information """
        # bandwidth / memory info outgoing servers
        bandwidth_load_target = SX.sym('bandwidth_load_target', self.n_out, 1)
        memory_load_target = SX.sym('memory_load_target', self.n_out, 1)
        # bandwidth / memory info incoming servers
        bandwidth_load_source = SX.sym('bandwidth_load_target', self.n_in, 1)
        memory_load_source = SX.sym('memory_load_target', self.n_in, 1)

        """ Circuit matching """
        # Assignment Matrix: Which element of each input is assigned to which output buffer:
        Pb = SX.sym('Pb', self.n_out, np.sum(self.n_circuit_in))
        # Assignment Matrix: Which input circuit is directed to which output circuit:
        Pc = SX.sym('Pc', np.sum(self.n_circuit_in), np.sum(self.n_circuit_in))

        """ System dynamics and constraints and objective"""
        # system dynamics, constraints and objective definition:
        s_tilde_next = s_buffer + self.dt*Pb@vc_in
        sc_tilde_next = s_circuit + self.dt*Pc@vc_in

        cv_out = [sc_i/s_tilde_next[i] for i, sc_i in enumerate(vertsplit(sc_tilde_next, np.cumsum([0]+self.n_circuit_out)))]
        vc_out = vertcat(*[v_out_i*cv_out_i for v_out_i, cv_out_i in zip(v_out_list, cv_out)])

        s_next = s_tilde_next - self.dt*v_out
        sc_next = sc_tilde_next - self.dt*vc_out
        # Note: cons <= 0
        cons = [
            # maximum bandwidth cant be exceeded
            sum1(v_in)+sum1(v_out) - self.v_max,
            -s_buffer,  # buffer memory cant be <0 (for each output buffer)
            -v_out,  # outgoing package stream cant be negative
            v_in_max-self.v_max,  # v_in_max cant be greater than self.v_max.
            v_out-v_out_max  # v_out cant be greater than v_out_max
        ]
        cons = vertcat(*cons)
        # Maximize bandwidth  and maximize buffer:(under consideration of outgoing server load)
        # Note that 0<bandwidth_load_target<1 and memory_load_target is normalized by s_max but can exceed 1.
        obj = sum1(((1-bandwidth_load_target)/fmax(memory_load_target, 1))*(-v_out/self.v_max+s_buffer/self.s_max))
        obj += sum1(((1-bandwidth_load_source)/fmax(memory_load_source, 1))*(-v_in_req/self.v_max))
        obj += self.v_delta_penalty*(sum1(((v_out-v_out_prev)/self.v_max)**2)+sum1(((v_in_max-v_in_max_prev)/self.v_max)**2))

        """ Problem dictionary """
        mpc_problem = {}
        mpc_problem['cons'] = Function('cons', [v_in_max, v_in_req, v_out, v_out_max, s_buffer], [cons])
        mpc_problem['obj'] = Function('obj', [v_in_req, v_in_max, v_in_max_prev, v_out, v_out_prev, s_buffer, bandwidth_load_target, memory_load_target, bandwidth_load_source, memory_load_source], [obj])
        mpc_problem['model'] = Function('model', [s_buffer, s_circuit, v_in_req, v_in_max, *cv_in, v_out, Pb, Pc], [s_next, sc_next, *cv_out])

        self.mpc_problem = mpc_problem

    def create_optim(self):
        # Initialize trajectory lists (each list item, one time-step):
        s_buffer = []  # [s_1, s_2, ..., s_N]
        s_circuit = []
        v_in_req = []
        v_in_max = []
        v_in_max_prev = []  # solution from previous timestep
        cv_in = []
        v_out = []
        v_out_prev = []  # solution from the previous timestep.
        v_out_max = []
        cv_out = []
        bandwidth_load_target = []
        memory_load_target = []
        bandwidth_load_source = []
        memory_load_source = []
        cons = []

        # Constant values:
        # Assignment Matrix: Which element of each input is assigned to which output buffer:
        Pb = SX.sym('Pb', self.n_out, np.sum(self.n_circuit_in))
        # Assignment Matrix: Which input circuit is directed to which output circuit:
        Pc = SX.sym('Pc', np.sum(self.n_circuit_in), np.sum(self.n_circuit_in))

        # Initialize objective value:
        obj = 0

        # Initial condition:
        s_buffer_0 = SX.sym('s_buffer_0', self.n_out, 1)
        s_circuit_0 = SX.sym('s_circuit_0', np.sum(self.n_circuit_out), 1)

        # Recursively evaluate system equation and add stage cost and stage constraints:
        for k in range(self.N_steps):

            # Incoming packet stream
            v_in_req.append(SX.sym('v_in_req', self.n_in, 1))
            v_in_max.append(SX.sym('v_in_max', self.n_in, 1))
            # Previous outgoing packet stream:
            v_in_max_prev.append(SX.sym('v_in_max_prev', self.n_in, 1))
            cv_in.append([SX.sym('cv_in_{0}_{1}'.format(k, i), self.n_circuit_in[i], 1) for i in range(self.n_in)])

            # Outgoing packet stream
            v_out.append(SX.sym('v_out', self.n_out, 1))
            # Maximum value for outgoing packet stream:
            v_out_max.append(SX.sym('v_out_max', self.n_out, 1))
            # Previous outgoing packet stream:
            v_out_prev.append(SX.sym('v_out_prev', self.n_out, 1))

            # bandwidth / memory info outgoing servers
            bandwidth_load_target.append(SX.sym('bandwidth_load_target', self.n_out, 1))
            memory_load_target.append(SX.sym('memory_load_target', self.n_out, 1))

            # bandwidth / memory info incoming servers
            bandwidth_load_source.append(SX.sym('bandwidth_load_source', self.n_in, 1))
            memory_load_source.append(SX.sym('memory_load_source', self.n_in, 1))

            # For the first step use s_buffer_0
            if k == 0:
                s_buffer_k = s_buffer_0
                s_circuit_k = s_circuit_0
            else:  # In suceeding steps use the previous s
                s_buffer_k = s_buffer[k-1]
                s_circuit_k = s_circuit[k-1]

            model_out = self.mpc_problem['model'](s_buffer_k, s_circuit_k, v_in_req[k], v_in_max[k], *cv_in[k], v_out[k], Pb, Pc)
            s_buffer.append(model_out[0])
            s_circuit.append(model_out[1])
            cv_out.append(model_out[2:])

            # Add the "stage cost" to the objective
            obj += self.mpc_problem['obj'](v_in_req[k], v_in_max[k], v_in_max_prev[k], v_out[k], v_out_prev[k], s_buffer[k], bandwidth_load_target[k], memory_load_target[k],
                                           bandwidth_load_source[k], memory_load_source[k])

            # Constraints for the current step
            cons.append(self.mpc_problem['cons'](v_in_max[k], v_in_req[k], v_out[k], v_out_max[k], s_buffer[k]))

        optim_dict = {'x': vertcat(*v_out, *v_in_max),    # Optimization variable
                      'f': obj,                        # objective
                      'g': vertcat(*cons),            # constraints (Note: cons<=0)
                      'p': vertcat(s_buffer_0, s_circuit_0, *v_in_req, *v_in_max_prev, *[j for i in cv_in for j in i], *v_out_max, *v_out_prev,
                                   Pb.reshape((-1, 1)), Pc.reshape((-1, 1)), *bandwidth_load_target,
                                   *memory_load_target, *bandwidth_load_source, *memory_load_source)}  # parameters

        # Create casadi optimization object:
        self.optim = nlpsol('optim', 'ipopt', optim_dict)
        # Create function to calculate buffer memory from parameter and optimization variable trajectories
        self.aux_fun = Function('aux_fun', [s_buffer_0, s_circuit_0]+v_in_max+v_in_req+[j for i in cv_in for j in i]+v_out+[Pb, Pc],
                                s_buffer+s_circuit+[j for i in cv_out for j in i])

    def solve(self, s_buffer_0, s_circuit_0, v_in_req, cv_in, v_out_max, Pb, Pc, bandwidth_load_target, memory_load_target, bandwidth_load_source, memory_load_source):
        """
        Solves the optimal control problem defined in optimal_traffic_scheduler.problem_formulation().
        Inputs:
        - s_buffer_0            : initial memory for each buffer (must be n_out x 1 vector)
        - s_circuit_0           : intitial memory for each circuit (must be np.sum(n_circuit_out) x 1 vector)
        Predicted trajectories (as lists with N_horizon elments):
        - v_in_req              : Requested incoming package stream for each buffer (n_in x 1 vector)
        - cv_in                : Composition of incoming streams. (n_in x 1 list with n_circuit_in[i] elements for list item i)
        - bandwidth_load_target : Bandwidth load of target server(s) (n_out x 1 vector)
        - memory_load_target    : Memory load of target server(s) (n_out x 1 vector)
        - bandwidth_load_source : Bandwidth load of source server(s) (n_in x 1 vector)
        - memory_load_source    : Memory load of source server(s) (n_in x 1 vector)

        Populates the "predict" and "record" dictonaries of the class.
        - Predict: Constantly overwritten variables that save the current optimized state and control trajectories of the node
        - Record:  Lists with items appended for each call of solve, recording the past states of the node.

        Returns the predict dictionary. Note that the predict dictionary contains some of the inputs trajectories as well as the
        calculated optimal trajectories.

        "Solve" also advances the time of the node by one time_step.
        """
        # Get previous solution:
        v_out_prev = self.predict[-1]['v_out']
        v_in_max_prev = self.predict[-1]['v_in_max']
        # From the perspecitve of the current timestep:
        v_out_0 = v_out_prev[1:]+[v_out_prev[-1]]
        v_in_max_0 = v_in_max_prev[1:]+[v_in_max_prev[-1]]
        # Create concatented parameter vector:
        param = np.concatenate((s_buffer_0, s_circuit_0, *v_in_req, *v_in_max_0, *[j for i in cv_in for j in i], *v_out_max, *v_out_0,
                                Pb.reshape((-1, 1)), Pc.reshape((-1, 1)), *bandwidth_load_target,
                                *memory_load_target, *bandwidth_load_source, *memory_load_source), axis=0)
        # Get initial condition:
        x0 = np.concatenate(v_out_0+v_in_max_0, axis=0)
        # Solve optimization problem for given conditions:
        optim_results = self.optim(ubg=0, p=param, x0=x0)  # Note: constraints were formulated, such that cons<=0.

        # Retrieve trajectory from solution:
        optim_sol = optim_results['x'].full()
        v_out, v_in_max = np.split(optim_sol, [self.N_steps*self.n_out])

        v_out = [v_out_i.reshape(-1, 1) for v_out_i in np.split(v_out, self.N_steps)]
        v_in_max = [v_in_max_i.reshape(-1, 1) for v_in_max_i in np.split(v_in_max, self.N_steps)]

        # Calculate additional trajectories:
        aux_values = self.aux_fun(s_buffer_0, s_circuit_0, *v_in_max, *v_in_req, *[j for i in cv_in for j in i], *v_out, Pb, Pc)
        aux_values = [aux_i.full() for aux_i in aux_values]
        #s_buffer+s_circuit+[j for i in cv_out for j in i]
        s_buffer, s_circuit, cv_out = self.split_list(aux_values, [self.N_steps, 2*self.N_steps])
        cv_out = self.split_list(cv_out, self.n_out)
        v_in = [np.minimum(v_in_req_i, v_in_max_i) for v_in_req_i, v_in_max_i in zip(v_in_req, v_in_max)]

        # Calculate trajectory for bandwidth and memory:
        bandwidth_load_node = [(np.sum(v_out_i, keepdims=True)+np.sum(v_in_i, keepdims=True))/self.v_max for v_out_i,
                               v_in_i in zip(v_out, v_in)]

        memory_load_node = [np.sum(s_buffer_i, keepdims=True)/self.s_max for s_buffer_i in s_buffer]

        self.time += self.dt

        self.predict.append({})
        self.predict[-1]['v_in'] = v_in
        self.predict[-1]['v_in_max'] = v_in_max
        self.predict[-1]['v_in_req'] = v_in_req
        self.predict[-1]['cv_in'] = cv_in
        self.predict[-1]['v_out'] = v_out
        self.predict[-1]['v_out_max'] = v_out_max
        self.predict[-1]['cv_out'] = cv_out
        self.predict[-1]['s_buffer'] = s_buffer
        self.predict[-1]['s_circuit'] = s_circuit
        self.predict[-1]['bandwidth_load'] = bandwidth_load_node
        self.predict[-1]['memory_load'] = memory_load_node
        self.predict[-1]['bandwidth_load_target'] = np.copy(bandwidth_load_target)
        self.predict[-1]['memory_load_target'] = np.copy(memory_load_target)
        self.predict[-1]['bandwidth_load_source'] = np.copy(bandwidth_load_source)
        self.predict[-1]['memory_load_source'] = np.copy(memory_load_source)

        if self.record_values:
            self.record['time'].append(np.copy(self.time))
            for key, val in self.predict[-1].items():
                self.record[key].append(val[0])
        return self.predict

    def latency_adaption(self, v_in_circuit, bandwidth_load_target, memory_load_target, input_delay, output_delay):
        """
        Adapting the incoming predictions due to latency.
        """
        assert self.dt > np.max(input_delay) and self.dt > np.max(output_delay), "Delays that are greater than one optimization time step are currently not supported."
        # Extend the current predictions by repeating the end value.
        # This is required since we need to extrapolate further than the current horizon, when information is delayed.
        v_in_circuit_ext = np.concatenate((v_in_circuit, v_in_circuit[[-1]]), axis=0)
        # Delay of incoming connections affect v_in_circuit. (n_timesteps x n_components x 1 tensor)
        t_in = self.dt*np.arange(-1, self.N_steps).reshape(-1, 1, 1)+input_delay.reshape(1, -1, 1)
        # At these times the values will be interpolated:
        t_interp_in = self.dt*np.arange(self.N_steps).reshape(-1, 1, 1)+np.zeros((1, v_in_circuit_ext.shape[1], 1))
        v_in_circuit_interp = self.interpol_nd(t_in, v_in_circuit_ext, t_interp_in)

        # Bandwidth and memory load of receiving servers are treated differently. Again, we are looking at "old" information,
        # but now we also need to take into accound that the actions at the current node influence the receiving nodes in the
        # future. This means we discard the first element of bandwidth_load_target and memory_load_target as they lie completely in the past.
        # Furthermore, we need to extend the prediction by two further timesteps, which is achieved by repeating the end value twice.
        bandwidth_load_target_ext = np.concatenate((bandwidth_load_target[1:], np.repeat(bandwidth_load_target[[-1]], 2, axis=0)), axis=0)
        memory_load_target_ext = np.concatenate((memory_load_target[1:], np.repeat(memory_load_target[[-1]], 2, axis=0)), axis=0)
        # The time at which the truncated predictions will be valid:
        t_out = self.dt*np.arange(self.N_steps+1).reshape(-1, 1, 1)  # (0, 1, 2 ... , N+1)
        # The time at which the current action will affect the receiving server:
        t_interp_out = self.dt*np.arange(self.N_steps).reshape(-1, 1, 1)+output_delay.reshape(1, -1, 1)  # (0+d, 1+d, 2+d ... , N+d)

        bandwidth_load_target_interp = self.interpol_nd(t_out, bandwidth_load_target_ext, t_interp_out)
        memory_load_target_interp = self.interpol_nd(t_out, memory_load_target_ext, t_interp_out)

        return v_in_circuit_interp, bandwidth_load_target_interp, memory_load_target_interp

    @staticmethod
    def interpol_nd(x, y, x_new, axis=0):
        """
        Very simple and fast interpolation of a function y=f(x) where x and x_new are ordered sequences and the following is true:
        x[k]<= x_new[k] <= x[k+1] for all k=0, ..., len(x_new)
        Sequence data is supplied in axis=axis (default = 0).
        """

        dx = np.diff(x, axis=axis)
        dy = np.diff(y, axis=axis)
        y = np.delete(y, -1, axis=axis)
        x = np.delete(x, -1, axis=axis)
        return y+dy/dx*(x_new-x)

    @staticmethod
    def split_list(arr, ind):
        """ Splits list into multiple sub-lists. Mimics numpy.split.
        Parameters:
        - arr: list with n elements.
        - ind: Either integer index or list of integer indices. When
          - integer: Split list in N equal lists of lenght ind. Raise error if list cant be split into sections of equal length.
          - list   : Split list at indices of ind. Such that, e.g: ind=[2,3] -> list[:2], list[2:3], list[3:]
        """
        if type(ind) == int:
            assert np.mod(len(arr), ind) == 0, "List of length {0} can't be split into {1} "
            split_arr = [arr[ind*i:ind*(i+1)] for i in range(len(arr)//ind)]
        if type(ind) == list or type(ind) == tuple:
            ind = [0]+ind+[len(arr)]
            split_arr = [arr[ind[i]:ind[i+1]] for i in range(len(ind)-1)]

        return split_arr
