import numpy as np
import matplotlib.pyplot as plt
from casadi import *
import pdb
from scipy.linalg import block_diag


class optimal_traffic_scheduler:
    def __init__(self, setup_dict, name='ots', record_values=True):
        self.obj_name = name
        self.v_max = setup_dict['v_max']
        self.s_max = setup_dict['s_max']
        self.dt = setup_dict['dt']
        self.N_steps = setup_dict['N_steps']
        self.weights = setup_dict['weights']
        self.record_values = record_values
        self.time = np.array([[0]])  # 1,1 array for consistency.

    def setup(self, n_in=None, n_out=None, input_circuits=None, output_circuits=None, output_delay=0):
        """
        n_in: Number of Inputs
        n_out: Number of outputs
        input_circuits: List with each item being a list of identifiers for the input circuits
        output_circuits: List with each item being a list of identifiers for the output circuits
        """

        self.n_in = n_in
        self.n_out = n_out

        self.n_circuit_in = [len(c_i) for c_i in input_circuits]
        self.n_circuit_out = [len(c_i) for c_i in output_circuits]

        self.Pb = self.Pb_fun(input_circuits, output_circuits)
        self.Pc = self.Pc_fun(input_circuits, output_circuits)

        assert len(self.n_circuit_in) == self.n_in
        assert len(self.n_circuit_out) == self.n_out
        assert np.sum(self.n_circuit_in) == np.sum(self.n_circuit_out)

        self.initialize_prediction()
        if self.record_values:
            self.initialize_record()
        self.problem_formulation()
        self.create_optim(output_delay)

    def initialize_prediction(self):
        # Initial conditions: all zeros.
        self.predict = [{
            'v_out': [np.zeros((self.n_out, 1))]*self.N_steps,
            'v_out_max': [np.zeros((self.n_out, 1))]*self.N_steps,
            'cv_out': [[np.zeros((n_circuit_out_i, 1)) for n_circuit_out_i in self.n_circuit_out]]*self.N_steps,
            'v_in': [np.zeros((self.n_in, 1))]*self.N_steps,
            'v_in_req': [np.zeros((self.n_in, 1))]*self.N_steps,
            'v_in_max': [self.v_max/(self.n_in+self.n_out)*np.ones((self.n_in, 1))]*self.N_steps,
            'cv_in': [[np.zeros((n_circuit_in_i, 1)) for n_circuit_in_i in self.n_circuit_in]]*self.N_steps,
            's_buffer': [np.zeros((self.n_out, 1))]*self.N_steps,
            's_transit': [np.zeros((self.n_out, 1))]*self.N_steps,
            's_circuit': [np.zeros((np.sum(self.n_circuit_in), 1))]*self.N_steps,
            'bandwidth_load': [np.zeros((1, 1))]*self.N_steps,
            'memory_load': [np.zeros((1, 1))]*self.N_steps,
            'bandwidth_load_target': [np.zeros((self.n_out, 1))]*self.N_steps,
            'memory_load_target': [np.zeros((self.n_out, 1))]*self.N_steps,
            'bandwidth_load_source': [np.zeros((self.n_in, 1))]*self.N_steps,
            'memory_load_source': [np.zeros((self.n_in, 1))]*self.N_steps,
        }]

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
        # Buffer memory for packets that are currently in transit (expected to be cleared after roundtrip time)
        s_transit = SX.sym('s_transit', self.n_out, 1)
        # Circuit memory:
        s_circuit = SX.sym('s_circuit', np.sum(self.n_circuit_in), 1)

        """ Incoming packet stream """
        # Incoming package stream for each buffer:
        v_in_req = SX.sym('v_in_req', self.n_in, 1)
        # Discarded packet stream:
        v_in_discard = SX.sym('v_in_max', self.n_in, 1)
        # Additional packet contigent that could be accepted.
        v_in_extra = SX.sym('v_in_extra', self.n_in, 1)
        # v_in_max from previous solution
        v_in_max_prev = SX.sym('v_in_max_prev', self.n_in, 1)
        # Resulting incoming packet stream:
        v_in = v_in_req - v_in_discard
        # Allowed incoming packet stream:
        v_in_max = v_in + v_in_extra
        # Concatenate v_in:
        v_in_list = vertsplit(v_in)
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
        # packet stream that is removed from s_buffer_transit:
        v_tr_remove = SX.sym('v_tr_remove', self.n_out, 1)

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

        eps = 1e-9
        cv_out = [sc_i/(s_tilde_next[i]+eps) for i, sc_i in enumerate(vertsplit(sc_tilde_next, np.cumsum([0]+self.n_circuit_out)))]
        #cv_out = [if_else(s_tilde_next[i] > 0, sc_i/s_tilde_next[i], 0*sc_i) for i, sc_i in enumerate(vertsplit(sc_tilde_next, np.cumsum([0]+self.n_circuit_out)))]
        vc_out = vertcat(*[v_out_i*cv_out_i for v_out_i, cv_out_i in zip(v_out_list, cv_out)])

        s_next = s_tilde_next - self.dt*v_out
        sc_next = sc_tilde_next - self.dt*vc_out
        s_transit_next = s_transit + self.dt*(v_out - v_tr_remove)

        cons_list = [
            # maximum bandwidth cant be exceeded
            #{'lb': [-np.inf], 'eq': sum1(v_in)+sum1(v_out) - self.v_max, 'ub': [0]},
            {'lb': [-np.inf], 'eq': sum1(v_in_max)+sum1(v_out)-self.v_max, 'ub': [0]},
            {'lb': [-np.inf]*self.n_in, 'eq': -v_in, 'ub': [0]*self.n_in},  # v_in cant be negative
            {'lb': [-np.inf]*self.n_in, 'eq': -v_in_discard, 'ub': [0]*self.n_in},  # discarded packet stream cant be negative
            {'lb': [-np.inf]*self.n_in, 'eq': -v_in_extra, 'ub': [0]*self.n_in},  # additional incoming packet stream cant be negative
            {'lb': [0]*self.n_in, 'eq': v_in_discard*v_in_extra, 'ub': [0]*self.n_in},  # packets can be discarded or added (not both)
            {'lb': [-np.inf]*self.n_out, 'eq': -s_buffer, 'ub': [0]*self.n_out},  # buffer memory cant be <0 (for each output buffer)
            {'lb': [-np.inf], 'eq': sum1(s_buffer)-self.s_max, 'ub': [0]},  # buffer memory cant exceed s_max
            {'lb': [-np.inf]*self.n_out, 'eq': -v_out, 'ub': [0]*self.n_out},  # outgoing packet stream cant be negative
            {'lb': [-np.inf]*self.n_out, 'eq': v_out-v_out_max, 'ub': [0]*self.n_out},  # outgoing packet stream cant be negative
        ]
        cons = vertcat(*[con_i['eq'] for con_i in cons_list])
        cons_lb = np.concatenate([con_i['lb'] for con_i in cons_list])
        cons_ub = np.concatenate([con_i['ub'] for con_i in cons_list])
        # Maximize bandwidth  and maximize buffer:(under consideration of outgoing server load)
        # Note that 0<bandwidth_load_target<1 and memory_load_target is normalized by s_max but can exceed 1.
        obj = sum1((1-bandwidth_load_target)*(1-memory_load_target)*(-self.weights['send']*v_out/self.v_max+self.weights['store']*s_buffer/self.s_max))
        obj += sum1((1+bandwidth_load_source*memory_load_source)*(self.weights['receive']*(v_in_discard-v_in_extra)/self.v_max))
        obj += self.weights['control_delta']*(sum1(((v_out-v_out_prev)/self.v_max)**2)+sum1(((v_in_max-v_in_max_prev)/self.v_max)**2))

        """ Problem dictionary """
        mpc_problem = {}
        mpc_problem['cons'] = Function('cons', [v_in_discard, v_in_extra, v_in_req, v_out, v_out_max, s_buffer], [cons])
        mpc_problem['cons_lb'] = cons_lb
        mpc_problem['cons_ub'] = cons_ub
        mpc_problem['obj'] = Function('obj', [v_in_req, v_in_discard, v_in_extra, v_in_max_prev, v_out, v_out_prev, s_buffer, bandwidth_load_target, memory_load_target, bandwidth_load_source, memory_load_source], [obj])
        mpc_problem['model'] = Function('model', [s_buffer, s_transit, s_circuit, v_tr_remove, v_in_req, v_in_discard, v_in_extra, *cv_in, v_out, Pb, Pc], [s_next, s_transit_next, sc_next, *cv_out])
        mpc_problem['aux'] = Function('aux', [v_in_req, v_in_discard, v_in_extra], [v_in_max])

        self.mpc_problem = mpc_problem

    def create_optim(self, output_delay):
        # Initialize trajectory lists (each list item, one time-step):
        s_buffer = []  # [s_1, s_2, ..., s_N]
        s_transit = []
        s_circuit = []
        v_tr_remove = []
        v_tr_remove_interp = []
        v_in_req = []
        v_in_max = []
        v_in_discard = []
        v_in_extra = []
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
        cons_ub = []
        cons_lb = []

        # For v_tr_remove we need to consider the latency (roundtrip time) and the timestep of the optimizer:
        n_tr_remove = np.int32(np.ceil(2*output_delay/self.dt)).tolist()

        # Constant values:
        # Assignment Matrix: Which element of each input is assigned to which output buffer:
        Pb = SX.sym('Pb', self.n_out, np.sum(self.n_circuit_in))
        # Assignment Matrix: Which input circuit is directed to which output circuit:
        Pc = SX.sym('Pc', np.sum(self.n_circuit_in), np.sum(self.n_circuit_in))

        # Initialize objective value:
        obj = 0

        # Initial condition:
        s_buffer_0 = SX.sym('s_buffer_0', self.n_out, 1)
        s_transit_0 = SX.sym('s_transit_0', self.n_out, 1)
        s_circuit_0 = SX.sym('s_circuit_0', np.sum(self.n_circuit_out), 1)

        assert all(output_delay > 0), 'Output delay cant be zero.'
        # Initial value for v_transit_remove
        v_tr_remove.append(vertcat(*[SX.sym('v_tr_remove_{0}_{1}'.format(-n_tr_remove[i], i), 1, 1) for i in range(self.n_out)]))

        # Recursively evaluate system equation and add stage cost and stage constraints:
        for k in range(self.N_steps):

            # Incoming packet stream
            v_in_req.append(SX.sym('v_in_req', self.n_in, 1))
            v_in_discard.append(SX.sym('v_in_discard', self.n_in, 1))
            v_in_extra.append(SX.sym('v_in_extra', self.n_in, 1))
            v_in_max.append(self.mpc_problem['aux'](v_in_req[k], v_in_discard[k], v_in_extra[k]))
            # Previous outgoing packet stream:
            v_in_max_prev.append(SX.sym('v_in_max_prev', self.n_in, 1))
            cv_in.append([SX.sym('cv_in_{0}_{1}'.format(k, i), self.n_circuit_in[i], 1) for i in range(self.n_in)])

            # Outgoing packet stream
            v_out.append(SX.sym('v_out', self.n_out, 1))

            # Maximum value for outgoing packet stream:
            v_out_max.append(SX.sym('v_out_max', self.n_out, 1))
            # Previous outgoing packet stream:
            v_out_prev.append(SX.sym('v_out_prev', self.n_out, 1))

            # Packet stream that is removed from transit buffer. This is not synchronized and has to be interpolated.
            v_tr_remove.append(vertcat(*[SX.sym('v_tr_remove_{0}_{1}'.format(k+1-n_tr_remove[i], i), 1, 1) if k+1 < n_tr_remove[i] else v_out[-n_tr_remove[i]][i] for i in range(self.n_out)]))
            t_interp = np.mod(2*output_delay, self.dt)/self.dt
            v_tr_remove_interp.append(v_tr_remove[k]+(v_tr_remove[k+1]-v_tr_remove[k])*t_interp.reshape(-1, 1))

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
                s_transit_k = s_transit_0
            else:  # In suceeding steps use the previous s
                s_buffer_k = s_buffer[k-1]
                s_circuit_k = s_circuit[k-1]
                s_transit_k = s_transit[k-1]
            model_out = self.mpc_problem['model'](s_buffer_k, s_transit_k, s_circuit_k, v_tr_remove_interp[k], v_in_req[k],  v_in_discard[k], v_in_extra[k], *cv_in[k], v_out[k], Pb, Pc)
            s_buffer.append(model_out[0])
            s_transit.append(model_out[1])
            s_circuit.append(model_out[2])
            cv_out.append(model_out[3:])

            # Add the "stage cost" to the objective
            obj += self.mpc_problem['obj'](v_in_req[k], v_in_discard[k], v_in_extra[k], v_in_max_prev[k], v_out[k], v_out_prev[k], s_buffer[k], bandwidth_load_target[k], memory_load_target[k],
                                           bandwidth_load_source[k], memory_load_source[k])

            # Constraints for the current step
            cons.append(self.mpc_problem['cons'](v_in_discard[k], v_in_extra[k], v_in_req[k], v_out[k], v_out_max[k], s_buffer[k]))
            cons_lb.append(self.mpc_problem['cons_lb'])
            cons_ub.append(self.mpc_problem['cons_ub'])

        # Get unique elements of the packet stream that is removed from buffer. These are parameters of the optimization problem
        v_tr_remove = vertcat(*v_tr_remove)
        v_tr_remove_unique = [v_tr_remove[i] for i in range(v_tr_remove.shape[0]) if 'v_tr_remove' in v_tr_remove[i].name()]

        optim_dict = {'x': vertcat(*v_out, *v_in_discard, *v_in_extra),    # Optimization variable
                      'f': obj,                        # objective
                      'g': vertcat(*cons),            # constraints (Note: cons<=0)
                      'p': vertcat(s_buffer_0, s_circuit_0, s_transit_0, *v_tr_remove_unique, *v_in_req, *v_in_max_prev, *[j for i in cv_in for j in i], *v_out_max, *v_out_prev,
                                   Pb.reshape((-1, 1)), Pc.reshape((-1, 1)), *bandwidth_load_target,
                                   *memory_load_target, *bandwidth_load_source, *memory_load_source)}  # parameters
        self.cons_lb = np.concatenate(cons_lb)
        self.cons_ub = np.concatenate(cons_ub)
        # Create casadi optimization object:
        opts = {'ipopt.linear_solver': 'MA27'}
        self.optim = nlpsol('optim', 'ipopt', optim_dict, opts)
        # Create function to calculate buffer memory from parameter and optimization variable trajectories
        self.aux_fun = Function('aux_fun', [s_buffer_0, s_circuit_0, s_transit_0]+v_tr_remove_unique+v_in_discard+v_in_extra+v_in_req+[j for i in cv_in for j in i]+v_out+[Pb, Pc],
                                s_buffer+s_circuit+s_transit+v_in_max+[j for i in cv_out for j in i])

    def solve(self, s_buffer_0, s_circuit_0, s_transit_0, v_in_req, cv_in, v_out_max, bandwidth_load_target, memory_load_target, bandwidth_load_source, memory_load_source, output_delay):
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
        v_out_prev_pred = self.predict[-1]['v_out']
        v_in_max_prev = self.predict[-1]['v_in_max']
        v_in_req_prev = self.predict[-1]['v_in_req']
        v_in_prev = self.predict[-1]['v_in']
        # From the perspecitve of the current timestep:
        v_out_0 = v_out_prev_pred[1:]+[v_out_prev_pred[-1]]
        v_in_max_0 = v_in_max_prev[1:]+[v_in_max_prev[-1]]
        v_in_req_0 = v_in_req_prev[1:]+[v_in_req_prev[-1]]
        v_in_0 = v_in_prev[1:]+[v_in_prev[-1]]

        v_in_extra_0 = [v_in_max_i-v_in_i for v_in_max_i, v_in_i in zip(v_in_max_0, v_in_0)]
        v_in_discard_0 = [v_in_req_i-v_in_i for v_in_req_i, v_in_i in zip(v_in_req_0, v_in_0)]

        # Get previous v_out values to calculate v_transir_remove:
        n_tr_remove = np.int32(np.ceil(2*output_delay/self.dt))
        n_tr_remove_max = np.max(n_tr_remove)
        len_record = len(self.record['v_out'])
        v_out_prev = np.concatenate(self.record['v_out'][-np.minimum(n_tr_remove_max, len_record):], axis=1)

        # For the first steps, extend the record, if necessary:
        if len_record < n_tr_remove_max:
            v_out_prev = np.concatenate((np.repeat(v_out_prev[:, [0]], n_tr_remove_max-len_record, axis=1), v_out_prev), axis=1)
        v_tr_remove_unique = [v_out_prev[n_tr_remove > i, i-n_tr_remove[n_tr_remove > i]].reshape(-1, 1) for i in range(n_tr_remove_max)]
        v_tr_remove_unique = [v_tr_remove_unique_i.reshape(-1, 1) for v_tr_remove_unique_i in np.concatenate(v_tr_remove_unique)]

        # Create concatented parameter vector:
        param = np.concatenate((s_buffer_0, s_circuit_0, s_transit_0, *v_tr_remove_unique, *v_in_req, *v_in_max_0, *[j for i in cv_in for j in i], *v_out_max, *v_out_0,
                                self.Pb.reshape((-1, 1)), self.Pc.reshape((-1, 1)), *bandwidth_load_target,
                                *memory_load_target, *bandwidth_load_source, *memory_load_source), axis=0)
        # Get initial condition:
        x0 = np.concatenate(v_out_0+v_in_discard_0+v_in_extra_0, axis=0)
        # Solve optimization problem for given conditions:
        optim_results = self.optim(ubg=self.cons_ub, lbg=self.cons_lb, p=param, x0=x0)  # Note: constraints were formulated, such that cons<=0.
        optim_stats = self.optim.stats()

        # Retrieve trajectory from solution:
        optim_sol = optim_results['x'].full()
        v_out, v_in_discard, v_in_extra = np.split(optim_sol, [self.N_steps*self.n_out, self.N_steps*(self.n_out+self.n_in)])

        v_out = [v_out_i.reshape(-1, 1) for v_out_i in np.split(v_out, self.N_steps)]
        v_in_discard = [v_in_discard_i.reshape(-1, 1) for v_in_discard_i in np.split(v_in_discard, self.N_steps)]
        v_in_extra = [v_in_extra_i.reshape(-1, 1) for v_in_extra_i in np.split(v_in_extra, self.N_steps)]
        # Calculate additional trajectories:
        aux_values = self.aux_fun(s_buffer_0, s_circuit_0, s_transit_0, *v_tr_remove_unique, *v_in_discard, *v_in_extra, *v_in_req, *[j for i in cv_in for j in i], *v_out, self.Pb, self.Pc)
        aux_values = [aux_i.full() for aux_i in aux_values]
        # s_buffer+s_circuit+[j for i in cv_out for j in i]
        s_buffer, s_circuit, s_transit, v_in_max, cv_out = self.split_list(aux_values, (np.array([1, 2, 3, 4])*self.N_steps).tolist())
        cv_out = self.split_list(cv_out, self.n_out)
        v_in = [v_in_req_i - v_in_discard_i for v_in_req_i, v_in_discard_i in zip(v_in_req, v_in_discard)]

        # Calculate trajectory for bandwidth and memory:
        bandwidth_load_node = [(np.sum(v_out_i, keepdims=True)+np.sum(v_in_i, keepdims=True))/self.v_max for v_out_i,
                               v_in_i in zip(v_out, v_in)]

        memory_load_node = [np.sum(s_buffer_i, keepdims=True)/self.s_max for s_buffer_i in s_buffer]

        self.time = self.time + self.dt

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
        self.predict[-1]['s_transit'] = s_transit
        self.predict[-1]['bandwidth_load'] = bandwidth_load_node
        self.predict[-1]['memory_load'] = memory_load_node
        self.predict[-1]['bandwidth_load_target'] = np.copy(bandwidth_load_target)
        self.predict[-1]['memory_load_target'] = np.copy(memory_load_target)
        self.predict[-1]['bandwidth_load_source'] = np.copy(bandwidth_load_source)
        self.predict[-1]['memory_load_source'] = np.copy(memory_load_source)

        if self.record_values:
            self.record_fun()

        if not optim_stats['success']:
            pdb.set_trace()
        return optim_stats['success']

    def record_fun(self):
        self.record['time'].append(np.copy(self.time))
        for key, val in self.predict[-1].items():
            self.record[key].append(val[0])

    def latency_adaption(self, sequence, type, input_delay=np.array([0]), output_delay=np.array([0])):
        """
        Adapting the incoming predictions due to latency.
        """
        assert self.dt > np.max(input_delay) and self.dt > np.max(output_delay), "Delays that are greater than one optimization time step are currently not supported."
        if type == 'input':
            # Extend the current predictions by repeating the end value.
            # This is required since we need to extrapolate further than the current horizon, when information is delayed.
            sequence_ext = np.concatenate((sequence, sequence[[-1]]), axis=0)
            # Delay of incoming connections affect v_in_circuit. (n_timesteps x n_components x 1 tensor)
            t_in = self.dt*np.arange(-1, self.N_steps).reshape(-1, 1, 1)+input_delay.reshape(1, -1, 1)
            # At these times the values will be interpolated:
            t_interp_in = self.dt*np.arange(self.N_steps).reshape(-1, 1, 1)+np.zeros((1, sequence_ext.shape[1], 1))
            sequence_interp = self.interpol_nd(t_in, sequence_ext, t_interp_in)

        # Output sequences are treated differently. Again, we are looking at "old" information,
        # but now we also need to take into accound that the actions at the current node influence the receiving nodes in the
        # future. This means we discard the first element as they lie completely in the past.
        # Furthermore, we need to extend the prediction by two further timesteps, which is achieved by repeating the end value twice.
        if type == 'output':
            sequence_ext = np.concatenate((sequence[1:], np.repeat(sequence[[-1]], 2, axis=0)), axis=0)
            # The time at which the truncated predictions will be valid:
            t_out = self.dt*np.arange(self.N_steps+1).reshape(-1, 1, 1)  # (0, 1, 2 ... , N+1)
            # The time at which the current action will affect the receiving server:
            t_interp_out = self.dt*np.arange(self.N_steps).reshape(-1, 1, 1)+output_delay.reshape(1, -1, 1)  # (0+d, 1+d, 2+d ... , N+d)
            sequence_interp = self.interpol_nd(t_out, sequence_ext, t_interp_out)

        return sequence_interp

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
        if type(ind) in [int, np.int64, np.int32]:
            assert np.mod(len(arr), ind) == 0, "List of length {0} can't be split into {1} "
            split_arr = [arr[ind*i:ind*(i+1)] for i in range(len(arr)//ind)]
        if type(ind) in [list, tuple]:
            ind = [0]+ind+[len(arr)]
            split_arr = [arr[ind[i]:ind[i+1]] for i in range(len(ind)-1)]

        return split_arr

    @ staticmethod
    def Pb_fun(c_in, c_out, concatenate=True):
        """ Returns the assignment matrix that determines, which element of each input is assigned to which output buffer.
        Natively concatenates P to return on Matrix with n_out x n_circuits_in. With concatenate set to false, the staticmethod
        returns one matrix for each input.

        E.g.: 2 inputs with 3 and 2 circuits and three outputs with 2, 1 and 2 circuits each:

        input_circuits = [[0, 1, 2], [3, 4]]
        output_circuits = [[1, 3], [0], [2, 4]]

        concatenate = True :
        P = array([[0., 1., 0., 1., 0.],
                   [1., 0., 0., 0., 0.],
                   [0., 0., 1., 0., 1.]])
        concatenate = False :
        P[0] = array([[0., 1., 0.],
                      [1., 0., 0.],
                      [0., 0., 1.]])
        P[1] = array([[1., 0.],
                      [0., 0.],
                      [0., 1.]])]
        """
        P = [np.zeros((len(c_out), len(c_in_i))) for c_in_i in c_in]

        for i, c_in_i in enumerate(c_in):
            for j, c_in_ij in enumerate(c_in_i):
                k = [ind for ind, b_i in enumerate(c_out) if c_in_ij in b_i][0]
                P[i][k, j] = 1
        if concatenate:
            P = np.concatenate(P, axis=1)
        return P

    @ staticmethod
    def Pc_fun(c_in, c_out):
        """
        Returns the assignment matrix to determine which input circuit is directed to which output circuit.
        Pc is of dimension n_circuit x n_circuit.

        E.g.: 2 inputs with 3 and 2 circuits and three outputs with 2, 1 and 2 circuits each:

        input_circuits = [[0, 1, 2], [3, 4]]
        output_circuits = [[1, 3], [0], [2, 4]]

        Pc = array([[0., 1., 0., 0., 0.],
                    [0., 0., 0., 1., 0.],
                    [1., 0., 0., 0., 0.],
                    [0., 0., 1., 0., 0.],
                    [0., 0., 0., 0., 1.]])
        """
        c_in = [j for i in c_in for j in i]
        c_out = [j for i in c_out for j in i]
        Pc = np.zeros((len(c_out), len(c_in)))
        for i, c_in_i in enumerate(c_in):
            j = np.argwhere(c_in_i == np.array(c_out)).flatten()
            Pc[j, i] = 1
        return Pc


class ots_client(optimal_traffic_scheduler):
    def __init__(self, setup_dict, name='ots_client', record_values=True):
        self.obj_name = name
        self.v_max = setup_dict['v_max']
        self.s_max = setup_dict['s_max']
        self.dt = setup_dict['dt']
        self.N_steps = setup_dict['N_steps']
        self.record_values = record_values
        self.time = np.array([[0]])  # 1,1 array for consistency.

    def setup(self, n_in=None, n_out=None, input_circuits=None, output_circuits=None):
        """
        n_in: Number of Inputs
        n_out: Number of outputs
        input_circuits: List with each item being a list of identifiers for the input circuits
        e.g.: input_circuits = [[0, 1, 2], [3, 4]] (n_in=2)
        output_circuits: List with each item being a list of identifiers for the output circuits
        e.g.: output_circuits = [[1, 3], [0], [2, 4]] (n_out=3)
        """
        self.n_in = n_in
        self.n_out = n_out

        if n_in == 0:
            self.n_circuit_in = [0]
        else:
            self.n_circuit_in = [len(c_i) for c_i in input_circuits]
        if n_out == 0:
            self.n_circuit_out = [0]
        else:
            self.n_circuit_out = [len(c_i) for c_i in output_circuits]

        self.Pc = block_diag(*[np.ones((n_circuit_out_i, 1)) for n_circuit_out_i in self.n_circuit_out])

        super().initialize_prediction()
        if self.record_values:
            super().initialize_record()

    def update_prediction(self, s_buffer_0, s_circuit_0, v_out_max=None, v_in_req=None):
        """
        Depending on whether v_out_max or v_in is supplied (or both) this method will update
        the prediction for a receiving or a sending node.

        With v_out_max supplied: Sending node. Will update v_out prediction.
        With v_in supplied     : Receiving node. Will update v_in_max prediction.
        """

        if v_out_max:
            s_buffer = []
            s_circuit = []
            v_out = []
            cv_out = []
            for v_out_max_i in v_out_max:
                n_out_i = np.minimum(s_buffer_0, v_out_max_i*self.dt)
                s_buffer_0 = s_buffer_0-n_out_i
                # TODO: Only works for one circuit per input!
                s_circuit_0 = s_circuit_0-n_out_i
                v_out_i = n_out_i/self.dt
                s_buffer.append(s_buffer_0)
                s_circuit.append(s_circuit_0)
                v_out.append(v_out_i)
                cv_out.append([np.array([[1]])])
            self.predict[-1]['v_out'] = v_out
            self.predict[-1]['cv_out'] = cv_out
            self.predict[-1]['s_buffer'] = s_buffer
            self.predict[-1]['s_circuit'] = s_circuit

        if v_in_req:
            # Use default values for this case.
            None

        if self.record_values:
            super().record_fun()

        self.time = self.time + self.dt
