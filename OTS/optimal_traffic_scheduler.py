import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import struct_symSX, struct_SX, entry
import pdb
from scipy.linalg import block_diag


class optimal_traffic_scheduler:
    def __init__(self, setup_dict, name='ots', record_values=True, silent=False):
        """
        Expected in setup_dict:
        ----------------------------------
        v_in_max_total         - Upper limit for the sum of all incoming packets from all incoming connections (in packets/s)
        v_out_max_total        - Upper limit for the sum of all outgoing packets from all outgoing connections (in packets/s)
        dt                     - Timestep (in seconds) of the optimizer.
        N_steps                - Horizon of the MPC optimization
        weights                - Dict with the keys (and suggested values): {'control_delta': 0.1, 'send': 1, 'store': 0, 'receive': 1}
            send                   - Focus on sending packets.
            store                  - Focus on keeping the buffer as low as possible. Not recommended as the desired effect is achieved with 'send'.
            receive                - Focus on receiving packets: Avoid discarding packets and allow for additional packets to be sent.
            control_delta          - Predictions of the current node are communicated and incorporated in connected nodes. control_delta penalizes
                                     rapid changes of the node behavior. This is important for stability of the distributed system.
        ----------------------------------
        """
        # Legacy check:
        assert 'v_max' not in setup_dict.keys(), 'Updated Framework where v_max paramter is not longer supported for ots init. Use v_in_max_total and v_out_max_total instead.'
        assert 's_max' not in setup_dict.keys(), 'Updated Framework where s_max parameter is not longer supported for ots init.'

        self.obj_name = name
        self.v_in_max_total = setup_dict['v_in_max_total']
        self.v_out_max_total = setup_dict['v_out_max_total']
        self.dt = setup_dict['dt']
        self.N_steps = setup_dict['N_steps']
        self.weights = setup_dict['weights']
        self.record_values = record_values
        self.time = np.array([[0]])  # 1,1 array for consistency.

        self.silent = silent

    def setup(self, n_in=None, n_out=None, input_circuits=None, output_circuits=None):
        """
        n_in:             Number of Inputs
        n_out:            Number of outputs
        input_circuits:   List with each item being a list of identifiers for the input circuits
        output_circuits:  List with each item being a list of identifiers for the output circuits
        """

        self.n_in = n_in
        self.n_out = n_out

        self.n_circuit_in = [len(c_i) for c_i in input_circuits]
        self.n_circuit_out = [len(c_i) for c_i in output_circuits]

        # Note: Pb is defined "transposed", as casadi will raise an error for n_out=1, since it cant handle row vectors.
        self.Pb = self.Pb_fun(input_circuits, output_circuits)
        self.Pc = self.Pc_fun(input_circuits, output_circuits)

        assert len(self.n_circuit_in) == self.n_in
        assert len(self.n_circuit_out) == self.n_out
        assert np.sum(self.n_circuit_in) == np.sum(self.n_circuit_out)

        self.problem_formulation()
        self.create_optim()

    def initialize_record(self):
        assert 'predict' in self.__dict__.keys(), 'Cant initialize record before solve() was called and initial values for predict created.'
        self.record = {}
        self.record['time'] = []
        # Set the first element of the predicted values as the first element of the recorded values.
        for predict_key in self.predict.keys():
            self.record[predict_key] = []
            self.record[predict_key].append(self.predict[predict_key][0])

    def problem_formulation(self):
        """ MPC states for stage k"""

        self.mpc_xk = struct_symSX([
            entry('s_buffer', shape=(self.n_out, 1)),
            entry('s_circuit', shape=(int(np.sum(self.n_circuit_in)), 1)),
        ])
        # States at next time-step. Same structure as mpc_xk. Will be assigned expressions later on.
        self.mpc_xk_next = struct_SX(self.mpc_xk)

        """ MPC control inputs for stage k"""
        self.mpc_uk = struct_symSX([
            entry('v_in_discard', shape=(self.n_in, 1)),
            entry('v_in_extra', shape=(self.n_in, 1)),
            entry('v_out', shape=(self.n_out, 1)),
        ])
        """ MPC time-varying parameters for stage k"""
        # Dummy struct:
        cv_in_struct = struct_symSX([
            entry('c_'+str(i), shape=(self.n_circuit_in[i], 1)) for i in range(self.n_in)
        ])

        self.mpc_tvpk = struct_symSX([
            entry('u_prev', struct=self.mpc_uk),
            entry('v_in_req', shape=(self.n_in, 1)),
            entry('cv_in', struct=cv_in_struct),
            entry('v_out_max', shape=(self.n_out, 1)),
            entry('s_buffer_source', shape=(self.n_in, 1)),
        ])
        """ MPC parameters for stage k"""
        self.mpc_pk = struct_symSX([
            # Note: Pb is defined "transposed", as casadi will raise an error for n_out=1, since it cant handle row vectors.
            entry('Pb', shape=(np.sum(self.n_circuit_in), self.n_out)),
            entry('Pc', shape=(np.sum(self.n_circuit_in), np.sum(self.n_circuit_in))),
        ])

        """ Memory """
        # Buffer memory
        s_buffer = self.mpc_xk['s_buffer']
        # Circuit memory:
        s_circuit = self.mpc_xk['s_circuit']

        """ Incoming packet stream """

        # Incoming package stream for each buffer:
        v_in_req = self.mpc_tvpk['v_in_req']
        # Discarded packet stream:
        v_in_discard = self.mpc_uk['v_in_discard']
        # Additional packet contigent that could be accepted.
        v_in_extra = self.mpc_uk['v_in_extra']
        # Resulting incoming packet stream:
        v_in = v_in_req - v_in_discard
        # Allowed incoming packet stream:
        v_in_max = v_in + v_in_extra
        # Composition of incoming stream cv_in (note that .prefix allows to use struct_symSX power indices).
        cv_in = self.mpc_tvpk.prefix['cv_in']
        # Incoming packet stream (on circuit level) (multiplying each incoming stream (scalar) with the respective composition vector)
        vc_in = vertcat(*[v_in[i]*cv_in['c_'+str(i)] for i in range(self.n_in)])

        """ Outgoing packet stream """
        # Outgoing packet stream:
        v_out = self.mpc_uk['v_out']
        # Maximum value for v_out (determined by target server):
        v_out_max = self.mpc_tvpk['v_out_max']
        # Outgoing packet stream (as list)
        v_out_list = vertsplit(v_out)
        # v_out from the previous solution:

        """ Load information """
        # memory info incoming servers
        s_buffer_source = self.mpc_tvpk['s_buffer_source']

        """ Circuit matching """
        # Assignment Matrix: Which element of each input is assigned to which output buffer:
        # Note: Pb is defined "transposed", as casadi will raise an error for n_out=1, since it cant handle row vectors.
        Pb = self.mpc_pk['Pb'].T
        # Assignment Matrix: Which input circuit is directed to which output circuit:
        Pc = self.mpc_pk['Pc']

        """ System dynamics"""
        # system dynamics, constraints and objective definition:
        s_tilde_next = s_buffer + self.dt*Pb@vc_in
        sc_tilde_next = s_circuit + self.dt*Pc@vc_in

        # Protected division. The denominator can only be zero, if the numerator is also zero. cv_out_i = 0 in that case
        # and would be NaN without using the eps value.
        eps = 1e-6
        # cv_out = s_circuit/(mtimes(Pb.T, s_buffer)+eps)
        # cv_out = (sc_tilde_next)/(self.Pc@self.Pb.T@s_tilde_next+eps)

        cv_out = [sc_i/(s_tilde_next[i]+eps) for i, sc_i in enumerate(vertsplit(sc_tilde_next, np.cumsum([0]+self.n_circuit_out)))]
        vc_out = vertcat(*[v_out_i*cv_out_i for cv_out_i, v_out_i in zip(cv_out, v_out_list)])

        # vc_out = (self.Pc@Pb.T@v_out)*cv_out

        s_buffer_next = s_tilde_next - self.dt*v_out
        s_circuit_next = sc_tilde_next - self.dt*vc_out

        self.mpc_xk_next['s_buffer'] = s_buffer_next
        self.mpc_xk_next['s_circuit'] = s_circuit_next

        """ Objective """

        # Objective function with fairness formulation:
        # TODO: +1 umwandeln in tuning parameter?
        stage_cost = sum1(-1/(10*s_buffer_source+1)*v_in_max)
        stage_cost += sum1(-1/(10*s_buffer+1)*v_out)

        # Control delta regularization
        stage_cost += self.weights['control_delta']*sum1((self.mpc_uk-self.mpc_tvpk['u_prev'])**2)

        # Terminal cost:
        terminal_cost = sum1(-1/(s_buffer+1))

        """ Constraints"""
        # All states with lower bound 0 and upper bound infinity
        self.mpc_xk_lb = self.mpc_xk(0)
        self.mpc_xk_ub = self.mpc_xk(np.inf)

        # All inputs with lower bound 0 and upper bound infinity
        self.mpc_uk_lb = self.mpc_uk(0)
        self.mpc_uk_ub = self.mpc_uk(np.inf)

        # Further (non-linear constraints on states and inputs)
        cons_list = [
            {'lb': [-np.inf]*self.n_in,  'eq': -v_in,                              'ub': [0]*self.n_in},   # v_in cant be negative (Note: v_in is not an input)
            {'lb': [-np.inf],            'eq': sum1(v_in_max)-self.v_in_max_total, 'ub': [0]},             # sum of all incoming traffic can't exceed v_in_max_total
            {'lb': [-eps]*self.n_in,     'eq': v_in_discard*v_in_extra,            'ub': [eps]*self.n_in},  # packets can be discarded or added (not both). Should be zero but is better to be within a certain tolerance.
            {'lb': [-np.inf]*self.n_out, 'eq': v_out-v_out_max,                    'ub': [0]*self.n_out},  # outgoing packet stream cant be greater than what is allowed individually
            {'lb': [-np.inf],            'eq': sum1(v_out)-self.v_out_max_total,   'ub': [0]},             # outgoing packet stream cant be greater than what is allowed in total.
            {'lb': [-np.inf]*self.n_in,  'eq': v_in_max*self.dt-s_buffer_source,   'ub': [0]*self.n_in},   # Can't receive more than what is available in source_buffer.
        ]

        # Constraints for fairness (all vs. all comparison with individuall limits for each v_out):
        # For insights on why this works see the IPYNB in ./fairness/fairness_scenario.ipynb
        # Input
        for i in range(self.n_in):
            for j in range(self.n_in):
                cons_list.append(
                    {'lb': [-np.inf], 'eq': (s_buffer_source[i]-s_buffer_source[j])*(v_in_max[j]-v_in_max[i]), 'ub': [0]},
                )
        # Output
        for i in range(self.n_out):
            for j in range(self.n_out):
                cons_list.append(
                    {'lb': [-np.inf], 'eq': (v_out_max[i]-v_out[i])*(v_out_max[j]-v_out[j])*(s_buffer[i]-s_buffer[j])*(v_out[j]-v_out[i]), 'ub': [0]},
                )

        cons = vertcat(*[con_i['eq'] for con_i in cons_list])
        cons_lb = np.concatenate([con_i['lb'] for con_i in cons_list])
        cons_ub = np.concatenate([con_i['ub'] for con_i in cons_list])

        """ Summarize auxiliary / intermediate variables in mpc_aux with their respective expression """
        bandwidth_load_in = sum1(v_in)/self.v_in_max_total
        bandwidth_load_out = sum1(v_out)/self.v_out_max_total

        # For debugging: Add intermediate variables to mpc_aux_expr and query them after solving the optimization problem.
        self.mpc_aux_expr = struct_SX([
            entry('v_in', expr=v_in),
            # entry('vc_in', expr=vc_in),
            # entry('s_tilde_next', expr=s_tilde_next),
            # entry('sc_tilde_next', expr=sc_tilde_next),
            entry('v_in_max', expr=v_in_max),
            entry('cv_out', expr=vertcat(*cv_out)),
            entry('bandwidth_load_in', expr=bandwidth_load_in),
            entry('bandwidth_load_out', expr=bandwidth_load_out),
        ])

        """ Problem dictionary """
        mpc_problem = {}
        mpc_problem['cons'] = Function('cons', [self.mpc_xk, self.mpc_uk, self.mpc_tvpk, self.mpc_pk], [cons])
        mpc_problem['cons_lb'] = cons_lb
        mpc_problem['cons_ub'] = cons_ub
        mpc_problem['stage_cost'] = Function('stage_cost', [self.mpc_xk, self.mpc_uk, self.mpc_tvpk, self.mpc_pk], [stage_cost])
        mpc_problem['terminal_cost'] = Function('terminal_cost', [self.mpc_xk], [terminal_cost])
        mpc_problem['model'] = Function('model', [self.mpc_xk, self.mpc_uk, self.mpc_tvpk, self.mpc_pk], [self.mpc_xk_next])
        mpc_problem['aux'] = Function('aux', [self.mpc_xk, self.mpc_uk, self.mpc_tvpk, self.mpc_pk], [self.mpc_aux_expr])

        self.mpc_problem = mpc_problem

    def create_optim(self):
        # Initialize trajectory lists (each list item, one time-step):
        self.mpc_obj_x = mpc_obj_x = struct_symSX([
            entry('x', repeat=self.N_steps+1, struct=self.mpc_xk),
            entry('u', repeat=self.N_steps, struct=self.mpc_uk),
        ])

        # Note that:
        # x = [x_0, x_1, ... , x_N+1]   (N+1 elements)
        # u = [u_0, u_1, ... , u_N]     (N elements)
        # For the optimization variable x_0 we introduce the simple equality constraint that it has
        # to be equal to the parameter x0 (mpc_obj_p)
        self.mpc_obj_p = mpc_obj_p = struct_symSX([
            entry('tvp', repeat=self.N_steps, struct=self.mpc_tvpk),
            entry('x0',  struct=self.mpc_xk),
            entry('p',   struct=self.mpc_pk),
        ])

        # Dummy struct with symbolic variables
        aux_struct = struct_symSX([
            entry('aux', repeat=self.N_steps, struct=self.mpc_aux_expr)
        ])
        # Create mutable symbolic expression from the struct defined above.
        self.mpc_obj_aux = mpc_obj_aux = struct_SX(aux_struct)

        # Initialize objective value:
        obj = 0
        # Initialize constraings:
        cons = []
        cons_ub = []
        cons_lb = []

        # Equality constraint for first state:
        cons.append(mpc_obj_x['x', 0]-mpc_obj_p['x0'])
        cons_lb.append(np.zeros(self.mpc_xk.shape))
        cons_ub.append(np.zeros(self.mpc_xk.shape))

        # Recursively evaluate system equation and add stage cost and stage constraints:
        for k in range(self.N_steps):
            mpc_xk_next = self.mpc_problem['model'](mpc_obj_x['x', k], mpc_obj_x['u', k], mpc_obj_p['tvp', k], mpc_obj_p['p'])

            # State constraint:
            cons.append(mpc_xk_next-mpc_obj_x['x', k+1])
            cons_lb.append(np.zeros(self.mpc_xk.shape))
            cons_ub.append(np.zeros(self.mpc_xk.shape))

            # Add the "stage cost" to the objective
            obj += self.mpc_problem['stage_cost'](mpc_obj_x['x', k], mpc_obj_x['u', k], mpc_obj_p['tvp', k], mpc_obj_p['p'])

            # Constraints for the current step
            cons.append(self.mpc_problem['cons'](mpc_obj_x['x', k], mpc_obj_x['u', k], mpc_obj_p['tvp', k], mpc_obj_p['p']))
            cons_lb.append(self.mpc_problem['cons_lb'])
            cons_ub.append(self.mpc_problem['cons_ub'])

            # Calculate auxiliary values:
            mpc_obj_aux['aux', k] = self.mpc_problem['aux'](mpc_obj_x['x', k], mpc_obj_x['u', k], mpc_obj_p['tvp', k], mpc_obj_p['p'])

        # Terminal cost:
        obj += self.mpc_problem['terminal_cost'](mpc_obj_x['x', -1])

        # Upper and lower bounds on objective x:
        self.mpc_obj_x_lb = self.mpc_obj_x(0)
        self.mpc_obj_x_ub = self.mpc_obj_x(0)

        self.mpc_obj_x_lb['x', :] = self.mpc_xk_lb
        self.mpc_obj_x_ub['x', :] = self.mpc_xk_ub

        self.mpc_obj_x_lb['u', :] = self.mpc_uk_lb
        self.mpc_obj_x_ub['u', :] = self.mpc_uk_ub

        optim_dict = {'x': mpc_obj_x,       # Optimization variable
                      'f': obj,             # objective
                      'g': vertcat(*cons),  # constraints
                      'p': mpc_obj_p}       # parameters
        self.cons_lb = vertcat(*cons_lb)
        self.cons_ub = vertcat(*cons_ub)

        # Use the structured data obj_x and obj_p and create identically organized copies with numerical values (initialized to zero)
        self.mpc_obj_x_num = self.mpc_obj_x(0)
        self.mpc_obj_p_num = self.mpc_obj_p(0)
        self.mpc_obj_aux_num = self.mpc_obj_aux(0)

        # TODO: Make optimization option available to user.
        # Create casadi optimization object:
        opts = {'ipopt.linear_solver': 'ma27', 'error_on_fail': True}
        self.optim = nlpsol('optim', 'ipopt', optim_dict, opts)
        if self.silent:
            opts['ipopt.print_level'] = 0
            opts['ipopt.sb'] = "yes"
            opts['print_time'] = 0

        # Create function to calculate buffer memory from parameter and optimization variable trajectories
        self.aux_fun = Function('aux_fun', [mpc_obj_x, mpc_obj_p], [mpc_obj_aux])

    def solve(self, s_buffer_0, s_circuit_0, v_in_req, cv_in, v_out_max, s_buffer_source, *args, debugging=True, **kwargs):
        """
        Solves the optimal control problem defined in optimal_traffic_scheduler.problem_formulation().
        Inputs:
        - s_buffer_0            : initial memory for each buffer (must be n_out x 1 vector)
        - s_circuit_0           : intitial memory for each circuit (must be np.sum(n_circuit_out) x 1 vector)

        Predicted trajectories as lists with N_horizon elments, where each(!!!) list item has the following configuration:
        - v_in_req              : Requested incoming package stream for each buffer (n_in x 1 vector)
        - cv_in                 : Composition of incoming streams. (List with n_in elements with n_circuit_in[i] x 1 vector for list item i)
        - v_out_max             : Maximum for outgoing packet stream. Supplied by target servers (n_out x 1 vector)
        - s_buffer_source       : Memory load of source server(s) (n_in x 1 vector)

        Populates the "predict" and optionally the "record" dictonaries of the class.
        - Predict: A dict with the optimized state and control trajectories of the node
        - Record:  Lists with items appended for each call of solve, recording the states of the node.

        "Solve" also advances the time of the node by one time_step.
        """

        """ Check if inputs are valid """
        if debugging:
            assert np.isclose(np.sum(s_buffer_0), np.sum(s_circuit_0)), 'Inconsistent initial conditions: sum of s_buffer_0 is not equal sum of s_circuit_0'
            assert np.allclose(self.Pb@self.Pc.T@s_circuit_0, s_buffer_0), 'Inconsistent initial conditions: s_circuit_0 and s_buffer_0 are not matching for the given setup.'
            assert np.allclose(np.array([np.sum(np.concatenate(cv_in_i)) for cv_in_i in cv_in]), self.n_in), 'Inconsistent value for cv_in. There is at least one connection, where the sum of the composition is not close to 1.'
            assert np.allclose([v_in_req_i.shape for v_in_req_i in v_in_req], (self.n_in, 1)), 'v_in must be a list of arrays where each element has shape (n_in,1)'
            assert np.allclose([v_out_max_i.shape for v_out_max_i in v_out_max], (self.n_out, 1)), 'v_out_max must be a list of arrays where each element has shape (n_out,1)'
            assert np.allclose([s_buffer_source_i.shape for s_buffer_source_i in s_buffer_source], (self.n_out, 1)), 's_buffer_source must be a list of arrays where each element has shape (n_in,1)'

        """ Set initial condition """
        self.mpc_obj_p_num['x0', 's_buffer'] = s_buffer_0
        self.mpc_obj_p_num['x0', 's_circuit'] = s_circuit_0

        """ Get previous inputs and assign to tvp u_prev after shifting"""
        self.mpc_obj_p_num['tvp', :-1, 'u_prev'] = self.mpc_obj_x_num['u', 1:]
        self.mpc_obj_p_num['tvp', -1, 'u_prev'] = self.mpc_obj_x_num['u', -1]

        """ Assign further parameters to tvp"""
        self.mpc_obj_p_num['tvp', :, 'v_in_req'] = v_in_req
        self.mpc_obj_p_num['tvp', :, 'cv_in'] = [vertcat(*cv_in_i) for cv_in_i in cv_in]
        self.mpc_obj_p_num['tvp', :, 'v_out_max'] = v_out_max
        self.mpc_obj_p_num['tvp', :, 's_buffer_source'] = s_buffer_source

        """ Assign parameters """
        # Note: Pb is defined "transposed", as casadi will raise an error for n_out=1, since it cant handle row vectors.
        self.mpc_obj_p_num['p', 'Pb'] = self.Pb.T
        self.mpc_obj_p_num['p', 'Pc'] = self.Pc

        """Solve optimization problem for given conditions:"""
        optim_results = self.optim(
            ubx=self.mpc_obj_x_ub,
            lbx=self.mpc_obj_x_lb,
            ubg=self.cons_ub,
            lbg=self.cons_lb,
            p=self.mpc_obj_p_num,
            x0=self.mpc_obj_x_num
        )
        optim_stats = self.optim.stats()

        """ Assign solution to mpc_obj_x_num to allow easy accessibility: """
        self.mpc_obj_x_num = self.mpc_obj_x(optim_results['x'])

        """ Calculate aux values: """
        self.mpc_obj_aux_num = self.mpc_obj_aux(self.aux_fun(self.mpc_obj_x_num, self.mpc_obj_p_num))

        """ Retrieve relevant trajectories """

        v_out = self.mpc_obj_x_num['u', :, 'v_out']
        cv_out = [np.split(self.mpc_obj_aux_num['aux', k, 'cv_out'], np.cumsum(self.n_circuit_out[:-1])) for k in range(self.N_steps)]

        v_in = self.mpc_obj_aux_num['aux', :, 'v_in']
        v_in_max = self.mpc_obj_aux_num['aux', :, 'v_in_max']

        s_buffer = self.mpc_obj_x_num['x', :, 's_buffer']
        s_circuit = self.mpc_obj_x_num['x', :, 's_circuit']

        bandwidth_load_in = self.mpc_obj_aux_num['aux', :, 'bandwidth_load_in']
        bandwidth_load_out = self.mpc_obj_aux_num['aux', :, 'bandwidth_load_out']

        """ Advance time and record values """
        self.time = self.time + self.dt

        self.predict = {}
        self.predict['v_in'] = v_in
        self.predict['v_in_max'] = v_in_max
        self.predict['v_in_req'] = v_in_req
        self.predict['cv_in'] = cv_in
        self.predict['v_out'] = v_out
        self.predict['v_out_max'] = v_out_max
        self.predict['cv_out'] = cv_out
        self.predict['s_buffer'] = s_buffer          # carefull: N_steps+1 elements
        self.predict['s_circuit'] = s_circuit        # carefull: N_steps+1 elements
        self.predict['bandwidth_load_in'] = bandwidth_load_in
        self.predict['bandwidth_load_out'] = bandwidth_load_out

        if self.record_values:
            if not 'record' in self.__dict__.keys():
                self.initialize_record()
            self.record_fun()

        if not optim_stats['success']:
            raise Exception(optim_stats['success'])
        return optim_stats['success']

    def record_fun(self):
        self.record['time'].append(np.copy(self.time))
        for key, val in self.predict.items():
            self.record[key].append(val[0])

    def latency_adaption(self, sequence, type, input_delay=np.array([0]), output_delay=np.array([0])):
        """
        Adapting the incoming predictions due to latency.
        """
        # TODO: Check if this is working as desired.
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
        # Same init as optimal_traffic_scheduler
        super().__init__(setup_dict, name, record_values)

    def setup(self, n_in=None, n_out=None, input_circuits=None, output_circuits=None, output_delay=0):
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
        # TODO: Check if this is working as desired.
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
