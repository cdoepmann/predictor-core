import numpy as np
import pdb


class input_node:
    def __init__(self, v_in_traj):
        self.v_in_traj = v_in_traj
        self.predict = {}

    def get_input(self, sim_step, N_horizon):
        self.predict['v_out'] = self.v_in_traj[sim_step:sim_step+N_horizon]


class output_node:
    def __init__(self, bandwidth_load, memory_load):
        self.bandwidth_load = bandwidth_load
        self.memory_load = memory_load
        self.predict = {}

    def get_output(self, sim_step, N_horizon):
        self.predict['bandwidth_load'] = self.bandwidth_load[sim_step:sim_step+N_horizon]
        self.predict['memory_load'] = self.memory_load[sim_step:sim_step+N_horizon]


class distributed_network:
    def __init__(self, inputs, outputs, connections, N_horizon):
        """
        Define a distributed network:
        """
        self.inputs = inputs
        self.outputs = outputs
        self.connections = connections
        self.sim_step = 0
        self.N_horizon = N_horizon

    def simulate(self, c_list):
        # TODO: Option to have multiple inputs.
        """
        One item in c_traj_list for each connection.
        Each connection with dict containing 'source', 'node' and  'connection'
        """
        for input_i in self.inputs:
            input_i.get_input(self.sim_step, self.N_horizon)

        for output_i in self.outputs:
            output_i.get_output(self.sim_step, self.N_horizon)

        for connection_i, c in zip(self.connections, c_list):
            if type(connection_i['source']) is list:
                raise ValueError('Multiple Inputs are not supported yet.')
                #assert len(connection_i['source']) == connection_i['node'].n_in
            if type(connection_i['target']) is list:
                raise ValueError('Multiple Inputs are not supported yet.')

            s0 = connection_i['node'].predict['s'][0]
            v_in = connection_i['source'].predict['v_out']
            bandwidth_load = connection_i['target'].predict['bandwidth_load']
            memory_load = connection_i['target'].predict['memory_load']
            scheduler_result = connection_i['node'].solve(s0, v_in, c, bandwidth_load, memory_load)

        self.sim_step += 1
