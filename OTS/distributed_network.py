import numpy as np
import pdb


class input_node:
    def __init__(self, v_in_traj):
        self.v_in_traj = v_in_traj
        self.state = {}

    def get_input(self, sim_step, N_horizon):
        self.state['v_out_traj'] = self.v_in_traj[sim_step:sim_step+N_horizon]

    # def self.state['keyword']


class output_node:
    def __init__(self, bandwidth_traj, memory_traj):
        self.bandwidth_traj = bandwidth_traj
        self.memory_traj = memory_traj
        self.state = {}

    def get_output(self, sim_step, N_horizon):
        self.state['bandwidth_traj'] = self.bandwidth_traj[sim_step:sim_step+N_horizon]
        self.state['memory_traj'] = self.memory_traj[sim_step:sim_step+N_horizon]

    # def state(self, keyword):
    #     return self.state['keyword']


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

    def simulate(self, c_traj_list):
        """
        One item in c_traj_list for each connection.
        Following the same order as connections.
        """
        for input_i in self.inputs:
            input_i.get_input(self.sim_step, self.N_horizon)

        for output_i in self.outputs:
            output_i.get_output(self.sim_step, self.N_horizon)

        for connection_i, c_traj in zip(self.connections, c_traj_list):
            s0 = connection_i[1].state['s_traj'][0]
            v_in_traj = connection_i[0].state['v_out_traj']
            bandwidth_traj = connection_i[2].state['bandwidth_traj']
            memory_traj = connection_i[2].state['memory_traj']
            scheduler_result = connection_i[1].solve(s0, v_in_traj, c_traj, bandwidth_traj, memory_traj)

        self.sim_step += 1
