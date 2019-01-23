import numpy as np
import pandas as pd
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
        self.con_pd = pd.DataFrame(connections)
        self.sim_step = 0
        self.N_horizon = N_horizon

    def simulate(self, c_list):
        # TODO: Option to have multiple inputs.
        """
        One item in c_traj_list for each connection.
        Each connection with dict containing 'source', 'node' and  'connection'
        """
        # Check consistency of c matrix (for all elements of sequence)
        for i, c_i in enumerate(c_list):
            if not all(np.concatenate([np.sum(c_i_k, axis=0) == 1 for c_i_k in c_i])):
                raise Exception('Composition Matrix has a row sum that is not equal to 1. Data is lost or created in connection: {}'.format(i))

        for input_i in self.inputs:
            input_i.get_input(self.sim_step, self.N_horizon)

        for output_i in self.outputs:
            output_i.get_output(self.sim_step, self.N_horizon)

        for connection_i, c in zip(self.connections, c_list):

            # Get information from connected nodes:

            # Current node:
            # -----------------------------
            s0 = connection_i['node'].predict['s'][0]

            # Source node(s):
            # -----------------------------
            # For multiple output sources, select respective channel:

            v_out_source = []  # All outputs that lead to the current node.
            for source_i in connection_i['source']:
                if type(source_i) is input_node:
                    v_out_source.append(source_i.predict['v_out'])
                else:
                    # Use Pandas to find source_node
                    source_node_ind = self.con_pd.index[self.con_pd['node'] == source_i].tolist()[0]
                    # Find index of source v_out that is connected to the current node:
                    source_channel_ind = [connection_i['node'] is source_ots for source_ots in self.con_pd['target'][source_node_ind]]
                    # Append the respective channel to v_out_source
                    v_out_source.append([source_i_k[[source_channel_ind]] for source_i_k in source_i.predict['v_out']])

            # if type(connection_i['source']) is list:
            #     assert len(connection_i['source']) == connection_i['node'].n_in
            #     # v_in = np.split(np.hstack([source_i.predict['v_out'] for source_i in connection_i['source']]).reshape(self.N_horizon, -1), self.N_horizon)
            #     v_in = np.hstack([source_i.predict['v_out'] for source_i in connection_i['source']])
            # else:
            #     v_in = connection_i['source'].predict['v_out']
            if len(v_out_source) > 1:
                v_in = np.hstack(v_out_source)
            else:
                v_in = v_out_source[0]

            # Target node(s):
            if type(connection_i['target']) is list:
                assert len(connection_i['target']) == connection_i['node'].n_out, 'Connection has the wrong number of outputs'.
                bandwidth_load = np.hstack([source_i.predict['bandwidth_load'] for source_i in connection_i['target']])
                memory_load = np.hstack([source_i.predict['memory_load'] for source_i in connection_i['target']])
            else:
                bandwidth_load = connection_i['target'].predict['bandwidth_load']
                memory_load = connection_i['target'].predict['memory_load']

            # Simulate Node
            scheduler_result = connection_i['node'].solve(s0, v_in, c, bandwidth_load, memory_load)

        self.sim_step += 1
