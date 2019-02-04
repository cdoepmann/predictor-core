import numpy as np
import pandas as pd
from optimal_traffic_scheduler_02 import optimal_traffic_scheduler
import pdb


class client_node:
    def __init__(self, N_horizon, name='client_node', source_fun=None, target_fun=None):
        """
        Mimics the behavior of a server node. Can be both source or target of a connection.
        Inputs:
        - N_horizon : Prediction horizon.
        - source_fun: [Optional, when used as target node] function that returns a sequence of package_streams, when evaluated with the current time_step.
        - target_fun: [Optional, when used as source node] function that returns a sequence of bandwidth_load and memory_load, when evaluated with the current time_step.
        """
        self.predict = {}
        self.obj_name = name
        self.source_fun = source_fun
        self.target_fun = target_fun

    def get_input(self, time_step):
        if not self.source_fun:
            raise Exception('source_fun was not defined. Cant get input for client node')
        else:
            self.predict['v_out'] = self.source_fun(time_step)

    def get_output(self, time_step):
        if not self.target_fun:
            raise Exception('source_fun was not defined. Cant get input for client node')
        else:
            self.predict['bandwidth_load'], self.predict['memory_load'] = self.target_fun(time_step)


class distributed_network:
    def __init__(self, circuits):
        self.connections, self.nodes = self.circ_2_network(circuits)
        self.analyze_connections()

    def simulate(self):
        # In every simulation:
        # a) Determine associated properties for every connection that are determined by the source and target:
        for i, connection_i in self.connections.iterrows():
            # Package stream is depending on the source. Create [N_timesteps x n_outputs x 1] array (with np.stack())
            # and access the element that is stored in 'output_ind' for each connection.
            connection_i['v'] = np.stack(connection_i['source'].predict['v_out'])[:, [connection_i['output_ind']], :]
            # Bandwidth and memory load are depending on the target. Create [N_timesteps x 1 x1] array.
            connection_i['bandwidth_load'] = np.stack(connection_i['target'].predict['bandwidth_load'])
            connection_i['memory_load'] = np.stack(connection_i['target'].predict['bandwidth_load'])

        # b) Iterate over all nodes, query respective I/O data from connections and simulate node
        for k, node_k in self.nodes.iterrows():
            # Simulate only if the node is an optimal_traffic_scheduler.
            if type(node_k.node) is optimal_traffic_scheduler:
                # Concatenate package streams for all inputs:
                v_in = np.concatenate(self.connections.loc[node_k['con_in'], 'v'].values, axis=1)
                # Concatenate bandwidth and memory load for all outputs:
                bandwidth_load = np.concatenate(self.connections.loc[node_k['con_out'], 'bandwidth_load'].values, axis=1)
                memory_load = np.concatenate(self.connections.loc[node_k['con_out'], 'memory_load'].values, axis=1)

                io_mapping = node_k.io_mapping
                output_partition = node_k.output_partition

                # np.matmul: If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
                v_s = output_partition@v_in[:, io_mapping]
                # Simulate Node
                # scheduler_result = connection_i['node'].solve(s0, v_in, c, bandwidth_load, memory_load)

    def circ_2_network(self, circuits):
        """
        Create nodes and connections DataFrame from circuit dict.
        Each connection has a new attribute 'circuit' that lists all the circuits that are active.
        """
        connections = []
        nodes = {'node': []}
        for i, circuit_i in enumerate(circuits):
            connections.extend([{'source': circuit_i['route'][k], 'target': circuit_i['route'][k+1], 'circuit':i}
                                for k in range(len(circuit_i['route'])-1)])
            nodes['node'].extend(circuit_i['route'])
        con_pd = pd.DataFrame(connections)
        nodes_pd = pd.DataFrame(nodes).drop_duplicates().reset_index().drop('index', axis=1)
        # Add name to each node:
        nodes_pd['name'] = nodes_pd['node'].apply(lambda obj: obj.obj_name)
        con_pd['source_name'] = con_pd['source'].apply(lambda obj: obj.obj_name)
        con_pd['target_name'] = con_pd['target'].apply(lambda obj: obj.obj_name)
        return con_pd, nodes_pd

    def analyze_connections(self):
        """
        Analyzes the connections and nodes and adds further information
        """
        self.connections['output_ind'] = None
        self.connections['v'] = None
        self.connections['circ_comp'] = None
        self.connections['bandwidth_load'] = None
        self.connections['memory_load'] = None

        self.nodes['con_in'] = None
        self.nodes['n_in'] = None
        self.nodes['con_out'] = None
        self.nodes['n_out'] = None
        self.nodes['io_mapping'] = None
        self.nodes['output_partition'] = None

        # Run only once to get information:
        for k, node_k in self.nodes.iterrows():
            # Boolean array that indicates in which connections node_k is the source.
            # This determines how many outputs the current node.
            node_k['con_out'] = (self.connections['source'] == node_k['node']).values
            node_k['n_out'] = sum(node_k['con_out'])
            # The output of each node is a vector with n_out elements. 'output_ind' marks which
            # of its elements refers to which connection:
            self.connections.loc[node_k['con_out'], 'output_ind'] = np.arange(node_k['n_out'], dtype='int16').tolist()

            # Boolean array that indicates in which connections node_k is the target. This determines the
            # number of inputs.
            node_k['con_in'] = (self.connections['target'] == node_k['node']).values
            node_k['n_in'] = sum(node_k['con_in'])

            if type(node_k['node']) is optimal_traffic_scheduler:
                # Query all incoming and outgoing connections and find the respective circuits:
                source_k = self.connections[node_k['con_out']]
                target_k = self.connections[node_k['con_in']]
                input_circ = source_k.circuit.values
                output_circ = target_k.circuit.values
                # io_mapping is an index vector that has the same length as the number of circuits in this node.
                # When indexing the incoming package stream vector with io_mapping the same elements are returned in the order of
                # the outgoing circuits:
                node_k['io_mapping'] = np.argsort(input_circ)[np.argsort(output_circ)]

                # Each circuit triggers on connection and therefore incoming package stream for the current node. However, the output buffers
                # may contain elements of several circuits.
                output_partition = []
                for i, source_k_i in source_k.drop_duplicates('target').iterrows():
                    # Which
                    output_partition.append((source_k.target == source_k_i.target).values)

                node_k['output_partition'] = np.stack(output_partition)
                # With io_mapping and output_partion:
                # v_s = output_partition @ v_in[io_mapping]

                node_k['node'].setup(n_in=node_k['n_in'], n_out=node_k['n_out'])


# Same configuration for all nodes:
setup_dict = {}
setup_dict['v_max'] = 20  # mb/s
setup_dict['s_max'] = 30  # mb
setup_dict['dt'] = 1  # s
setup_dict['N_steps'] = 20
setup_dict['v_delta_penalty'] = 1

ots_1 = optimal_traffic_scheduler(setup_dict, name='ots_1')
ots_2 = optimal_traffic_scheduler(setup_dict, name='ots_2')
ots_3 = optimal_traffic_scheduler(setup_dict, name='ots_3')
ots_4 = optimal_traffic_scheduler(setup_dict, name='ots_4')
# Input -> ots_1 -> out_2 -> out_3 -> Output
input_node_1 = client_node(setup_dict['N_steps'], name='input_node_1')
input_node_2 = client_node(setup_dict['N_steps'], name='input_node_2')
input_node_3 = client_node(setup_dict['N_steps'], name='input_node_3')
output_node_1 = client_node(setup_dict['N_steps'], name='output_node_1')
output_node_2 = client_node(setup_dict['N_steps'], name='output_node_2')
output_node_3 = client_node(setup_dict['N_steps'], name='output_node_3')

circuits = [
    {'route': [input_node_1, ots_1, ots_2, ots_3, ots_4, output_node_1]},
    {'route': [input_node_2, ots_1, ots_3, ots_4, output_node_2]},
    {'route': [input_node_3, ots_1, ots_2, ots_3, ots_4, output_node_3]},
]


dn = distributed_network(circuits)

dn.nodes.loc[1].output_partition
dn.connections


node_k = dn.nodes.loc[1]
source_k = dn.connections[node_k.node == dn.connections['source']]
target_k = dn.connections[node_k.node == dn.connections['target']]
output_circ = source_k.circuit.values  # output = for which connection is the node 'source'
input_circ = target_k.circuit.values  # input = for which connection is the node 'target'
# index vector to sort all incoming streams to the appropriate output buffers:
io_mapping = np.argsort(input_circ)[np.argsort(output_circ)]

# For each unique outgoing connection (find all outgoing connections and drop duplicates):
output_partition = []
for i, source_k_i in source_k.drop_duplicates('target').iterrows():
    # Which
    output_partition.append((source_k.target == source_k_i.target).values)

output_partition


node_k.output_partition
