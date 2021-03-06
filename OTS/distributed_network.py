import numpy as np
import pandas as pd
from optimal_traffic_scheduler import optimal_traffic_scheduler
import pdb


class client_node:
    def __init__(self, dt, name='client_node', source_fun=None, target_fun=None):
        """
        Mimics the behavior of a server node. Can be both source or target of a connection.
        Inputs:
        - dt : time step.
        - source_fun: [Optional, when used as target node] function that returns a sequence of package_streams, when evaluated with the current time_step.
        - target_fun: [Optional, when used as source node] function that returns a sequence of bandwidth_load and memory_load, when evaluated with the current time_step.
        """
        self.predict = []
        self.record = {}
        self.obj_name = name
        self.source_fun = source_fun
        self.target_fun = target_fun
        self.dt = dt
        # Initialize client:
        self.time = np.array([[0]])  # 1,1 array for consistency.
        self.simulate()

    def simulate(self):
        """
        Mimics the simulation of the server nodes.
        """
        self.predict.append({})
        if self.source_fun:
            self.predict[-1]['v_out_circuit'] = self.source_fun(self.time)
        if self.target_fun:
            self.predict[-1]['bandwidth_load'], self.predict[-1]['memory_load'] = self.target_fun(self.time)

        # Set the first element of the predicted values as the first element of the recorded values.
        for predict_key in self.predict[0].keys():
            if not predict_key in self.record.keys():  # When initializing.
                self.record[predict_key] = []
            self.record[predict_key].append(self.predict[0][predict_key][0])

        self.time += self.dt


class distributed_network:
    def __init__(self, circuits, delay=0):
        self.connections, self.nodes = self.circ_2_network(circuits)
        self.analyze_connections(delay=delay)
        self.time_step = 0

    def simulate(self):
        # In every simulation:
        # a) Determine associated properties for every connection that are determined by the source and target:

        # Package stream is depending on the source. Create [N_timesteps x n_outputs x 1] array (with np.stack())
        # and access the element that is stored in 'output_ind' for each connection.
        self.connections['v_circuit'] = self.connections.apply(lambda row: np.stack(row['source'].predict[-1]['v_out_circuit'])[:, [row['output_ind']], :], axis=1)
        # Bandwidth and memory load are determined by the source. Create [N_timesteps x n_outputs x 1] array (with np.stack()).
        # Note that each server only has one value for bandwidth and memory load.
        self.connections['bandwidth_load'] = self.connections.apply(lambda row: np.stack(row['target'].predict[-1]['bandwidth_load']), axis=1)
        self.connections['memory_load'] = self.connections.apply(lambda row: np.stack(row['target'].predict[-1]['memory_load']), axis=1)

        # b) Iterate over all nodes, query respective I/O data from connections and simulate node
        for k, node_k in self.nodes.iterrows():
            # Simulate only if the node is an optimal_traffic_scheduler.
            if type(node_k.node) is optimal_traffic_scheduler:
                # Concatenate package streams for all inputs:
                v_in_circuit = np.concatenate(self.connections.loc[node_k['con_in'], 'v_circuit'].values, axis=1)
                # Concatenate bandwidth and memory load for all outputs:
                bandwidth_load = np.concatenate(self.connections.loc[node_k['con_out'], 'bandwidth_load'].values, axis=1)
                memory_load = np.concatenate(self.connections.loc[node_k['con_out'], 'memory_load'].values, axis=1)
                # Delay Information:
                input_delay = self.connections.loc[node_k['con_in'], 'delay'].values.reshape(-1, 1)
                output_delay = self.connections.loc[node_k['con_out'], 'delay'].values.reshape(-1, 1)

                # Correct inputs due to delay (latency):
                v_in_circuit, bandwidth_load, memory_load = node_k.node.latency_adaption(
                    v_in_circuit, bandwidth_load, memory_load, input_delay, output_delay)

                io_mapping = node_k.io_mapping
                output_partition = node_k.output_partition

                # np.matmul: If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
                v_in_buffer = output_partition@v_in_circuit[:, io_mapping]

                # bandwidth_load and memory_load is queried for each circuit (multiple circuits) may lead to the same node with the same load values.
                # use the output_partition matrix to get the load information of the connected nodes. This effectively calculates the mean value of
                # all connections that lead to the same node.
                bandwidth_load = output_partition/np.sum(output_partition, axis=1, keepdims=True)@bandwidth_load
                memory_load = output_partition/np.sum(output_partition, axis=1, keepdims=True)@memory_load

                # Simulate Node with intial condition:
                s0 = node_k.node.record['s_buffer'][-1]
                node_k.node.solve(s0, v_in_buffer, bandwidth_load, memory_load)
                # Calculate and add circuit information to node:
                node_k.node.predict[-1]['v_in_circuit'] = [v_in_circuit[k] for k in range(node_k['node'].N_steps)]
                node_k.node.simulate_circuits(output_partition)
            if type(node_k.node) is client_node:
                node_k.node.simulate()

        self.time_step += 1

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
        # Add names to each node:
        nodes_pd['name'] = nodes_pd['node'].apply(lambda obj: obj.obj_name)
        con_pd['source_name'] = con_pd['source'].apply(lambda obj: obj.obj_name)
        con_pd['target_name'] = con_pd['target'].apply(lambda obj: obj.obj_name)
        return con_pd, nodes_pd

    def analyze_connections(self, delay=0):
        """
        Analyzes the connections and nodes and adds further information
        """
        self.connections['output_ind'] = None
        # Not populated with content in this function.
        self.connections['v_circuit'] = None
        self.connections['bandwidth_load'] = None
        self.connections['memory_load'] = None
        if delay == 'random':
            self.connections['delay'] = np.random.rand(len(self.connections), 1)
        else:
            self.connections['delay'] = delay

        self.nodes['con_in'] = None
        self.nodes['n_in'] = None
        self.nodes['con_out'] = None
        self.nodes['n_out_buffer'] = None
        self.nodes['n_out_circuit'] = None
        self.nodes['io_mapping'] = None
        self.nodes['output_partition'] = None

        # Run only once to get information:
        for k, node_k in self.nodes.iterrows():
            # Boolean array that indicates in which connections node_k is the source.
            node_k['con_out'] = (self.connections['source'] == node_k['node']).values
            node_k['n_out_circuit'] = sum(node_k['con_out'])
            # The output of each node is a vector with n_out elements. 'output_ind' marks which
            # of its elements refers to which connection:
            if any(node_k['con_out']):
                self.connections.loc[node_k['con_out'], 'output_ind'] = np.arange(node_k['n_out_circuit'], dtype='int16').tolist()

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

                # Each circuit triggers one connection and therefore incoming package stream for the current node. However, the output buffers
                # may contain elements of several circuits.
                output_partition = []
                for i, source_k_i in source_k.drop_duplicates('target').iterrows():
                    output_partition.append((source_k.target == source_k_i.target).values)

                node_k['output_partition'] = np.stack(output_partition)

                node_k['n_out_buffer'] = node_k['output_partition'].shape[0]
                # With io_mapping and output_partion:
                # v_in_buffer = output_partition @ v_in_circuit[io_mapping]

                # Setup the node with the now defined n_in and n_out.
                node_k['node'].setup(n_in=node_k['n_in'], n_out_buffer=node_k['n_out_buffer'], n_out_circuit=node_k['n_out_circuit'])


# def fun(row): return np.stack(row['source'].predict['v_out_circuit'])[:, [row['output_ind']], :]
#
#
# def fun(row):
#     if type(row['source']) is client_node:
#         row['source'].get_input(10)
#     if type(row['target']) is client_node:
#         row['target'].get_output(10)
#     return np.stack(row['source'].predict['v_out_circuit'])[:, [row['output_ind']], :]
#
#
# dn.connections['v_circuit'] = dn.connections.apply(fun, axis=1)
#
# dn.connections.loc[9, 'v_circuit'].shape
#
#
# dn.connections.loc[0, 'source'].predict
#
# dn['source']


# %matplotlib qt
# from ots_visu_02 import ots_gt_plot
# visu = ots_gt_plot(dn)
# visu.show_gt()
"""
Experiment
"""
# node = dn.nodes.loc[1]
# v_out_buffer = node.node.predict['v_out_buffer']
# s_circuit = node.node.predict['s_circuit']
# s_buffer = node.node.predict['s_buffer']
# output_partition = node.output_partition
# v_out_circuit = s_circuit*(output_partition.T@(v_out_buffer/np.maximum(s_buffer, 1e-6)))
#
#
# node.n_out_buffer
# node.output_partition.shape
# s_circuit
"""
Experiment with IO
"""
# node_k = dn.nodes.loc[1]
# source_k = dn.connections[node_k.node == dn.connections['source']]
# target_k = dn.connections[node_k.node == dn.connections['target']]
# output_circ = source_k.circuit.values  # output = for which connection is the node 'source'
# input_circ = target_k.circuit.values  # input = for which connection is the node 'target'
# # index vector to sort all incoming streams to the appropriate output buffers:
# io_mapping = np.argsort(input_circ)[np.argsort(output_circ)]
#
# # For each unique connection that has the current node as a target:
# output_partition = []
# for i, source_k_i in source_k.drop_duplicates('target').iterrows():
#     # Determine all other nodes, that have the current node as a target.
#     output_partition.append((source_k.target == source_k_i.target).values)
#
# output_partition
#
#
# node_k.output_partition
