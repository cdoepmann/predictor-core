import numpy as np
import pandas as pd
import pdb
from itertools import compress
from sklearn.utils import shuffle


class server:
    def __init__(self, setup_dict, data, name):
        self.data = data
        self.obj_name = name
        self.s_max = setup_dict['s_max']
        self.v_max = setup_dict['v_max']
        self.timeout = setup_dict['timeout']
        self.s = 0

    def setup(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out

        self.input_buffer = []
        for i in range(self.n_in):
            self.input_buffer.append([])

        self.output_buffer = []
        for i in range(self.n_out):
            self.output_buffer.append([])

    def add_2_buffer(self, buffer_ind, circuit, n_packets):
        index, self.data.empty_list = np.split(self.data.empty_list, [n_packets])
        self.data.package_list.loc[index, 'circuit'] = circuit
        self.output_buffer[buffer_ind] += index.tolist()
        self.s += n_packets


class network:
    def __init__(self, data, t0=0, dt=0.01, ):
        self.t = t0  # s
        self.dt = dt  # s
        self.data = data

        """ Settings """
        # Connections are processed in a loop and therefore packets in the input_buffer are sorted, even though they arrive during the same time interval.
        # To have a more realistic output_buffer it is adviced to shuffle the newly processed packets before they are added.
        self.shuffle_incoming_packets = True
        # Shuffle the order in which connections are processed in each time step.
        self.shuffle_connection_processing = False

    def from_circuits(self, circuits, packet_list_size=1000):
        self.connections, self.nodes = self.circ_2_network(circuits)
        self.analyze_connections()

    def circ_2_network(self, circuits):
        """
        Create nodes and connections DataFrame from circuit dict.
        Each connection has a new attribute 'circuit' that lists all the circuits that are active.
        """
        connections = []
        nodes = {'node': []}
        for i, circuit_i in enumerate(circuits):
            connections.extend([{'source': circuit_i['route'][k], 'target': circuit_i['route'][k+1], 'circuit':[i]}
                                for k in range(len(circuit_i['route'])-1)])
            nodes['node'].extend(circuit_i['route'])
        con_pd = pd.DataFrame(connections)
        # Drop duplicates and create lists for circuits. For all items that are identical in 'source' and 'target', sum all elements
        # of 'circuit'. As circuit is a (one-dimensional) list, summing will lead to concatenation.
        con_pd = con_pd[['source', 'target', 'circuit']].groupby(['source', 'target'], as_index=False, sort=False).sum()
        nodes_pd = pd.DataFrame(nodes).drop_duplicates().reset_index().drop('index', axis=1)
        # Add names to each node:
        nodes_pd['name'] = nodes_pd['node'].apply(lambda obj: obj.obj_name)
        con_pd['source_name'] = con_pd['source'].apply(lambda obj: obj.obj_name)
        con_pd['target_name'] = con_pd['target'].apply(lambda obj: obj.obj_name)
        return con_pd, nodes_pd

    def analyze_connections(self):
        self.connections['latency_fun'] = None
        self.connections['window_size'] = None
        self.connections['window'] = None
        self.connections['transit'] = None
        self.connections['transit_reply'] = None
        self.connections['source_ind'] = None
        self.connections['target_ind'] = None

        for i, con in self.connections.iterrows():
            con['latency_fun'] = lambda t: 0.01
            con['window_size'] = 2
            con['window'] = []         # Packages currently in transit
            con['transit'] = []        # Packages currently beeing processed.
            con['transit_reply'] = []  # Replies for successfully received packages currently in transit

        self.nodes['con_target'] = None
        self.nodes['n_in'] = None
        self.nodes['con_source'] = None
        self.nodes['n_out'] = None
        self.nodes['output_circuits'] = None

        for k, node_k in self.nodes.iterrows():
            # Boolean array that indicates in which connections node_k is the source.
            node_k['con_source'] = (self.connections['source'] == node_k['node']).values
            node_k['n_out'] = sum(node_k['con_source'])
            if any(node_k['con_source']):
                # The output of each node is a vector with n_out elements. 'source_ind' marks which
                # of its elements refers to which connection:
                self.connections.loc[node_k['con_source'], 'source_ind'] = np.arange(node_k['n_out'], dtype='int16').tolist()
                # A list item for each output_buffer that contains the circuits that are in this buffer:
                node_k['output_circuits'] = self.connections.loc[node_k['con_source'], 'circuit'].tolist()

            # Boolean array that indicates in which connections node_k is the target. This determines the
            # number of inputs.
            node_k['con_target'] = (self.connections['target'] == node_k['node']).values
            node_k['n_in'] = sum(node_k['con_target'])

            if any(node_k['con_target']):
                self.connections.loc[node_k['con_target'], 'target_ind'] = np.arange(node_k['n_in'], dtype='int16').tolist()

            node_k['node'].setup(n_in=node_k['n_in'], n_out=node_k['n_out'])

    def simulate(self):
        for i, con in self.connections.iterrows():
            source_buffer = con.source.output_buffer[con.source_ind]
            target_buffer = con.target.input_buffer[con.target_ind]

            """ Check for time-outs in window """
            t_sent = self.data.package_list.loc[con.window, 'ts']
            timeout_bool = t_sent + con.source.timeout <= self.t
            # Remove items from current window, and adapt the window size:
            if any(timeout_bool):
                timeout_ind = list(compress(con.window, timeout_bool))
                # Remove items from current window:
                con.window = self.remove_from_list(con.window, timeout_ind)
                con.window_size = max(2, int(con.window_size/2))

            """ Send packages """
            # If the last window is emptied or the current window is not completely in transit, start sending packages:
            if not con.window or len(con.window) < con.window_size:
                send_candidate_ind = list(set(source_buffer)-set(con.window))
                n_send = min(con.window_size-len(con.window), len(send_candidate_ind), int(con.source.v_max*self.dt))
                send_ind = send_candidate_ind[:n_send]
                # Add indices to current window:
                con.window += send_ind
                # Send packages and update t_sent:
                con.transit += send_ind
                self.data.package_list.loc[send_ind, 'ts'] = self.t

            """ Receive packages and send replies """
            # Receive packages, if the current time is greater than the sending time plus the connection delay.
            t_sent = self.data.package_list.loc[con.transit, 'ts']
            received_bool = t_sent + con.latency_fun(t_sent) <= self.t  # boolean table.
            if any(received_bool):
                # Packages that are candidates to enter the node:
                received_candidate_ind = list(compress(con.transit, received_bool))
                # Server cant receive packets if buffer is full:
                n_received = min(len(received_candidate_ind), con.target.s_max-con.target.s)
                received_ind = received_candidate_ind[:n_received]  # TODO: optional with np.random.choice()
                # Add the index and circuit information of these packages to the target_buffer:
                target_buffer += received_ind
                con.target.s += len(received_ind)
                # Reply that packages have been successfully sent and update the time.
                self.data.package_list.loc[received_ind, 'tr'] = self.t
                con.transit_reply += received_ind
                # Reset ts for all received packages:
                self.data.package_list.loc[received_ind, 'ts'] = np.inf
                # Remove packages from transit, including those that were not accepted in the buffer.
                con.transit = list(set(con.transit)-set(received_candidate_ind))

            """ Receive replies """
            # Receive replies, if the current time is greater than the sending time plus the connection delay.
            t_replied = self.data.package_list.loc[con.transit_reply, 'tr']
            replied_bool = t_replied+con.latency_fun(t_replied) <= self.t
            if any(replied_bool):
                replied_ind = list(compress(con.transit_reply, replied_bool))
                # Remove received packages from window:
                con.window = self.remove_from_list(con.window, replied_ind)
                # Remove received packages from source buffer:
                source_buffer = self.remove_from_list(source_buffer, replied_ind)
                con.source.s -= len(replied_ind)
                # Remove received packages from transit reply:
                con.transit_reply = self.remove_from_list(con.transit_reply, replied_ind)

                # Reset tr:
                self.data.package_list.loc[replied_ind, 'tr'] = np.inf

            # Adjust window_size if all packages have been successfully sent (only if packages were also received.)
            if not con.window and any(replied_bool):
                con.window_size = min(int(con.source.v_max*self.dt), con.window_size*2)

            """ Save changes """
            con.source.output_buffer[con.source_ind] = source_buffer
            con.target.input_buffer[con.target_ind] = target_buffer

        """ Process the newly arrived packages in the server """
        for i, nod in self.nodes.iterrows():
            # concatenate all input buffers
            input_buffer_ind = sum(nod.node.input_buffer, [])
            # If something is in the input buffer and there exists at least one output buffer:
            if input_buffer_ind and nod.output_circuits:
                # Create a view of the data_table that is currently in all the input buffers:
                input_buffer = self.data.package_list.loc[input_buffer_ind]
                # Add a new row that defines which outputbuffer each packet is assigned to:
                input_buffer['to_output'] = input_buffer.apply(lambda row: self.get_output_buffer_ind(row, nod.output_circuits), axis=1)

                for k in range(len(nod.node.output_buffer)):
                    if self.shuffle_incoming_packets:
                        nod.node.output_buffer[k] += np.random.permutation(input_buffer[input_buffer['to_output'] == k].index.tolist()).tolist()
                    else:
                        nod.node.output_buffer[k] += input_buffer[input_buffer['to_output'] == k].index.tolist()
                # Reset input buffer:
                for i in range(nod.node.n_in):
                    nod.node.input_buffer[i] = []
            # For a node without output:
            if input_buffer_ind and not nod.output_circuits:
                # Reset input buffer:
                for i in range(nod.node.n_in):
                    nod.node.input_buffer[i] = []
                    nod.node.s -= len(input_buffer_ind)
                # Clear indices and allow new packets in the list.
                self.data.empty_list = np.append(self.data.empty_list, input_buffer_ind)

        # Update time:
        self.t += self.dt

    def make_measurement(self):
        self.nodes['composition'] = self.nodes.apply(self.get_server_composition, axis=1)

    @staticmethod
    def remove_from_list(list_a, list_b):
        """
        Remove all elements from list_a that are in list_b
        and return the remaining elements from list_a without changing their order.
        """
        set_b = set(list_b)
        return [a_i for a_i in list_a if a_i not in set_b]

    @staticmethod
    def get_output_buffer_ind(row, output_circuits):
        """
        Should be applied on copies of the data DataFrame (input_buffer) to determine for a given node, which
        packet is assigned to which output_buffer.

        Best explained with an example:
        Assume three output_buffer carrying the circuits:
        output_circuits = [[1,2],[3],[4,5]]

        row['circuit'] = 1 -> returns 0
        row['circuit'] = 2 -> returns 0
        row['circuit'] = 3 -> returns 1
        row['circuit'] = 4 -> returns 2
        row['circuit'] = 5 -> returns 2
        """
        return [ind_k for ind_k, output_circuits_k in enumerate(output_circuits) if row['circuit'] in output_circuits_k][0]

    @staticmethod
    def get_server_composition(row):
        if row['node'].output_buffer:
            s_i = [len(buffer_i) for buffer_i in row['node'].output_buffer]
            s_tot = sum(s_i)
            if s_tot:
                c = np.array(s_i)/s_tot
            else:
                c = np.ones(len(s_i))/len(s_i)
        else:
            s_tot = len(row['node'].output_buffer)
            c = np.array([1.0])

        assert s_tot == row['node'].s, "Calculation of storage memory seems flawed for node {}.".format(row['name'])

        return c


class data:
    def __init__(self, packet_list_size=1000):
        self.package_list = pd.DataFrame([], index=range(packet_list_size), columns=['circuit', 'ts', 'tr'])
        self.package_list['ts'] = np.inf
        self.package_list['tr'] = np.inf
        self.empty_list = np.arange(packet_list_size)
