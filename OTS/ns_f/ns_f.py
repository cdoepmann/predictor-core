import numpy as np
import pandas as pd
import pdb
from itertools import compress


class server:
    def __init__(self, setup_dict, ident, data, name):
        self.ident = ident
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

    def add_2_buffer(self, buffer_ind, circuit, n_packets, ident=None):
        if not ident:
            ident = self.ident.get(n_packets)
        index, self.data.empty_list = np.split(self.data.empty_list, [n_packets])
        self.data.package_list.loc[index, 'ident'] = ident
        self.data.package_list.loc[index, 'circuit'] = circuit
        self.output_buffer[buffer_ind] += index.tolist()
        self.s += n_packets


class connection:
    def __init__(self, latency_fun=lambda t: 0.01, window_size=2):
        self.latency_fun = latency_fun
        self.window_size = window_size
        self.transit_size = 0  # Number of packages in transit
        self.transit = []  # Packages currently in transit
        self.window = []  # Packages currently beeing processed.
        self.transit_reply = []  # Replies for successfully received packages currently in transit


class network:
    def __init__(self, data, t0=0, dt=0.01, ):
        self.t = t0  # s
        self.dt = dt  # s
        self.data = data

    def from_circuits(self, circuits, packet_list_size=10000):
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
        self.connections['feat'] = None
        self.connections['source_ind'] = None
        self.connections['target_ind'] = None

        for i, con in self.connections.iterrows():
            con['feat'] = connection()

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
            t_sent = self.data.package_list.loc[con.feat.window, 'ts']
            timeout_bool = t_sent + con.source.timeout <= self.t
            # Remove items from current window, and adapt the window size:
            if any(timeout_bool):
                timeout_ind = list(compress(con.feat.window, timeout_bool))
                # Remove items from current window:
                con.feat.window = list(set(con.feat.window)-set(timeout_ind))
                con.feat.window_size = max(2, int(con.feat.window_size/2))

            """ Send packages """
            # If the last window is emptied or the current window is not completely in transit, start sending packages:
            if not con.feat.window or len(con.feat.window) < con.feat.window_size:
                send_candidate_ind = list(set(source_buffer)-set(con.feat.window))
                n_send = min(con.feat.window_size-len(con.feat.window), len(send_candidate_ind), int(con.source.v_max*self.dt))
                send_ind = send_candidate_ind[:n_send]
                # Add indices to current window:
                con.feat.window += send_ind
                # Send packages and update t_sent:
                con.feat.transit += send_ind
                self.data.package_list.loc[send_ind, 'ts'] = self.t

            """ Receive packages and send replies """
            # Receive packages, if the current time is greater than the sending time plus the connection delay.
            t_sent = self.data.package_list.loc[con.feat.transit, 'ts']
            received_bool = t_sent + con.feat.latency_fun(t_sent) <= self.t  # boolean table.
            if any(received_bool):
                # Packages that are candidates to enter the node:
                received_candidate_ind = list(compress(con.feat.transit, received_bool))
                # Server cant receive packets if buffer is full:
                n_received = min(len(received_candidate_ind), con.target.s_max-con.target.s)
                received_ind = received_candidate_ind[:n_received]  # TODO: optional with np.random.choice()
                # Add the ident and circuit information of these packages to the target_buffer:
                target_buffer += received_ind
                con.target.s += len(received_ind)
                # Reply that packages have been successfully sent and update the time.
                self.data.package_list.loc[received_ind, 'tr'] = self.t
                con.feat.transit_reply += received_ind
                # Reset ts for all received packages:
                self.data.package_list.loc[received_ind, 'ts'] = np.inf
                # Remove packages from transit, including those that were not accepted in the buffer.
                con.feat.transit = list(set(con.feat.transit)-set(received_candidate_ind))

            """ Receive replies """
            # Receive replies, if the current time is greater than the sending time plus the connection delay.
            t_replied = self.data.package_list.loc[con.feat.transit_reply, 'tr']
            replied_bool = t_replied+con.feat.latency_fun(t_replied) <= self.t
            if any(replied_bool):
                replied_ind = list(compress(con.feat.transit_reply, replied_bool))
                # Remove received packages from window:
                con.feat.window = list(set(con.feat.window)-set(replied_ind))
                # Remove received packages from source buffer:
                source_buffer = list(set(source_buffer)-set(replied_ind))
                con.source.s -= len(replied_ind)
                # Remove received packages from transit reply:
                con.feat.transit_reply = list(set(con.feat.transit_reply)-set(replied_ind))

                # Reset tr:
                self.data.package_list.loc[replied_ind, 'tr'] = np.inf

            # Adjust window_size if all packages have been successfully sent:
            if not con.feat.window:
                con.feat.window_size *= 2

            """ Save changes """
            con.source.output_buffer[con.source_ind] = source_buffer
            con.target.input_buffer[con.target_ind] = target_buffer

        """ Process the newly arrived packages in the server """
        for i, nod in self.nodes.iterrows():
            # concatenate all input buffers
            input_buffer_ind = sum(nod.node.input_buffer, [])
            input_buffer = self.data.package_list.loc[input_buffer_ind]
            # One group for each circuit in the input buffer:
            circuits = input_buffer.groupby('circuit').groups
            # This returns a dict with one key for each circuit. Each element of the dict is a list with indices
            # of packets belonging to that circuit.

            if input_buffer_ind:
                for k, buffer_circuit_k in enumerate(nod.node.output_buffer):
                    # nod.output_circuits[k] is a list of circuits in the current output_buffer.
                    # list(map(circuits.get, nod.output_circuits[k])) returns a list of lists with all the indices that belong in the current buffer.
                    nod.node.output_buffer[k] += np.concatenate(list(map(circuits.get, nod.output_circuits[k]))).tolist()
                # Reset input buffer:
                for i in range(nod.node.n_in):
                    nod.node.input_buffer[i] = []

        # Update time:
        self.t += self.dt


class global_ident:
    """
    Class that is shared among all nodes of the network and used to create unique identifiers
    for the packages.
    """

    def __init__(self):
        self.ident = 0

    def get(self, n=1):
        out = self.ident+np.arange(n)
        self.ident += n
        return out


class data:
    def __init__(self, packet_list_size=1000):
        self.package_list = pd.DataFrame([], index=range(packet_list_size), columns=['ident', 'circuit', 'ts', 'tr'])
        self.package_list['ts'] = np.inf
        self.package_list['tr'] = np.inf
        self.empty_list = np.arange(packet_list_size)


dat = data()
ident = global_ident()

setup_dict = {}
setup_dict['v_max'] = 1000  # packets / s
setup_dict['s_max'] = 30  # packets
setup_dict['timeout'] = 1  # s

input_1 = server(setup_dict, ident, dat, name='input_1')
input_2 = server(setup_dict, ident, dat, name='input_2')
output_1 = server(setup_dict, ident, dat, name='output_1')
output_2 = server(setup_dict, ident, dat, name='output_2')
server_1 = server(setup_dict, ident, dat, name='server_1')
server_2 = server(setup_dict, ident, dat, name='server_2')
circuits = [
    {'route': [input_1, server_1, server_2, output_1]},
    {'route': [input_2, server_1, server_2, output_2]},
]

nw = network(data=dat)
nw.from_circuits(circuits)
input_1.add_2_buffer(buffer_ind=0, circuit=0, n_packets=10)
input_2.add_2_buffer(buffer_ind=0, circuit=1, n_packets=10)

for i in range(100):
    nw.simulate()
