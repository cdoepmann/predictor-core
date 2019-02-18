import numpy as np
import pandas as pd
import pdb


class server:
    def __init__(self, setup_dict, ident, name):
        self.ident = ident
        self.obj_name = name

        if 'n_in' and 'n_out' in setup_dict:
            self.n_in = setup_dict['n_in']
            self.n_out = setup_dict['n_out']
            self.setup()

    def setup(self, n_in=None, n_out=None):
        if n_in:
            self.n_in = n_in
        if n_out:
            self.n_out = n_out

        self.df_template_buffer = pd.DataFrame({'circuit': [], 'ident': []}, dtype='int32')

        self.input_buffer = []
        for i in range(self.n_in):
            self.input_buffer.append(self.df_template_buffer)

        self.output_buffer = []
        for i in range(self.n_out):
            self.output_buffer.append(self.df_template_buffer)

    def add_2_buffer(self, buffer_ind, circuit, n_packets, ident=None):
        if not ident:
            ident = self.ident.get(n_packets)
        self.output_buffer[buffer_ind] = self.output_buffer[buffer_ind].append(pd.DataFrame({'circuit': circuit, 'ident': ident}, dtype='int32'))


class connection:
    def __init__(self, latency_fun=lambda t: 0, window_size=2):
        self.latency_fun = latency_fun
        self.window_size = window_size
        self.transit_size = 0  # Number of packages in transit
        self.transit = pd.DataFrame({'circuit': [], 'ident': [], 'tsent': []})
        self.transit_reply = pd.DataFrame({'circuit': [], 'ident': [], 'tsent': []})


class network:
    def __init__(self, circuits, t0=0, dt=0.01):
        self.connections, self.nodes = self.circ_2_network(circuits)
        self.analyze_connections()

        self.t = t0  # s
        self.dt = dt  # s

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

        for k, node_k in self.nodes.iterrows():
            # Boolean array that indicates in which connections node_k is the source.
            node_k['con_source'] = (self.connections['source'] == node_k['node']).values
            node_k['n_out'] = sum(node_k['con_source'])
            # The output of each node is a vector with n_out elements. 'source_ind' marks which
            # of its elements refers to which connection:
            if any(node_k['con_source']):
                self.connections.loc[node_k['con_source'], 'source_ind'] = np.arange(node_k['n_out'], dtype='int16').tolist()

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

            """ Send packages """
            n_send = np.minimum(con.feat.window_size-con.feat.transit_size, len(source_buffer))
            send = source_buffer.head(n_send).copy()
            send['tsent'] = self.t
            con.feat.transit_size += n_send
            con.feat.transit = con.feat.transit.append(send)

            """ Receive packages  and send replies """
            # Receive packages, if the current time is greater than the sending time plus the connection delay.
            received = con.feat.transit[con.feat.transit['tsent']+con.feat.latency_fun(con.feat.transit['tsent']) >= self.t]
            # Add the ident and circuit information of these packages to the target_buffer:
            target_buffer = target_buffer.append(received[target_buffer.columns])
            # Reply that packages have been successfully sent and update the time.
            received['tsent'] = self.t
            con.feat.transit_reply = con.feat.transit_reply.append(received)
            # Remove packages from transit.
            con.feat.transit = con.feat.transit[con.feat.transit['ident'].isin(received['ident']) == 0]

            """ Receive replies """
            # Receive replies, if the current time is greater than the sending time plus the connection delay.
            replied = con.feat.transit_reply[con.feat.transit_reply['tsent']+con.feat.latency_fun(con.feat.transit['tsent']) >= self.t]
            con.feat.transit_size -= len(replied)
            # Remove packages from source_buffer for each reply.
            source_buffer = source_buffer[source_buffer['ident'].isin(replied['ident']) == 0]
            con.feat.transit_reply = con.feat.transit_reply[con.feat.transit_reply['ident'].isin(replied['ident']) == 0]

            """ Save changes """
            con.source.output_buffer[con.source_ind] = source_buffer
            con.target.input_buffer[con.target_ind] = target_buffer

        for i, nod in self.nodes.iterrows():
            # concatenate all input buffers
            if nod.node.input_buffer:
                input_buffer = pd.concat(nod.node.input_buffer)
                k = 0
                for _, con in self.connections[nod.con_source].iterrows():
                    nod.node.output_buffer[k] = nod.node.output_buffer[k].append(input_buffer[input_buffer['circuit'].isin(con.circuit)])
                    k += 1
            # Reset input buffer:
            for i in range(nod.node.n_in):
                nod.node.input_buffer[i] = nod.node.df_template_buffer


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


""" 01 """
# ident = global_ident()
#
# setup_dict = {}
# input = server({'n_in': 0, 'n_out': 1}, ident, name='input')
# server_1 = server({'n_in': 1, 'n_out': 1}, ident, name='server_1')
# output = server({'n_in': 1, 'n_out': 0}, ident, name='output')
# connection_1 = connection(source=input, source_ind=0, target=server_1, target_ind=0)
# connection_2 = connection(source=server_1, source_ind=0, target=output, target_ind=0)
#
# mapping = np.array([[-1, 1], [-1, 1]])
#
# nw = network([input, server_1, output], [connection_1, connection_2], mapping)
#
# input.add_2_buffer(buffer_ind=0, circuit=0, n_packets=10)
#
#
# nw.simulate()

""" 02 """
ident = global_ident()
input_1 = server({'n_in': 0, 'n_out': 1}, ident, name='input_1')
input_2 = server({'n_in': 0, 'n_out': 1}, ident, name='input_2')
output_1 = server({'n_in': 1, 'n_out': 0}, ident, name='output_1')
output_2 = server({'n_in': 1, 'n_out': 0}, ident, name='output_2')
server_1 = server({}, ident, name='server_1')
server_2 = server({}, ident, name='server_2')
circuits = [
    {'route': [input_1, server_1, server_2, output_1]},
    {'route': [input_2, server_1, server_2, output_2]},
]

nw = network(circuits)
input_1.add_2_buffer(buffer_ind=0, circuit=0, n_packets=10)
input_2.add_2_buffer(buffer_ind=0, circuit=1, n_packets=10)

for i in range(100):
    nw.simulate()


server_1.output_buffer
# server_1.input_buffer
server_2.output_buffer
#
# # nw.connections
# df = pd.concat(nw.nodes.iloc[1].node.input_buffer)
# df
# nw.connections[nw.nodes.iloc[2].con_source].iloc[0].circuit
#
# df[df['circuit'].isin([0])]

#
# a = pd.DataFrame([1, 2, 3, 4])
# a['b'] = pd.DataFrame([9, 3, 2])
# a
