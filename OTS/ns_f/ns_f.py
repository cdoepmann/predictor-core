import numpy as np
import pandas as pd
import pdb
from itertools import compress


class server:
    def __init__(self, setup_dict, ident, name):
        self.ident = ident
        self.obj_name = name

    def setup(self, n_in, n_out, package_list, empty_list):
        self.n_in = n_in
        self.n_out = n_out
        self.package_list = package_list
        self.empty_list = empty_list

        self.input_buffer = []
        for i in range(self.n_in):
            self.input_buffer.append([])

        self.output_buffer = []
        for i in range(self.n_out):
            self.output_buffer.append([])

    def add_2_buffer(self, buffer_ind, circuit, n_packets, ident=None):
        if not ident:
            ident = self.ident.get(n_packets)
        index, self.empty_list = np.split(self.empty_list,[n_packets])
        self.package_list.loc[index] = pd.DataFrame({'ident':ident, 'circuit':circuit})
        self.output_buffer[buffer_ind]+=index.tolist()


class connection:
    def __init__(self, latency_fun=lambda t: 0, window_size=2):
        self.latency_fun = latency_fun
        self.window_size = window_size
        self.transit_size = 0  # Number of packages in transit
        self.transit = []
        self.transit_reply = []


class network:
    def __init__(self, t0=0, dt=0.01, packet_list_size=10000):
        self.t = t0  # s
        self.dt = dt  # s

    def from_circuits(self, circuits, t0=0, dt=0.01, packet_list_size=10000):
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

            node_k['node'].setup(n_in=node_k['n_in'], n_out=node_k['n_out'], package_list=self.package_list, empty_list=self.empty_list)

    def simulate(self):
        pdb.set_trace()
        for i, con in self.connections.iterrows():
            source_buffer = con.source.output_buffer[con.source_ind]
            target_buffer = con.target.input_buffer[con.target_ind]

            """ Send packages """
            n_send = np.minimum(con.feat.window_size-con.feat.transit_size, len(source_buffer))
            send_ind = source_buffer[:n_send]
            self.package_list.loc[send_ind,'ts'] = self.t
            con.feat.transit_size += n_send
            con.feat.transit += send_ind

            """ Receive packages  and send replies """
            # Receive packages, if the current time is greater than the sending time plus the connection delay.
            t_sent = self.package_list.loc[con.feat.transit, 'ts']
            received_bool = t_sent + con.feat.latency_fun(t_sent) >= self.t #boolean table.
            received_ind = list(compress(con.feat.transit, received_bool))
            # Add the ident and circuit information of these packages to the target_buffer:
            target_buffer += received_ind
            # Reply that packages have been successfully sent and update the time.
            self.package_list.loc[received_ind,'tr'] = self.t
            con.feat.transit_reply += received_ind

            # Reset ts:
            self.package_list.loc[con.feat.transit, 'ts'] = np.inf

            # Remove packages from transit.
            con.feat.transit = list(set(con.feat.transit)-set(con.feat.transit_reply))

            """ Receive replies """
            # Receive replies, if the current time is greater than the sending time plus the connection delay.
            t_replied = self.package_list.loc[con.feat.transit_reply, 'tr']
            replied_bool = t_replied+con.feat.latency_fun(t_replied) >= self.t
            replied_ind = list(compress(con.feat.transit_reply, replied_bool))
            con.feat.transit_size -= len(replied_ind)
            # Remove packages from source_buffer for each reply.
            source_buffer = list(set(source_buffer)-set(replied_ind))

            # Reset tr:
            self.package_list.loc[con.feat.transit_reply, 'tr'] = np.inf

            # Remove packages from transit reply:
            con.feat.transit_reply = list(set(con.feat.transit_reply)-set(replied_ind))

            """ Save changes """
            con.source.output_buffer[con.source_ind] = source_buffer
            con.target.input_buffer[con.target_ind] = target_buffer

        for i, nod in self.nodes.iterrows():
            # concatenate all input buffers
            pdb.set_trace()
            input_buffer = sum(nod.node.input_buffer)

            if input_buffer:
                pdb.set_trace()
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

class data:
    def __init__(self, packet_list_size=10000):
        self.package_list = pd.DataFrame([], index=range(packet_list_size), columns=['ident', 'circuit','ts','tr'])
        self.package_list['ts'] = np.inf
        self.package_list['tr'] = np.inf
        self.empty_list = np.arange(packet_list_size)




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

nw = network()
nw.from_circuits(circuits)
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
