import numpy as np
import pandas as pd
import pdb


class server:
    def __init__(self, setup_dict, ident, name):
        self.n_in = setup_dict['n_in']
        self.n_out = setup_dict['n_out']
        self.ident = ident
        self.obj_name = name

        df_template_buffer = pd.DataFrame({'circuit': [], 'ident': []}, dtype='int32')

        self.input_buffer = []
        for i in range(self.n_in):
            self.input_buffer.append(df_template_buffer)

        self.output_buffer = []
        for i in range(self.n_out):
            self.output_buffer.append(df_template_buffer)

    def add_2_buffer(self, buffer_ind, circuit, n_packets, ident=None):
        if not ident:
            ident = self.ident.get(n_packets)
        self.output_buffer[buffer_ind] = self.output_buffer[buffer_ind].append(pd.DataFrame({'circuit': circuit, 'ident': ident}, dtype='int32'))


class connection:
    def __init__(self, source, source_ind, target, target_ind, latency_fun=lambda t: 0, window_size=2):
        self.source = source
        self.source_ind = source_ind
        self.target = target
        self.target_ind = target_ind
        self.latency_fun = latency_fun
        self.window_size = window_size
        self.transit_size = 0  # Number of packages in transit
        self.transit = pd.DataFrame({'circuit': [], 'ident': [], 'tsent': []})
        self.transit_reply = pd.DataFrame({'circuit': [], 'ident': [], 'tsent': []})


class network:
    def __init__(self, nodes, connections, mapping, t0=0, dt=0.01):
        if not type(nodes) is list:
            self.nodes = [nodes]
        else:
            self.nodes = nodes
        if not type(connections) is list:
            self.connections = [connections]
        else:
            self.connections = connections

        self.mapping = mapping

        self.t = t0  # s
        self.dt = dt  # s

    def simulate(self):
        for con in self.connections:
            source_buffer = con.source.output_buffer[con.source_ind]
            target_buffer = con.target.input_buffer[con.target_ind]

            """ Send packages """
            n_send = np.minimum(con.window_size-con.transit_size, len(source_buffer))
            send = source_buffer.head(n_send)
            send['tsent'] = self.t
            con.transit_size += n_send
            con.transit = con.transit.append(send)

            """ Receive packages  and send replies """
            # Receive packages, if the current time is greater than the sending time plus the connection delay.
            received = con.transit[con.transit['tsent']+con.latency_fun(con.transit['tsent']) >= self.t]
            # Add the ident and circuit information of these packages to the target_buffer:
            target_buffer = target_buffer.append(received[target_buffer.columns])
            # Reply that packages have been successfully sent and update the time.
            received['tsent'] = self.t
            con.transit_reply = con.transit_reply.append(received)
            # Remove packages from transit.
            con.transit = con.transit[con.transit['ident'].isin(received['ident']) == 0]

            """ Receive replies """
            # Receive replies, if the current time is greater than the sending time plus the connection delay.
            replied = con.transit_reply[con.transit_reply['tsent']+con.latency_fun(con.transit['tsent']) >= self.t]
            # Remove packages from source_buffer for each reply.
            source_buffer = source_buffer[source_buffer['ident'].isin(replied['ident']) == 0]
            con.transit_reply = con.transit_reply[con.transit_reply['ident'].isin(replied['ident']) == 0]

            """ Save changes """
            con.source.output_buffer[con.source_ind] = source_buffer
            con.target.input_buffer[con.target_ind] = target_buffer

        for nod in nodes:
            None


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


ident = global_ident()

setup_dict = {}
input = server({'n_in': 0, 'n_out': 1}, ident, name='input')
server_1 = server({'n_in': 1, 'n_out': 1}, ident, name='server_1')
output = server({'n_in': 1, 'n_out': 0}, ident, name='output')
connection_1 = connection(source=input, source_ind=0, target=server_1, target_ind=0)
connection_2 = connection(source=server_1, source_ind=0, target=output, target_ind=0)

mapping = np.array([[-1, 1], [-1, 1]])

nw = network([input, server_1, output], [connection_1, connection_2], mapping)

input.add_2_buffer(buffer_ind=0, circuit=0, n_packets=10)


nw.simulate()


# df = nw.nodes[0].output_buffer[0]
#
# df[df['ident'].isin([0, 3, 6]) == 0]
