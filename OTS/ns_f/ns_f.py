import numpy as np
import pandas as pd
import pdb
from itertools import compress
from sklearn.utils import shuffle
import sys
sys.path.insert(0, '../')
from optimal_traffic_scheduler import optimal_traffic_scheduler, ots_client


class server:
    def __init__(self, setup_dict, data, name):
        self.data = data
        self.obj_name = name
        self.s_max = setup_dict['s_max']
        self.v_max = setup_dict['v_max']
        self.timeout = setup_dict['timeout']
        self.s = 0
        self.control_mode = 'tcp'

    def setup(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out

        self.input_buffer = []
        for i in range(self.n_in):
            self.input_buffer.append([])

        self.output_buffer = []
        for i in range(self.n_out):
            self.output_buffer.append([])

    def set_ots(self, dt_ots, N_steps, weights):
        ots_setup = {}
        ots_setup['v_max'] = self.v_max
        ots_setup['s_max'] = self.s_max
        ots_setup['dt'] = dt_ots
        ots_setup['N_steps'] = N_steps
        ots_setup['weights'] = weights
        self.ots = optimal_traffic_scheduler(ots_setup, name='ots_'+self.obj_name, record_values=True)
        self.control_mode = 'ots'

    def set_ots_client(self, dt_ots, N_steps):
        client_setup = {}
        client_setup['v_max'] = self.v_max
        client_setup['s_max'] = self.s_max
        client_setup['dt'] = dt_ots
        client_setup['N_steps'] = N_steps
        self.ots = ots_client(client_setup, name='ots_client_'+self.obj_name)
        self.control_mode = 'ots'

    def add_2_buffer(self, buffer_ind, circuit, n_packets, tnow=0):
        if n_packets > len(self.data.empty_list):
            n_append = max(int(self.data.numel/10), n_packets)
            self.data.append_list(n_append)
        index, self.data.empty_list = np.split(self.data.empty_list, [n_packets])
        self.data.packet_list.loc[index, 'circuit'] = circuit
        self.data.packet_list.loc[index, 'tspawn'] = tnow
        self.output_buffer[buffer_ind] += index.tolist()
        self.s += n_packets


class connection_cls:
    def __init__(self, latency_fun, window_size=2):
        self.latency_fun = latency_fun
        self.window_size = 2
        self.window = []  # Packages currently in transit
        self.transit = []  # Packages currently beeing processed.
        self.transit_reply = []  # Replies for successfully received packages currently in transit


class network:
    def __init__(self, data, t0=0, dt=0.01, ):
        self.t = t0  # s
        self.dt = dt  # s
        self.data = data

        """ Settings """
        # Connections are processed in a loop and therefore packets in the input_buffer are sorted, even though they arrive during the same time interval.
        # To have a more realistic output_buffer it is adviced to shuffle the newly processed packets before they are added.
        self.shuffle_incoming_packets = True

        self.linear_growth_threshold = 10

        self.shuffle_connection_processing = True

        self.t_transmission = []

        self.control_mode = 'ots'

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
        self.connections['source_ind'] = None
        self.connections['target_ind'] = None
        self.connections['prop'] = None

        for i, con in self.connections.iterrows():
            con['prop'] = connection_cls(self.latency_fun(mean=0.07), window_size=2)

        self.nodes['con_target'] = None
        self.nodes['n_in'] = None
        self.nodes['con_source'] = None
        self.nodes['n_out'] = None
        self.nodes['output_circuits'] = None
        self.nodes['input_circuits'] = None

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
            node_k['input_circuits'] = self.connections.loc[node_k['con_target'], 'circuit'].tolist()
            node_k['n_in'] = sum(node_k['con_target'])

            if any(node_k['con_target']):
                self.connections.loc[node_k['con_target'], 'target_ind'] = np.arange(node_k['n_in'], dtype='int16').tolist()

            node_k['node'].setup(n_in=node_k['n_in'], n_out=node_k['n_out'])

    def simulate(self):
        if self.shuffle_connection_processing:
            con_list = self.connections.iloc[np.random.permutation(len(self.connections))]
        else:
            con_list = self.connections

        for i, con in con_list.iterrows():
            source_buffer = con.source.output_buffer[con.source_ind]
            target_buffer = con.target.input_buffer[con.target_ind]

            """ Check for time-outs in window """
            t_sent = self.data.packet_list.loc[con.prop.window, 'ts']
            timeout_bool = t_sent + con.source.timeout <= self.t
            # Remove items from current window, and adapt the window size:
            if any(timeout_bool):
                timeout_ind = list(compress(con.prop.window, timeout_bool))
                # Remove items from current window:
                con.prop.window = self.remove_from_list(con.prop.window, timeout_ind)
                if self.control_mode is 'tcp':
                    con.prop.window_size = max(2, con.prop.window_size/2)

            """ Send packages """
            if self.control_mode is 'tcp':
                # If the current window is smaller than the allowed window size:
                if len(con.prop.window) < int(con.prop.window_size):
                    send_candidate_ind = self.remove_from_list(source_buffer, con.prop.window)
                    n_send = int(min(con.prop.window_size-len(con.prop.window), len(send_candidate_ind), con.source.v_max*self.dt))
                    send_ind = send_candidate_ind[:n_send]
                    # Add indices to current window:
                    con.prop.window += send_ind
                    # Send packages and update t_sent:
                    con.prop.transit += send_ind
                    self.data.packet_list.loc[send_ind, 'ts'] = self.t
            elif self.control_mode is 'ots':
                send_candidate_ind = self.remove_from_list(source_buffer, con.prop.window)
                n_send = int(con.v_con[0]*self.dt)
                send_ind = send_candidate_ind[:n_send]
                # Add indices to current window:
                con.prop.window += send_ind
                # Send packages and update t_sent:
                con.prop.transit += send_ind
                self.data.packet_list.loc[send_ind, 'ts'] = self.t

            """ Receive packages and send replies """
            # Receive packages, if the current time is greater than the sending time plus the connection delay.
            t_sent = self.data.packet_list.loc[con.prop.transit, 'ts']
            received_bool = t_sent + con.prop.latency_fun(self.t) <= self.t  # boolean table.
            if any(received_bool):
                # Packages that are candidates to enter the node:
                received_candidate_ind = list(compress(con.prop.transit, received_bool))
                # Server cant receive packets if buffer is full:
                n_received = min(len(received_candidate_ind), con.target.s_max-con.target.s)
                received_ind = received_candidate_ind[:n_received]  # TODO: optional with np.random.choice()
                # Add the index and circuit information of these packages to the target_buffer:
                target_buffer += received_ind
                con.target.s += len(received_ind)
                # Reply that packages have been successfully sent and update the time.
                self.data.packet_list.loc[received_ind, 'tr'] = self.t
                con.prop.transit_reply += received_ind
                # Reset ts for all received packages:
                self.data.packet_list.loc[received_ind, 'ts'] = np.inf
                # Remove packages from transit, including those that were not accepted in the buffer.
                con.prop.transit = self.remove_from_list(con.prop.transit, received_candidate_ind)
                # Note if packets were dropped:
                dropped_ind = self.remove_from_list(received_candidate_ind, received_ind)
                self.data.packet_list.loc[dropped_ind, 'n_dropped'] += 1

            """ Receive replies """
            # Receive replies, if the current time is greater than the sending time plus the connection delay.
            t_replied = self.data.packet_list.loc[con.prop.transit_reply, 'tr']
            replied_bool = t_replied+con.prop.latency_fun(self.t) <= self.t
            if any(replied_bool):
                replied_ind = list(compress(con.prop.transit_reply, replied_bool))
                # Remove received packages from window:
                con.prop.window = self.remove_from_list(con.prop.window, replied_ind)
                # Remove received packages from source buffer:
                source_buffer = self.remove_from_list(source_buffer, replied_ind)
                con.source.s -= len(replied_ind)
                # Remove received packages from transit reply:
                con.prop.transit_reply = self.remove_from_list(con.prop.transit_reply, replied_ind)

                # Reset tr:
                self.data.packet_list.loc[replied_ind, 'tr'] = np.inf

            # Adjust window_size if any packages were received. Adjust window size exponentially if it is lower than the linear_growth_threshold.
            if any(replied_bool) and self.control_mode is 'tcp':
                if con.prop.window_size < self.linear_growth_threshold:
                    con.prop.window_size += sum(replied_bool)
                else:
                    con.prop.window_size += sum(replied_bool)/con.prop.window_size
                # Limited by maximum bandwith of connection:
                con.prop.window_size = min(con.source.v_max*self.dt, con.prop.window_size)

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
                input_buffer = self.data.packet_list.loc[input_buffer_ind]
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
                # Create a view of the data_table that is currently in all the input buffers:
                input_buffer = self.data.packet_list.loc[input_buffer_ind]
                # Get total transmission time of received packets:
                t_transmission = self.t - input_buffer['tspawn']
                self.data.packet_list.loc[input_buffer_ind, 'ttransit'] = t_transmission
                self.t_transmission += t_transmission.tolist()
                # Reset input buffer:
                for i in range(nod.node.n_in):
                    nod.node.input_buffer[i] = []
                    nod.node.s -= len(input_buffer_ind)

        # Update time:
        self.t += self.dt

    def run_ots(self):
        """
        a) Measure the current state of the network:
        This populates the columns 's_circuit' and 's_buffer' of the self.nodes DataFrame.
        """
        self.make_measurement()

        """
         b) Determine associated properties for every connection that are determined by the source and target:
        """
        # Packet stream is depending on the source. Create [N_timesteps x n_outputs x 1] array (with np.stack())
        # and access the element that is stored in 'output_ind' for each connection.
        self.connections['v_con'] = self.connections.apply(lambda row: np.stack(row['source'].ots.predict[-1]['v_out'])[:, [row['source_ind']], :], axis=1)
        self.connections['c_con'] = self.connections.apply(lambda row: [cv_out_i[row['source_ind']] for cv_out_i in row['source'].ots.predict[-1]['cv_out']], axis=1)
        # Allowed packet stream is determinded by the target:
        self.connections['v_max'] = self.connections.apply(lambda row: np.stack(row['target'].ots.predict[-1]['v_in_max'])[:, [row['target_ind']], :], axis=1)
        # Create [N_timesteps x n_outputs x 1] array (with np.stack()).
        # Note that each server only has one value for bandwidth and memory load.
        self.connections['bandwidth_load_target'] = self.connections.apply(lambda row: np.stack(row['target'].ots.predict[-1]['bandwidth_load']), axis=1)
        self.connections['memory_load_target'] = self.connections.apply(lambda row: np.stack(row['target'].ots.predict[-1]['memory_load']), axis=1)
        self.connections['bandwidth_load_source'] = self.connections.apply(lambda row: np.stack(row['source'].ots.predict[-1]['bandwidth_load']), axis=1)
        self.connections['memory_load_source'] = self.connections.apply(lambda row: np.stack(row['source'].ots.predict[-1]['memory_load']), axis=1)

        # b) Iterate over all nodes, query respective I/O data from connections and simulate node
        for k, node_k in self.nodes.iterrows():
            # Simulate only if the node is an optimal_traffic_scheduler.
            if type(node_k.node.ots) is optimal_traffic_scheduler:
                # Concatenate package streams for all inputs:
                v_in_req = np.concatenate(self.connections.loc[node_k['con_target'], 'v_con'].values, axis=1)
                # cv_in = [[cv_i_k for cv_i_k in cv_i] for cv_i in zip(*self.connections.loc[node_k['con_target'], 'c_con'].values)]
                cv_in = np.stack([np.concatenate(cv_i, axis=0) for cv_i in zip(*self.connections.loc[node_k['con_target'], 'c_con'].values)], axis=0)
                # cv_in = np.concatenate(self.connections.loc[node_k['con_target'], 'c_con'].values, axis=1)
                # And the allowed packet streams:
                v_out_max = np.concatenate(self.connections.loc[node_k['con_source'], 'v_max'].values, axis=1)
                # Concatenate bandwidth and memory load for all outputs:
                bandwidth_load_target = np.concatenate(self.connections.loc[node_k['con_source'], 'bandwidth_load_target'].values, axis=1)
                memory_load_target = np.concatenate(self.connections.loc[node_k['con_source'], 'memory_load_target'].values, axis=1)
                # Concatenate bandwidth and memory load for all input:
                bandwidth_load_source = np.concatenate(self.connections.loc[node_k['con_target'], 'bandwidth_load_source'].values, axis=1)
                memory_load_source = np.concatenate(self.connections.loc[node_k['con_target'], 'memory_load_source'].values, axis=1)
                # # Delay Information:
                input_delay = self.connections.loc[node_k['con_target']].apply(lambda con: con.prop.latency_fun(self.t), axis=1).values.reshape(-1, 1)
                output_delay = self.connections.loc[node_k['con_source']].apply(lambda con: con.prop.latency_fun(self.t), axis=1).values.reshape(-1, 1)
                # # Correct inputs due to delay (latency):
                cv_in = node_k.node.ots.latency_adaption(cv_in, type='input', input_delay=input_delay)
                v_in_req = node_k.node.ots.latency_adaption(v_in_req, type='input', input_delay=input_delay)
                v_out_max = node_k.node.ots.latency_adaption(v_out_max, type='output', input_delay=output_delay)
                bandwidth_load_target = node_k.node.ots.latency_adaption(bandwidth_load_target, type='output', output_delay=output_delay)
                memory_load_target = node_k.node.ots.latency_adaption(memory_load_target, type='output', output_delay=output_delay)
                bandwidth_load_source = node_k.node.ots.latency_adaption(bandwidth_load_source, type='input', output_delay=input_delay)
                memory_load_source = node_k.node.ots.latency_adaption(memory_load_source, type='input', output_delay=input_delay)
                # Make lists from concatenated numpy arrays:
                v_in_req, v_out_max, bandwidth_load_target, memory_load_target, bandwidth_load_source, memory_load_source = [[el for el in arr] for arr in [v_in_req, v_out_max, bandwidth_load_target, memory_load_target, bandwidth_load_source, memory_load_source]]
                cv_in = [np.split(el, np.cumsum(node_k.node.ots.n_circuit_in)[:-1].tolist(), axis=0) for el in cv_in]
                # Simulate Node with intial condition:
                s_buffer_0 = np.array(node_k['s_buffer']).reshape(-1, 1)
                s_circuit_0 = np.array(node_k['s_circuit']).reshape(-1, 1)
                s_transit_0 = np.array(node_k['s_transit']).reshape(-1, 1)
                success = node_k.node.ots.solve(s_buffer_0, s_circuit_0, s_transit_0, v_in_req, cv_in, v_out_max, bandwidth_load_target, memory_load_target, bandwidth_load_source, memory_load_source, output_delay)
                if not success:
                    pdb.set_trace()

            if type(node_k.node.ots) is ots_client:
                s_circuit_0 = np.array(node_k['s_circuit']).reshape(-1, 1)
                s_buffer_0 = np.array(node_k['s_buffer']).reshape(-1, 1)

                if node_k.n_in > 0:
                    v_in_req = [el for el in np.concatenate(self.connections.loc[node_k['con_target'], 'v_con'].values, axis=1)]
                    node_k.node.ots.update_prediction(s_buffer_0, s_buffer_0, v_in_req=v_in_req)
                if node_k.n_out > 0:
                    v_out_max = [el for el in np.concatenate(self.connections.loc[node_k['con_source'], 'v_max'].values, axis=1)]
                    node_k.node.ots.update_prediction(s_buffer_0, s_buffer_0, v_out_max=v_out_max)

        self.t_next_iter += self.dt_ots

    def make_measurement(self):
        self.nodes['s_transit'] = self.nodes.apply(self.get_transit_size, axis=1)
        self.nodes['s_circuit'] = self.nodes.apply(self.get_circuit_size, axis=1)
        self.nodes['s_buffer'] = self.nodes.apply(self.get_buffer_size, axis=1)

        check = (self.nodes['s_buffer'].apply(np.sum)-self.nodes['s_circuit'].apply(np.sum)).dropna().values
        if not np.allclose(check, 0):
            pdb.set_trace()

    def setup_ots(self, dt_ots, N_steps):
        """
        Apply the .setup((n_in, n_out, circuits_in, circuits_ou)) method for each ots object assigned to the servers.
        """
        self.dt_ots = dt_ots
        self.N_steps = N_steps
        for i, node_i in self.nodes.iterrows():
            output_delay = self.connections.loc[node_i['con_source']].apply(lambda con: con.prop.latency_fun(self.t), axis=1).values
            node_i.node.ots.setup(node_i['n_in'], node_i['n_out'], node_i['input_circuits'], node_i['output_circuits'], output_delay=output_delay)
            assert node_i.node.ots.dt == dt_ots, "The node: {0} has an inconsistent timestep.".format(node_i.node.obj_name)
            assert node_i.node.ots.N_steps == N_steps, "The node: {0} has an inconsistent prediction horizon.".format(node_i.node.obj_name)

        self.t_next_iter = 0

        self.control_mode = 'ots'

        # self.nodes.apply(lambda row: row['node'].ots.setup(row['n_in'], row['n_out'], row['input_circuits'], row['output_circuits']), axis=1)

    @staticmethod
    def latency_fun(mean, var=0):
        return lambda x: mean

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

    def get_buffer_size(self, row):
        if row['node'].output_buffer:
            sb = [len(buffer_i) for buffer_i in row['node'].output_buffer]
            st = self.connections.loc[row.con_source].apply(lambda con: len(con.prop.transit)+len(con.prop.transit_reply), axis=1).tolist()
            #sb = [sb_i-st_i for sb_i, st_i in zip(sb, st)]
        else:
            sb = len(row['node'].output_buffer)
            st = len(row['node'].output_buffer)

        return sb

    def get_transit_size(self, row):
        if row['node'].output_buffer:
            st = self.connections.loc[row.con_source].apply(lambda con: len(con.prop.transit)+len(con.prop.transit_reply), axis=1).tolist()
        else:
            st = len(row['node'].output_buffer)
        return st

    def get_circuit_size(self, row):
        if row['node'].output_buffer:
            output_buffer_concat = np.concatenate(row['node'].output_buffer)
        else:
            output_buffer_concat = []

        if row['output_circuits'] is not None:
            packets_per_circuit = self.data.packet_list.loc[output_buffer_concat, ['circuit']].groupby('circuit').size()
            sc = [[packets_per_circuit[output_i_circuits_j] if output_i_circuits_j in packets_per_circuit else 0 for output_i_circuits_j in output_i_circuits] for output_i_circuits in row['output_circuits']]
        else:
            sc = None

        return sc


class data:
    def __init__(self, packet_list_size=1000):
        self.df_dict = {'circuit': 0, 'ts': np.inf, 'tr': np.inf, 'tspawn': np.inf, 'ttransit': np.inf, 'n_dropped': 0}
        self.packet_list = pd.DataFrame(self.df_dict, index=range(packet_list_size))
        self.empty_list = np.arange(packet_list_size)
        self.numel = packet_list_size

    def append_list(self, packet_list_size=1000):
        packet_list_append = pd.DataFrame(self.df_dict, index=range(self.numel, packet_list_size+self.numel))
        self.packet_list = pd.concat((self.packet_list, packet_list_append))
        self.empty_list = np.append(self.empty_list, np.arange(self.numel, packet_list_size+self.numel))
        self.numel += packet_list_size
