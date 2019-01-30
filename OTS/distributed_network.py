import numpy as np
import pandas as pd
from optimal_traffic_scheduler import optimal_traffic_scheduler
import pdb
import graph_tool.all as gt
import graph_tool
# We need some Gtk and gobject functions
from gi.repository import Gtk, Gdk, GdkPixbuf, GObject


class input_node:
    def __init__(self, v_in_traj):
        """
        Mimics the behavior of a server node.
        Contains information about the future package stream of the client.
        """
        self.v_in_traj = v_in_traj
        self.predict = {}

    def get_input(self, sim_step, N_horizon):
        self.predict['v_out'] = self.v_in_traj[sim_step:sim_step+N_horizon]


class output_node:
    def __init__(self, bandwidth_load, memory_load):
        """
        Mimics the behavior of a server node.
        Contains information about the future bandwidth and memory load of the receiver.
        """
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
            if not all(np.concatenate([np.sum(c_i_k, axis=0)-1 < 1e-6 for c_i_k in c_i])):
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

            if len(v_out_source) > 1:
                v_in = np.hstack(v_out_source)
            else:
                v_in = v_out_source[0]

            # Target node(s):
            if type(connection_i['target']) is list:
                assert len(connection_i['target']) == connection_i['node'].n_out, 'Connection has the wrong number of outputs.'
                bandwidth_load = np.hstack([source_i.predict['bandwidth_load'] for source_i in connection_i['target']])
                memory_load = np.hstack([source_i.predict['memory_load'] for source_i in connection_i['target']])
            else:
                bandwidth_load = connection_i['target'].predict['bandwidth_load']
                memory_load = connection_i['target'].predict['memory_load']

            # Simulate Node
            scheduler_result = connection_i['node'].solve(s0, v_in, c, bandwidth_load, memory_load)

        self.sim_step += 1


class ots_gt_plot:
    def __init__(self, dn, connections):
        # Base line settings:
        self.settings = {}
        self.settings['min_vert_size'] = 20

        self.dn = dn

        # Standard matplotlib colors as vectors with opacity as last entry.
        self.colors = [(0.12, 0.47, 0.71, 1.), (1.0, 0.5, 0.05, 1.),
                       (0.17, 0.62, 0.17, 1.), (0.84, 0.15, 0.16, 1.),
                       (0.58, 0.4, 0.74, 1.), (0.55, 0.34, 0.29, 1.),
                       (0.89, 0.47, 0.76, 1.)]

        # Create Graph and populate with edges and vertices:
        g = gt.Graph()
        self.g = g
        # Convert connections to edges and nodes:
        edge_list, node_list = get_edges_nodes(connections)

        # Add a vertex to the graph for every node and save it with the respective node object:
        node_list['vert'] = None
        for i, node_i in node_list.iterrows():
            node_list['vert'][i] = g.add_vertex()
        self.node_list = node_list

        # Add an edge to the graph for every connection and save it with the respective 'source' and 'target' objects:
        edge_list['edge'] = None
        for i, edge_i in edge_list.iterrows():
            source_i = node_list[node_list['node'] == edge_i['source']]
            target_i = node_list[node_list['node'] == edge_i['target']]
            edge_list['edge'][i] = g.add_edge(source_i.vert.values[0], target_i.vert.values[0])
        self.edge_list = edge_list

        # Create property maps for the vertices:
        vert_prop = {}
        vert_prop['size'] = g.new_vertex_property('double')
        vert_prop['shape'] = g.new_vertex_property('string')
        vert_prop['text'] = g.new_vertex_property('string')
        vert_prop['pie_fractions'] = g.new_vertex_property('vector<double>')
        vert_prop['halo'] = g.new_vertex_property('bool')
        vert_prop['halo_size'] = g.new_vertex_property('double')
        vert_prop['halo_color'] = g.new_vertex_property('vector<double>')
        vert_prop['fill_color'] = g.new_vertex_property('vector<double>')
        self.vert_prop = vert_prop

        # Create property maps for the edges:
        edge_prop = {}
        edge_prop['pen_width'] = g.new_edge_property('double')
        edge_prop['text'] = g.new_edge_property('string')

        self.edge_prop = edge_prop
        self.pos = gt.sfdp_layout(g, K=0.5)

    def show_gt(self):
        self.set_properties()
        gt.graph_draw(self.g, pos=self.pos, vprops=self.vert_prop, eprops=self.edge_prop, output_node=(800, 400))

    def anim_gt(self, c_list):
        self.win = gt.GraphWindow(self.g, self.pos, geometry=(1000, 500), vprops=self.vert_prop, eprops=self.edge_prop)
        self.c_list = c_list

        # Bind the function above as an 'idle' callback.
        cid = GObject.idle_add(self.anim_update)

        self.win.connect("delete_event", Gtk.main_quit)
        self.win.show_all()
        Gtk.main()

    def anim_update(self):
        """
        Update function that is called by the animation. Simulates the distributed network, sets the properties of
        the visualization depending on the current state and plots the visualization.
        """
        self.dn.simulate(c_list=self.c_list)
        self.set_properties()

        self.win.graph.regenerate_surface()
        self.win.graph.queue_draw()

        return True

    def set_properties(self):
        # Assign properties for each node:
        for i, node_i in self.node_list.iterrows():
            # Differentiate between Input object, output object and ots object:
            if type(node_i['node']) == optimal_traffic_scheduler:
                # Get information from node and calculate composition of buffer
                bandwidth_load = node_i['node'].record['bandwidth_load'][-1]
                memory_load = node_i['node'].record['memory_load'][-1]
                s = node_i['node'].record['s'][-1]
                if not np.sum(s) == 0:
                    pie_fractions = s / np.sum(s)
                else:
                    # If all buffer are empty, display that all are equally full.
                    pie_fractions = np.ones(s.shape)/np.size(s)
                # Set properties:
                self.vert_prop['size'][node_i['vert']] = np.maximum(np.sum(s), self.settings['min_vert_size'])
                self.vert_prop['text'][node_i['vert']] = '{:.0f}'.format(float(np.sum(s)))
                self.vert_prop['fill_color'][node_i['vert']] = [0.15, 0.73, 0.05, 0.8]  # dummy value
                self.vert_prop['halo'][node_i['vert']] = True
                self.vert_prop['halo_color'][node_i['vert']] = [1.0, 0.16, 0.27, 0.5]
                self.vert_prop['halo_size'][node_i['vert']] = 1 + 20*bandwidth_load/self.vert_prop['size'][node_i['vert']]
                self.vert_prop['shape'][node_i['vert']] = 'pie'
                self.vert_prop['pie_fractions'][node_i['vert']] = pie_fractions.ravel().tolist()
            elif type(node_i['node']) == input_node:
                self.vert_prop['size'][node_i['vert']] = 20
                self.vert_prop['shape'][node_i['vert']] = 'square'
                self.vert_prop['fill_color'][node_i['vert']] = [0.15, 0.73, 0.05, 0.8]
            elif type(node_i['node']) == output_node:
                self.vert_prop['size'][node_i['vert']] = 20
                self.vert_prop['shape'][node_i['vert']] = 'square'
                self.vert_prop['fill_color'][node_i['vert']] = [1.0, 0.16, 0.27, 0.98]
        # Assign properties for each edge:
        for i, edge_i in self.edge_list.iterrows():
            if edge_i['con_type'] == 0:  # source to node connection
                v = edge_i['target'].record['v_in'][-1][edge_i['node_ind']]
            elif edge_i['con_type'] == 1:  # node to target connection
                v = edge_i['source'].record['v_out'][-1][edge_i['node_ind']]
            self.edge_prop['pen_width'][edge_i['edge']] = v
            self.edge_prop['text'][edge_i['edge']] = '{:.2f}'.format(float(v))


def get_edges_nodes(connections):
    """
    Helper function for ots_gt_plot.

    Input: Connections must be of the format:
    connections = [
        {'source': [input], 'node': ots_1, 'target': [ots_2, ots_3]},
        ...
        }
    where each element of the list is a dict with the keywords 'source', 'node' and 'target'.
    'source' and 'target' must be lists of input, output or ots objects. 'node' must be an 'ots' object.
    Returns:
    - pandas.DataFrame with columns 'source' and 'target'. That define edges of a network.
    There are no duplicate edges.
    - pandas.DataFrame with columns 'nodes'. That defines the nodes of a network
    """
    # Initialize empty list. Each list item will contain a dict with 'source' and 'target'.
    # Note that each element in the "connections" list contains exactly one node but may contain multiple sources and targets.
    edges = []
    nodes = {'node': []}
    for connection_i in connections:
        nodes['node'].append(connection_i['node'])
        for k, source_k in enumerate(connection_i['source']):
            con_obj = 'ots' if type(source_k) == optimal_traffic_scheduler else 'input_node'
            edges.append({'source': source_k, 'target': connection_i['node'], 'con_type': 0, 'node_ind': k, 'con_obj': con_obj})
            nodes['node'].append(source_k)
        for k, target_k in enumerate(connection_i['target']):
            con_obj = 'ots' if type(target_k) == optimal_traffic_scheduler else 'output_node'
            edges.append({'source': connection_i['node'], 'target': target_k, 'con_type': 1, 'node_ind': k, 'con_obj': con_obj})
            nodes['node'].append(target_k)
    edges_df = pd.DataFrame(edges)
    nodes_df = pd.DataFrame(nodes)

    # Drop the duplicates, reset the index and delete the column (axis=1) that contains the old indices.
    edges_df_rmv_duplicate = edges_df.drop_duplicates(subset=['source', 'target']).reset_index().drop('index', axis=1)
    nodes_df_rmv_duplicate = nodes_df.drop_duplicates().reset_index().drop('index', axis=1)

    return edges_df_rmv_duplicate, nodes_df_rmv_duplicate
