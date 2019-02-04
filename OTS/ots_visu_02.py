import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from casadi import *

from optimal_traffic_scheduler import optimal_traffic_scheduler
from distributed_network import input_node, output_node
try:
    import graph_tool.all as gt
    import graph_tool
    # We need some Gtk and gobject functions
    from gi.repository import Gtk, Gdk, GdkPixbuf, GObject
except:
    from urllib.parse import quote
    l1 = quote('medium.com/@ronie/installing-graph-tool-for-python-3-on-anaconda-3f76d9004979')
    l2 = quote('git.skewed.de/count0/graph-tool/wikis/installation-instructions')
    print('Graph-Tool could not be imported. See: https://{0}\
    and See: https://{1} for install informations.\
    This prohibits the use of ots_gt_plot for network animation.\
    The ots_plotter class is still available for network analysis.'.format(l1, l2))


class ots_plotter:
    """
    Creates plots with MPC state and prediction for one or multiple OTS objects.
    Can be used for animations.
    """

    def __init__(self, ots):
        # ots must be list object to iterate over its entries.
        if type(ots) == list:
            self.ots = ots
        else:
            self.ots = [ots]

        nrows = 0
        for ots_i in self.ots:
            nrows += ots_i.n_out

        self.fig, self.ax = plt.subplots(nrows=nrows, ncols=3, figsize=(16, 9), sharex=True)
        self.ax = np.atleast_2d(self.ax)  # Otherwise indexing fails, when nrows=1

        self.color = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def plot(self, k):
        for ax_i in self.ax.ravel():
            ax_i.cla()

        lines = []
        offset = 0

        for server_i, ots_i in enumerate(self.ots):
            # Get time instances of prediction:
            pred_time = np.arange(start=ots_i.time, stop=ots_i.time+(ots_i.N_steps+1)*ots_i.dt, step=ots_i.dt)
            # Calculate from v__in and composition matrix the individual incoming package streams for each buffer:
            v_in_buffer = np.concatenate([ots_i.record['v_in_buffer'][-1]]+[c@v for c, v in zip(ots_i.predict['c'], ots_i.predict['v_in'])], axis=1)
            # Concatenate records and predictions (create numpy n-d-array) to comply with matplotlib API
            record = {name: np.concatenate(val, axis=1) for name, val in ots_i.record.items()}
            # For the predictions: Add the last element of the record to the sequence to avoid discontinuities in lines.
            predict = {name: np.concatenate((ots_i.record[name][-1], *val), axis=1) for name, val in ots_i.predict.items()}

            for out_k in range(ots_i.n_out):
                """Diagram 01: Incoming and Outgoing packages. """
                lines.append(self.ax[out_k+offset, 0].plot([], [], linewidth=0))  # Dummy to get legend entry
                lines.append(self.ax[out_k+offset, 0].step(record['time'][0], record['v_in_buffer'][out_k], color=self.color[0]))
                lines.append(self.ax[out_k+offset, 0].step(record['time'][0], record['v_out'][out_k], color=self.color[1]))
                lines.append(self.ax[out_k+offset, 0].plot([], [], linewidth=0))  # Dummy to get legend entry
                lines.append(self.ax[out_k+offset, 0].step(pred_time, v_in_buffer[out_k], color=self.color[0], linestyle='--'))
                lines.append(self.ax[out_k+offset, 0].step(pred_time, predict['v_out'][out_k], color=self.color[1], linestyle='--'))
                self.ax[out_k+offset, 0].legend([line[0] for line in lines[-6:]], ['Recorded', 'Incoming', 'Outoing', 'Predicted', 'Incoming', 'Outgoing'],
                                                loc='upper left', ncol=2, title='Package Streams')
                self.ax[out_k+offset, 0].set_ylim([0, ots_i.v_max*1.1])

                """Diagram 02: Buffer Memory. """
                self.ax[out_k+offset, 1].set_title('Server {0}, Output buffer {1}'.format(server_i+1, out_k+1))
                lines.append(self.ax[out_k+offset, 1].plot([], [], linewidth=0))  # Dummy to get legend entry
                lines.append(self.ax[out_k+offset, 1].step(record['time'][0], record['s'][out_k], color=self.color[0]))
                lines.append(self.ax[out_k+offset, 1].step(pred_time, predict['s'][out_k], color=self.color[0], linestyle='--'))
                self.ax[out_k+offset, 1].legend([line[0] for line in lines[-3:]], ['Buffer Memory', 'Recorded', 'Predicted'], loc='upper left')
                self.ax[out_k+offset, 1].set_ylim([0, ots_i.s_max*1.1])

                """Diagram 03: Load. """
                lines.append(self.ax[out_k+offset, 2].plot([], [], linewidth=0))  # Dummy to get legend entry
                lines.append(self.ax[out_k+offset, 2].step(record['time'][0], record['bandwidth_load_target'][out_k], color=self.color[0]))
                lines.append(self.ax[out_k+offset, 2].step(record['time'][0], record['memory_load_target'][out_k], color=self.color[1]))
                lines.append(self.ax[out_k+offset, 2].plot([], [], linewidth=0))  # Dummy to get legend entry
                lines.append(self.ax[out_k+offset, 2].step(pred_time, predict['bandwidth_load_target'][out_k], color=self.color[0], linestyle='--'))
                lines.append(self.ax[out_k+offset, 2].step(pred_time, predict['memory_load_target'][out_k], color=self.color[1], linestyle='--'))
                self.ax[out_k+offset, 2].legend([line[0] for line in lines[-6:]], ['Recorded', 'Bandwidth', 'Memory', 'Predicted', 'Bandwidth', 'Memory'],
                                                loc='upper left', ncol=2, title='Server Load')
                self.ax[out_k+offset, 2].set_ylim([-0.1, 1.1])

            offset += ots_i.n_out

        # Return all line objects
        return lines


class ots_gt_plot:
    """
    Graph-tool animation for the distributed network with optimal_traffic_scheduler.
    """

    def __init__(self, dn):
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
        edge_list, node_list = dn.connections, dn.nodes

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
        """
        Display current state of the network without animation.
        """
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
