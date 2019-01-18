import graph_tool.all as gt
import graph_tool
import numpy as np
# We need some Gtk and gobject functions
from gi.repository import Gtk, Gdk, GdkPixbuf, GObject
#import matplotlib.pyplot as plt

g = gt.Graph()

n_vert = 50
n_edge = 50
g.add_vertex(n_vert)

for i_, o_ in zip(np.random.choice(np.arange(n_vert), n_edge), np.random.choice(np.arange(n_vert), n_edge)):
    g.add_edge(g.vertex(i_), g.vertex(o_))


vert_size = g.new_vertex_property('double')
vert_size.a = 20*np.random.rand(g.num_vertices())+5

e_width = g.new_edge_property('double')
e_width.a = 5*np.random.rand(g.num_edges())+1

vprops = {'size': vert_size}
eprops = {'pen_width': e_width}


pos = gt.sfdp_layout(g, K=0.5)


win = gt.GraphWindow(g, pos, geometry=(500, 400), vprops=vprops, eprops=eprops)


def update_state():
    e_width.a = 5*np.random.rand(g.num_edges())+1
    vert_size.a = 20*np.random.rand(g.num_vertices())+5

    win.graph.regenerate_surface()
    win.graph.queue_draw()

    return True


# Bind the function above as an 'idle' callback.
cid = GObject.idle_add(update_state)

win.connect("delete_event", Gtk.main_quit)
win.show_all()
Gtk.main()
