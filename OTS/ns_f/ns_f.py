import numpy as np
import pandas as pd
import pdb


class server:
    def __init__(self, setup_dict):
        self.n_in = setup_dict['n_in']
        self.n_out = setup_dict['n_out']

        df_template_buffer = pd.DataFrame({'circuit': [], 'ident': []})

        input_buffer = []
        for i in range(self.n_in):
            input_buffer.append(df_template_buffer)

        output_buffer = []
        for i in range(self.n_out):
            output_buffer.append(df_template_buffer)

    def ident(self):
        self.get_ident(3))


class connection:
    def __init__(self):
        df_con=pd.DataFrame({'circuit': [], 'ident': [], 'tsent': []})
        df_rep=pd.DataFrame({'circuit': [], 'ident': [], 'tsent': []})


class network:
    def __init__(self, nodes, connections):
        self.ident=0
        self.nodes=nodes
        self.connections=connections

    def get_ident(self, n = 1):
        return self.ident+np.arange(1, n+1)


setup_dict={}
setup_dict['n_in']=1
setup_dict['n_out']=1
server_i=server(setup_dict)
connection_i=connection()

nw=network(server_i, connection_i)

nw.nodes.ident()
