from casadi import *
from casadi.tools import struct_symSX, struct_SX, entry


class MPC:
    def __init__(self, N_steps):
        self.N_steps = N_steps

    def setup(self, n_in=None, n_out=None, n_c=None):
        """
        n_in:             Number of Inputs
        n_out:            Number of outputs
        n_c:              Number of circuits
        """

        self.n_in = n_in
        self.n_out = n_out
        self.n_c = n_c

        self.problem_formulation()
        self.create_optim()

    def problem_formulation(self):
        """ MPC states for stage k"""
        self.mpc_xk = struct_symSX([
            entry('x', shape=(self.n_out, 1)),
            entry('x_c', shape=(self.n_c, 1)),
        ])

        """ MPC control inputs for stage k"""
        self.mpc_uk = struct_symSX([
            entry('v_in', shape=(self.n_in, 1)),
            entry('v_out', shape=(self.n_out, 1)),
        ])

        """ MPC parameters for stage k"""
        self.mpc_pk = struct_symSX([
            entry('Pb', shape=(self.n_out, self.n_c)),
            entry('Pc', shape=(self.n_c, self.n_c)),
        ])

    def create_optim(self):
        # Initialize trajectory lists (each list item, one time-step):
        self.mpc_obj_x = struct_symSX([
            entry('x', repeat=self.N_steps+1, struct=self.mpc_xk),
            entry('u', repeat=self.N_steps, struct=self.mpc_uk),
        ])
        print('mpc_xk.shape = {}'.format(self.mpc_xk.shape))

        self.mpc_obj_p = struct_symSX([
            entry('x0',  struct=self.mpc_xk),
            entry('p',   struct=self.mpc_pk),
        ])


mpc = MPC(N_steps=20)

message = 'n_in = {n_in}, n_out = {n_out}, n_c = {n_c}'
cases = [
    # Those are examples that are working:
    {'n_in': 2, 'n_out': 2, 'n_c': 2},
    {'n_in': 3, 'n_out': 4, 'n_c': 6},
    {'n_in': 7, 'n_out': 3, 'n_c': 12},
    # Those are examples that are not working:
    {'n_in': 3, 'n_out': 1, 'n_c': 6},
    {'n_in': 2, 'n_out': 1, 'n_c': 3},
]

for case in cases:
    try:
        print(message.format(**case))
        mpc.setup(**case)
        print('Working.')
        print('-------------------------------------')
    except RuntimeError as err:
        print('Not working. With the following message:')
        print(err)
