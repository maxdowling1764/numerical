from simulations.IterativeSolver import IterativeSolver

class ImplicitSolver(IterativeSolver):
    def __init__(self, t0, initial_state, ds):
        self.t0 = t0
        self.initial_state = initial_state
        self.ds = ds
        IterativeSolver.__init__()