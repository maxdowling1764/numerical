from Simulation import Simulation

class ImplicitSolver(Simulation):
    def __init__(self, t0, initial_state, ds):
        Simulation.__init__(self, t0, initial_state, ds)