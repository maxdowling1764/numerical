import numpy as np
from simulations.Simulation import Simulation

def rlc_transfer(dV_sdt, R, L, C):
    def transfer(t, state):
        return dV_sdt(t)/L(t) - (R(t)*state[1,:,:]/L(t)) - (state[0,:,:]/C(t))
    return [lambda t, state: state[1,:,:], transfer]

class NodalRLCSimulation(Simulation):
    def __init__(self, initial_state, V_s, dV_sdt, R, L, C, delta_t=0.001):
        self.V_s = V_s
        self.dV_sdt = dV_sdt
        self.R = R
        self.L = L
        self.C = C

        NodalRLCSimulation.__init__(self, initial_state, transfer=rlc_transfer(dV_sdt, R, L, C), dt=delta_t)
    
    