import numpy as np
from simulations.ODESolver import ODESolver

def rlc_transfer(V0, w, R, L, C):
    def transfer(t, state):
        return (V0*w/L)*np.cos(w*t) - (R*state[1]/L) - (state[0]/C)
    return [lambda t, state: state[1], transfer]

class RLC(ODESolver):
    def __init__(self, initial_state, V0, w, R, L, C, delta_t=0.001):
        self.V0 = V0
        self.w = w*2*np.pi
        self.R = R
        self.L = L
        self.C = C

        ODESolver.__init__(self, initial_state, transfer=rlc_transfer(V0, w, R, L, C), dt=delta_t)
    
    