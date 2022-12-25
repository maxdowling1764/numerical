import numpy as np
from simulations.Simulation import Simulation

def lorenz_transfer(t, state, sigma, rho, beta):
    return np.array([sigma*(state[0,1] - state[0,0]), state[0,0]*(rho - state[0,2]) - state[0,1], state[0,0]*state[0,1] - beta*state[0,2]])

class LorenzSimulation(Simulation):
    def __init__(self, initial_state, sigma, rho, beta, delta_t=0.001):
        Simulation.__init__(self, initial_state, transfer=[lambda t, s: lorenz_transfer(t, s, sigma, rho, beta)], dt=delta_t)
    
    