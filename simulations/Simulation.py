from numerical import rk4_order_n
import numpy as np
import threading

class Simulation:
    def __init__(self, initial_state, transfer, t0=0.0, dt=0.001):

        self.past_states = [initial_state]
        self.t_series = [t0]
        self.curr_state = initial_state
        self.transfer = transfer
        self.dt = dt
        self.running = False
        self.t = t0

    def update(self, dt):
        self.t, self.curr_state = rk4_order_n(self.transfer, self.t, self.curr_state, dt)
        self.past_states.append(self.curr_state)
        self.t_series.append(self.t)

    def start(self):
        self.running=True
    
    def stop(self):
        self.running=False