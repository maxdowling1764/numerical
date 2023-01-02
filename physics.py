import numpy as np

def lorentz(t, state, charge, e_field, b_field):
    return charge*e_field(t,state) + charge*(np.cross(state[1], b_field(t, state)))
 
def accel(m, F):
    return F/m