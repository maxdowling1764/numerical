import numpy as np

def lorentz(t, x_derivs, charge, e_field, b_field):
    return charge*e_field(t,x_derivs) + charge*(np.cross(x_derivs[1], b_field(t, x_derivs)))
 
def accel(m, F):
    return F/m