import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from numerical import *
from physics import *

def simulate_RLC():
    fig, ax = plt.subplots()
    t_series = [0.0]
    i_series = [0.0]
    t = 0.0
    x_derivs = np.array([[0], [0]])

    V0 = 120.0
    w = 376.99
    L = 5.0
    C = 0.01
    R = 100.0

    ddydx = lambda t, x_derivs: (V0*w/L)*np.cos(w*t) - (R*x_derivs[1]/L) - (x_derivs[0]/C)
    dydx = lambda t, x_derivs: x_derivs[1]
    
    for i in range(0, 1000):
        t, x_derivs = rk4_order_n([dydx, ddydx], t, x_derivs)
        t_series.append(t)
        i_series.append(x_derivs[0,0])

    ax.plot(t_series, i_series)
    plt.show()