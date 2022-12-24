import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from numerical import *
from physics import *

def simulate_lorenz():
    plt.ion()
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x_series = []
    y_series = []
    z_series = []

    l1, = ax.plot3D(x_series, y_series, z_series, 'blue')

    t = 0.0
    x_derivs = np.random.rand(1,3)*20
    sigma = 10.0
    rho = 28.0
    beta = 8.0/3.0
    dxdt = lambda t, x_derivs: np.array([sigma*(x_derivs[0,1] - x_derivs[0,0]), x_derivs[0,0]*(rho - x_derivs[0,2]) - x_derivs[0,1], x_derivs[0,0]*x_derivs[0,1] - beta*x_derivs[0,2]])

    while(True):
        t, x_derivs = rk4_order_n([dxdt], t, x_derivs)
        x_series.append(x_derivs[0, 0])
        y_series.append(x_derivs[0, 1])
        z_series.append(x_derivs[0, 2])
        l1.set_data_3d(x_series, y_series, z_series)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show()