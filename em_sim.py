import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from numerical import *
from physics import *

# pos, vel - 3 vector (position and velocity)
# charge - signed scalar
# e_field, b_field - 3 dimensional vector field (electric and magnetic respectively) 

mu0 = 2

def compute_b_field(t, state):
    b_fields = [lambda t, s: np.cross(state[i,1,:-2], state[i, 0, :-2] -s[:,0,:-2]) for i in range(state.shape[0])]
    for i in range(state.shape[0]):
        b_fields += [lambda t, s: np.cross(state[i,1,:-2], state[i, 0, :-2] -s[:,0,:-2])] 
    return lambda t, s: np.array([b(t, s) for b in b_fields])

def simulate_EM():
    plt.ion()
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    t_series = [0.0]
    x_series = [0.0]
    y_series = [0.0]
    z_series = [0.0]
    l1, = ax.plot3D(x_series, y_series, z_series, 'blue')

    t = 0.0

    state = np.array([[[0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0, -1.0]], [[1.0, 0.5, 0.0, 1.0, -1.0], [1.0, -0.5, 1.0, 1.0, 1.0]]])
    #print(state.shape)
    print(compute_b_field(t, state)(t, state))
    # B = lambda t, x_derivs: np.array([np.sin(500*t), np.cos(500*t)*np.sin(300*x_derivs[0,0]), -np.sin(200*t)])
    # E = lambda t, x_derivs: -x_derivs[0]/(1 + np.linalg.norm(x_derivs[0])**2)
    # ddxdt = lambda t, x_derivs: accel(m, lorentz(t, x_derivs, q, E, B))
    # dxdt = lambda t, x_derivs: x_derivs[1]
    # while(True):
    #     t, state = rk4_order_n([dxdt, ddxdt], t, state)
    #     t_series.append(t)
    #     x_series.append(state[0, 0])
    #     y_series.append(state[0, 1])
    #     z_series.append(state[0, 2])
    #     l1.set_data_3d(x_series, y_series, z_series)
    #     fig.canvas.draw()
    #     fig.canvas.flush_events()
    #     #print(t)
    #     plt.show()

    #ax.plot(x_series, y_series_rot)
    #plt.show()
simulate_EM()