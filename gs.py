import numpy as np
import matplotlib.pyplot as plt

# Gauss-Seidel Method
# 
# @param A $n \times n$ invertible, diagonally dominant coeff matrix
# @param b $n$-dimensional vector
# Return: $n$ dimensional vector $x$ satisfying $Ax=b$
def gs(A, b, max_iter=100, epsilon=0.0001):
    x_curr = np.random.rand(b.shape)
    x_last = np.random.rand(b.shape)
    
    k = 0
    err = epsilon + 1
    while err > epsilon and k < max_iter:
        err = np.linalg.norm(x_curr - x_last)
        x_last = x_curr
        # Actual computation time: $O(n^2)
        for i in range(x_curr.shape[0]):
            # Lower will be using the values that have already been updated for the current iteration
            lower = sum([A[i, j]*x_curr[j] for j in range(i-1)])
            upper = sum([A[i, j]*x_last[j] for j in range(i, x_curr.shape[0])])

            x_curr[i] = (1.0/A[i, i])*(b - lower - upper)
    
    return x_curr

def rk_4_order1(f, x, y, h = 0.0001):
    k1 = h * f(x, y)
    k2 = h * f(x + 0.5*h, y + 0.5*k1)
    k3 = h * f(x + 0.5*h, y + 0.5*k2)
    k4 = h * f(x + h, y + k3)
    return (x + h, y + (1.0 / 6.0) * (k1 + 2*k2 + 2*k3 + k4))

def rk4_order2(f, g, t, x, dxdt, dt = 0.001):
    k1 = dt * f(t, x, dxdt)
    l1 = dt * g(t, x, dxdt)
    k2 = dt * f(t + 0.5*dt, x + 0.5*k1, dxdt + 0.5*l1)
    l2 = dt * g(t + 0.5*dt, x + 0.5*k1, dxdt + 0.5*l1)
    k3 = dt * f(t + 0.5*dt, x + 0.5*k2, dxdt + 0.5*l2)
    l3 = dt * g(t + 0.5*dt, x + 0.5*k2, dxdt + 0.5*l2)
    k4 = dt * f(t + dt, x + k3, dxdt + l3)
    l4 = dt * g(t + dt, x + k3, dxdt + l3)

    return (t + dt, x + (1.0/6.0) * (k1 + 2*k2 + 2*k3 + k4), dxdt + (1.0/6.0) * (l1 + 2*l2 + 2*l3 + l4))

def rk4_order_n(f_n, t, x_derivs, dt = 0.001):
    # Offsets will be our generalization of k and l used in rk4_order2
    # Each offset vector is indexed by [i, j]
    # i is RK order, j is the index of the function in f_n used to compute the offset 
    offsets = np.zeros((4, len(f_n), x_derivs.shape[1]))
    offsets[0] = dt * np.array([f(t, x_derivs) for f in f_n])
    offsets[1] = dt * np.array([f(t + 0.5*dt, x_derivs + 0.5*offsets[0]) for f in f_n])
    offsets[2] = dt * np.array([f(t + 0.5*dt, x_derivs + 0.5*offsets[1]) for f in f_n])
    offsets[3] = dt * np.array([f(t + dt, x_derivs + offsets[2]) for f in f_n])
    #print(offsets)
    return (t + dt, x_derivs + (1.0/6.0) * (offsets[0] + 2*offsets[1] + 2*offsets[2] + offsets[3]))

# pos, vel - 3 vector (position and velocity)
# charge - signed scalar
# e_field, b_field - 3 dimensional vector field (electric and magnetic respectively) 
def lorentz(pos, vel, charge, e_field, b_field):
    return charge*e_field(pos) + charge*(np.cross(vel, b_field(pos)))
 
def accel(m, F):
    return (1.0/m)*F

def sim_loop():
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


sim_loop()