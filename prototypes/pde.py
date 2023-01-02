import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt

def cidx_to_offset(cidx, n_coeffs):
    return int(cidx - (n_coeffs - 1)/2)

def create_diagonals(c, n):
    A = np.zeros((n, n))
    for i in range(c.shape[0]):
        offset = cidx_to_offset(i, c.shape[0])
        diag = np.array([c[i]]*(n - abs(offset)))
        A += np.diag(diag, offset)
    return A

def create_diagonals_2d(c, n, m):
    A = np.zeros((n*m, n*m))
    for i in range(c.shape[0]):
        offset = cidx_to_offset(i, c.shape[0])
        diag = np.array([c[i]]*(n*m - abs(offset)))
        A += np.diag(diag, offset)
    
    for i in range(n-1):
        k = (i+1)*m
        A[k, k - 1] = 0
        A[k - 1, k] = 0
    
    return A


# @param u0: initial state of u
# @param c: Coefficients to use along diagonals of A
# @param b: Coefficients to use along diagonals of B
# @param ds: spatial step size
# @param dt: temporal step size
# @return vector uf such that A*uf = B*u0
# Evolves the system with initial condition u0 forward in time by increment dt using
# the implicit difference scheme represented by the coefficients c and b
# 
def solve_pde_implicit(u0, c, b):
    A = create_diagonals(c, u0.shape[0])
    B = create_diagonals(b, u0.shape[0])
    b_prime = np.dot(B,u0)
    #print(b_prime)
    #print(np.linalg.solve(A, b_prime))
    return np.linalg.solve(A, b_prime)

def solve_heat_dirichlet(T0, alpha, dt, ds, L):
    plt.ion()
    fig = plt.figure()
    ax = plt.axes()
    s = np.linspace(0, L, 1000)
    sigma = alpha*dt/(2*ds**2)
    c = np.array([-sigma, 1+2*sigma, -sigma])
    b = np.array([sigma, 1-2*sigma, sigma])
    T = T0
    l1, = ax.plot(s, T)
    while True:
        T = solve_pde_implicit(T, c, b)
        l1.set_ydata(T)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show()

def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def create_coeffs_2d(coeffs, n, m):
    return np.array([coeffs[0]] + [0]*(m-2) + coeffs[1:3] + [0]*(m-2) + coeffs[4])

def solve_diffusion_2d(t0, U0, U0_last, alpha, dt, ds, L):
    plt.ion()
    fig, ax = plt.subplots()
    n = int(L/ds)
    x, y = np.meshgrid(np.linspace(0, L, n), np.linspace(0, L, n))
    U = np.random.rand(n,n)*500
    U_last = U
    heatmap = ax.pcolormesh(x, y, U)
    sigma_x = alpha * dt/ds**2
    sigma_y = sigma_x
    t = t0
    print(sigma_x)
    A = create_diagonals_2d(np.array([-sigma_x] + [0]*(n-2) + [-sigma_y, 1.0+2.0*(sigma_x + sigma_y), -sigma_y] + [0]*(n-2) + [-sigma_x]), n, n)
    B = create_diagonals_2d(np.array([sigma_x] + [0]*(n-2) + [sigma_y, -2.0*(sigma_x + sigma_y), sigma_y] + [0]*(n-2) + [sigma_x]), n, n)
    
    u_flat = np.ndarray.flatten(U)
    u_last_flat = np.ndarray.flatten(U)
    # TODO: use solve banded to find inverse
    Ainv = np.linalg.inv(A)

    while True:
        tmp = np.copy(u_flat)
        
        u_flat = np.dot(Ainv, np.dot(B, u_flat) + u_last_flat)
        u_last_flat = tmp
        U = u_flat.reshape((n,n))
        t += dt
        heatmap.set_array(U)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show()


def solve_wave_dirichlet(t0, U0_last, U0_curr, c, dt, ds, L):
    plt.ion()
    fig = plt.figure()
    ax = plt.axes()
    n = int(L/ds)
    s = np.linspace(0, L, n)
    sigma = c/ds
    gamma = 1.0/dt
    A = create_diagonals(np.array([sigma**2/2.0, -(sigma**2 + gamma**2), sigma**2/2.0]), n)
    B = create_diagonals(np.array([-sigma**2/2.0, (sigma**2 - 2*gamma**2), -sigma**2/2.0]), n)
    U_curr = s**2
    U_last = U_curr
    l1, = ax.plot(s, U_curr)
    t = t0
    while True:
        tmp = np.copy(U_curr)
        U_curr = np.linalg.solve(A, np.dot(B, U_curr) + gamma**2*U_last)
        U_last = tmp
        t += dt
        l1.set_ydata(U_curr**2)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show()


alpha = 1.0
dt = 0.001
L = 2.0
ds = 0.05
m = int(L/ds)
U0_last = np.random.rand(m,m)
solve_diffusion_2d(0.0, U0_last, U0_last, 0.03, dt, ds, L)