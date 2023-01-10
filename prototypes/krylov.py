import numpy as np

def get_krylov_basis(r, m, b):
    return [b] + [np.dot(np.linalg.matrix_power(m, k), b) for k in range(1, r)] 