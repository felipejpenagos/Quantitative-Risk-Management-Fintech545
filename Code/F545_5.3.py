import numpy as np
import pandas as pd
from numpy.linalg import eigh

def near_psd(matrix, epsilon=0.0):
    n = matrix.shape[0]
    invSD = None
    out = matrix.copy()
    if not np.allclose(np.diag(out), 1.0):
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD
    vals, vecs = eigh(out)
    vals = np.maximum(vals, epsilon)
    T = 1.0 / (vecs ** 2 @ vals)
    T = np.diag(np.sqrt(T))
    B = T @ vecs @ np.diag(np.sqrt(vals))
    out = B @ B.T
    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        out = invSD @ out @ invSD
    return out

def chol_psd(matrix):
    n = matrix.shape[0]
    root = np.zeros((n, n))
    for j in range(n):
        s = np.dot(root[j, :j], root[j, :j]) if j > 0 else 0.0
        temp = matrix[j, j] - s
        if 0 >= temp >= -1e-8:
            temp = 0.0
        root[j, j] = np.sqrt(temp) if temp > 0 else 0.0
        if root[j, j] != 0.0:
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (matrix[i, j] - s) * ir
    return root

np.random.seed(0)
cov = pd.read_csv('data/test5_3.csv').values
cols = pd.read_csv('data/test5_3.csv').columns

# fix with near_psd then simulate
fixed = near_psd(cov)
L = chol_psd(fixed)
Z = np.random.normal(0, 1, size=(cov.shape[0], 100000))
sim = (L @ Z).T

result = pd.DataFrame(np.cov(sim.T), columns=cols)
expected = pd.read_csv('data/testout_5.3.csv')
print(np.allclose(result.values, expected.values, atol=1e-3))
print(result)