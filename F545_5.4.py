import numpy as np
import pandas as pd
from numpy.linalg import eigh

# Higham's algorithm to find nearest PSD matrix
def higham_psd(matrix, max_iterations=100, tol=1e-9):
    n = matrix.shape[0]
    invSD = None
    out = matrix.copy()
    if not np.allclose(np.diag(out), 1.0):
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD
    Y = out.copy()
    Delta_S = np.zeros((n, n))
    prev_norm = np.inf
    for _ in range(max_iterations):
        R = Y - Delta_S
        vals, vecs = eigh(R)
        vals = np.maximum(vals, 0)
        X = vecs @ np.diag(vals) @ vecs.T
        Delta_S = X - R
        Y = X.copy()
        np.fill_diagonal(Y, 1.0)
        norm = np.linalg.norm(Y - out, 'fro')
        if abs(norm - prev_norm) < tol:
            break
        prev_norm = norm
    out = Y
    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        out = invSD @ out @ invSD
    return out

# simulate using higham_psd then chol_psd
# read input covariance matrix

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

# fix with Higham then simulate
fixed = higham_psd(cov)
L = chol_psd(fixed)
Z = np.random.normal(0, 1, size=(cov.shape[0], 100000))
sim = (L @ Z).T

result = pd.DataFrame(np.cov(sim.T), columns=cols)
expected = pd.read_csv('data/testout_5.4.csv')
print(np.allclose(result.values, expected.values, atol=1e-3))