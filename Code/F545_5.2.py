import numpy as np
import pandas as pd

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
cov = pd.read_csv('data/test5_2.csv').values
cols = pd.read_csv('data/test5_2.csv').columns

# simulate using chol_psd since matrix is PSD
L = chol_psd(cov)
Z = np.random.normal(0, 1, size=(cov.shape[0], 100000))
sim = (L @ Z).T

result = pd.DataFrame(np.cov(sim.T), columns=cols)
expected = pd.read_csv('data/testout_5.2.csv')
print(np.allclose(result.values, expected.values, atol=1e-2))
# The results are close to the expected covariance matrix, but there is some variability due to the randomness.

