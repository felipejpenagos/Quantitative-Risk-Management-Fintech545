import numpy as np
import pandas as pd
from numpy.linalg import eigh

# fix non-PSD matrix using Rebonato-Jackel method
def near_psd(matrix, epsilon=0.0):
    n = matrix.shape[0]
    
    # check if it's a correlation matrix (diagonal = 1) or covariance
    invSD = None
    out = matrix.copy()
    
    # if covariance, convert to correlation
    if not np.allclose(np.diag(out), 1.0):
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD
    
    # eigenvalue decomposition
    vals, vecs = eigh(out)
    
    # set negative eigenvalues to epsilon (or 0)
    vals = np.maximum(vals, epsilon)
    
    # reconstruct with scaling matrix T
    T = 1.0 / (vecs ** 2 @ vals)
    T = np.diag(np.sqrt(T))
    
    B = T @ vecs @ np.diag(np.sqrt(vals))
    out = B @ B.T
    
    # convert back to covariance if needed
    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        out = invSD @ out @ invSD
    
    return out

# read the non-PSD matrix from 1.3
input_matrix = pd.read_csv('data/testout_1.3.csv')
matrix = input_matrix.values

# fix it
result = near_psd(matrix)

# check if it matches
expected = pd.read_csv('data/testout_3.1.csv')
print(np.allclose(result, expected.values))