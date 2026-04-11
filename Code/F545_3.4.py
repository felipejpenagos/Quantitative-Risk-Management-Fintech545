import numpy as np
import pandas as pd
from numpy.linalg import eigh

# Higham's algorithm for nearest PSD matrix
def higham_psd(matrix, max_iterations=100, tol=1e-9):
    n = matrix.shape[0]
    
    # check if covariance or correlation
    invSD = None
    out = matrix.copy()
    
    if not np.allclose(np.diag(out), 1.0):
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD
    
    # initialize
    Y = out.copy()
    Delta_S = np.zeros((n, n))
    prev_norm = np.inf
    
    for iteration in range(max_iterations):
        # project to PSD (fix negative eigenvalues)
        R = Y - Delta_S
        vals, vecs = eigh(R)
        vals = np.maximum(vals, 0)  # set negative to 0
        X = vecs @ np.diag(vals) @ vecs.T
        
        Delta_S = X - R
        
        # project to unit diagonal (set diagonal to 1)
        Y = X.copy()
        np.fill_diagonal(Y, 1.0)
        
        # check convergence
        norm = np.linalg.norm(Y - out, 'fro')
        if abs(norm - prev_norm) < tol:
            break
        prev_norm = norm
    
    out = Y
    
    # convert back to covariance if needed
    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        out = invSD @ out @ invSD
    
    return out

# read non-PSD correlation from 1.4
input_matrix = pd.read_csv('data/testout_1.4.csv')
matrix = input_matrix.values

# fix using Higham
result = higham_psd(matrix)

# check if it matches
expected = pd.read_csv('data/testout_3.4.csv')
print(np.allclose(result, expected.values))