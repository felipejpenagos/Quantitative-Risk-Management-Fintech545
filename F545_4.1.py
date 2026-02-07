import numpy as np
import pandas as pd

# Cholesky for PSD matrices (handles zero eigenvalues)
def chol_psd(matrix, epsilon=1e-8):
    n = matrix.shape[0]
    root = np.zeros((n, n))
    
    for j in range(n):
        # calculate diagonal element
        s = 0.0
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])
        
        temp = matrix[j, j] - s
        
        # check if close to zero (handles floating point error)
        if 0 >= temp >= -epsilon:
            temp = 0.0
        
        root[j, j] = np.sqrt(temp) if temp > 0 else 0.0
        
        # calculate off-diagonal elements
        if root[j, j] != 0.0:
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (matrix[i, j] - s) * ir
    
    return root

# INPUT: testout_3_1.csv
input_matrix = pd.read_csv('data/testout_3.1.csv')
matrix = input_matrix.values

# calculate Cholesky root
result = chol_psd(matrix)

# OUTPUT: should match testout_4_1.csv
expected = pd.read_csv('data/testout_4.1.csv')
print(np.allclose(result, expected.values, atol=1e-7))