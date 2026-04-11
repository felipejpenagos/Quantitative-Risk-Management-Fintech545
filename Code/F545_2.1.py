import numpy as np
import pandas as pd

# exponentially weighted covariance with lambda decay
def ew_covariance(filepath, lambda_decay=0.97):
    df = pd.read_csv(filepath)
    n = len(df)
    
    # calculate exponential weights (most recent gets highest weight)
    weights = np.array([(1 - lambda_decay) * (lambda_decay ** i) for i in range(n)])
    weights = weights[::-1]  # reverse so most recent is last
    weights = weights / weights.sum()  # normalize to sum to 1
    
    # calculate weighted mean
    weighted_mean = (df * weights[:, np.newaxis]).sum(axis=0)
    
    # calculate weighted covariance manually
    n_vars = len(df.columns)
    cov_matrix = np.zeros((n_vars, n_vars))
    
    for i in range(n_vars):
        for j in range(n_vars):
            dev_i = df.iloc[:, i] - weighted_mean.iloc[i]
            dev_j = df.iloc[:, j] - weighted_mean.iloc[j]
            cov_matrix[i, j] = (weights * dev_i * dev_j).sum()
    
    return pd.DataFrame(cov_matrix, columns=df.columns, index=df.columns)

# calculate EW covariance
result = ew_covariance('data/test2.csv', lambda_decay=0.97)

# check if it matches
expected = pd.read_csv('data/testout_2.1.csv')
print(np.allclose(result.values, expected.values))