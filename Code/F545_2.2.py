import numpy as np
import pandas as pd

# exponentially weighted correlation with lambda decay
def ew_correlation(filepath, lambda_decay=0.94):
    df = pd.read_csv(filepath)
    n = len(df)
    
    # calculate exponential weights (most recent gets highest weight)
    weights = np.array([(1 - lambda_decay) * (lambda_decay ** i) for i in range(n)])
    weights = weights[::-1]  # reverse so most recent is last
    weights = weights / weights.sum()  # normalize
    
    # calculate weighted mean
    weighted_mean = (df * weights[:, np.newaxis]).sum(axis=0)
    
    # calculate weighted covariance
    n_vars = len(df.columns)
    cov_matrix = np.zeros((n_vars, n_vars))
    
    for i in range(n_vars):
        for j in range(n_vars):
            dev_i = df.iloc[:, i] - weighted_mean.iloc[i]
            dev_j = df.iloc[:, j] - weighted_mean.iloc[j]
            cov_matrix[i, j] = (weights * dev_i * dev_j).sum()
    
    # convert covariance to correlation
    std_devs = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
    
    return pd.DataFrame(corr_matrix, columns=df.columns, index=df.columns)

# calculate EW correlation
result = ew_correlation('data/test2.csv', lambda_decay=0.94)

# check if it matches
expected = pd.read_csv('data/testout_2.2.csv')
print(np.allclose(result.values, expected.values))