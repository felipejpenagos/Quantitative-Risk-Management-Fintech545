import numpy as np
import pandas as pd

# hybrid: EW variance (lambda=0.97) and EW correlation (lambda=0.94)
def ew_cov_hybrid(filepath, lambda_var=0.97, lambda_corr=0.94):
    df = pd.read_csv(filepath)
    n = len(df)
    
    # weights for variance calculation (lambda=0.97)
    weights_var = np.array([(1 - lambda_var) * (lambda_var ** i) for i in range(n)])
    weights_var = weights_var[::-1]
    weights_var = weights_var / weights_var.sum()
    
    # weights for correlation calculation (lambda=0.94)
    weights_corr = np.array([(1 - lambda_corr) * (lambda_corr ** i) for i in range(n)])
    weights_corr = weights_corr[::-1]
    weights_corr = weights_corr / weights_corr.sum()
    
    # calculate weighted mean for variance
    weighted_mean_var = (df * weights_var[:, np.newaxis]).sum(axis=0)
    
    # calculate weighted mean for correlation
    weighted_mean_corr = (df * weights_corr[:, np.newaxis]).sum(axis=0)
    
    # calculate variances using lambda=0.97
    n_vars = len(df.columns)
    variances = np.zeros(n_vars)
    for i in range(n_vars):
        dev_i = df.iloc[:, i] - weighted_mean_var.iloc[i]
        variances[i] = (weights_var * dev_i ** 2).sum()
    
    # calculate correlation matrix using lambda=0.94
    corr_matrix = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(n_vars):
            dev_i = df.iloc[:, i] - weighted_mean_corr.iloc[i]
            dev_j = df.iloc[:, j] - weighted_mean_corr.iloc[j]
            cov_ij = (weights_corr * dev_i * dev_j).sum()
            std_i = np.sqrt((weights_corr * dev_i ** 2).sum())
            std_j = np.sqrt((weights_corr * dev_j ** 2).sum())
            corr_matrix[i, j] = cov_ij / (std_i * std_j)
    
    # combine: Cov = D * Corr * D where D is diagonal std dev matrix
    std_devs = np.sqrt(variances)
    cov_matrix = np.outer(std_devs, std_devs) * corr_matrix
    
    return pd.DataFrame(cov_matrix, columns=df.columns, index=df.columns)

# calculate hybrid EW covariance
result = ew_cov_hybrid('data/test2.csv', lambda_var=0.97, lambda_corr=0.94)

# check if it matches
expected = pd.read_csv('data/testout_2.3.csv')
print(np.allclose(result.values, expected.values))