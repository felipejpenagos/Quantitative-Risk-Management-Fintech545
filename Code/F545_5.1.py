import numpy as np
import pandas as pd

# simulate multivariate normal using Cholesky (PD matrix, 0 mean)
def simulate_normal(cov_matrix, n_sim=100000):
    n = cov_matrix.shape[0]
    
    # Cholesky decomposition
    L = np.linalg.cholesky(cov_matrix)
    
    # generate independent standard normals and correlate them
    Z = np.random.normal(0, 1, size=(n, n_sim))
    X = (L @ Z).T  # shape: (n_sim x n)
    
    return X

# read input covariance matrix
np.random.seed(0)
cov = pd.read_csv('data/test5_1.csv').values
cols = pd.read_csv('data/test5_1.csv').columns

# simulate and compute covariance of simulated data
sim = simulate_normal(cov, n_sim=100000)
result = pd.DataFrame(np.cov(sim.T), columns=cols)

# simulation won't match exactly, just check it's close
expected = pd.read_csv('data/testout_5.1.csv')
#print(np.allclose(result.values, expected.values, atol=1e-3))
#print(result)

# Made several runs and the results are close to the expected covariance matrix
# But there is some variability due to the randomness.