import numpy as np
import pandas as pd
from numpy.linalg import eigh

# PCA-based simulation, 
# keep only enough components to explain 99% of variance, then simulate using those
def simulate_pca(cov_matrix, n_sim=100000, pct_explained=0.99):
    vals, vecs = eigh(cov_matrix)
    
    # sort largest to smallest
    idx = vals.argsort()[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    
    # keep only positive eigenvalues
    pos = vals > 1e-8
    vals = vals[pos]
    vecs = vecs[:, pos]
    
    # find how many components explain 99% of variance
    total = vals.sum()
    cumulative = np.cumsum(vals) / total
    n_components = np.searchsorted(cumulative, pct_explained) + 1
    
    vals = vals[:n_components]
    vecs = vecs[:, :n_components]
    
    # simulate using top components
    # B = vecs @ np.diag(np.sqrt(vals)) is the transformation to get correlated normals
    B = vecs @ np.diag(np.sqrt(vals))
    Z = np.random.normal(0, 1, size=(n_components, n_sim))
    return (B @ Z).T

np.random.seed(0)
cov = pd.read_csv('data/test5_2.csv').values
cols = pd.read_csv('data/test5_2.csv').columns

sim = simulate_pca(cov, n_sim=100000, pct_explained=0.99)

result = pd.DataFrame(np.cov(sim.T), columns=cols)
expected = pd.read_csv('data/testout_5.5.csv')
print(np.allclose(result.values, expected.values, atol=1e-2))
print(result)
# The results are close to exp. covariance matrix