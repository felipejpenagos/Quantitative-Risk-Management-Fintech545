import numpy as np
import pandas as pd
from scipy.optimize import minimize

cov = pd.read_csv('data/test5_2.csv').values
means = pd.read_csv('data/test10_3_means.csv')['Mean'].values
rf = 0.04
n = len(means)

def neg_sharpe(w, means, cov, rf):
    ret = w @ means
    vol = np.sqrt(w @ cov @ w)
    return -(ret - rf) / vol

w0 = np.ones(n) / n
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = [(0.1, 0.5)] * n

result = minimize(neg_sharpe, w0, args=(means, cov, rf), method='SLSQP',
                  bounds=bounds, constraints=constraints,
                  options={'ftol': 1e-12, 'maxiter': 1000})

weights = result.x
print(pd.DataFrame({'W': weights}).to_string(index=False))

expected = pd.read_csv('data/testout10_4.csv')['W'].values
print(f"\nMatches expected: {np.allclose(weights, expected, atol=1e-6)}")