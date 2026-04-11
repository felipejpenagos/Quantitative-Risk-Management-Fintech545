import numpy as np
import pandas as pd
from scipy.optimize import minimize

cov = pd.read_csv('data/test5_2.csv').values
n = len(cov)
b = np.array([1.0, 1.0, 1.0, 1.0, 0.5])

def portfolio_vol(w, cov):
    return np.sqrt(w @ cov @ w)

def risk_contributions(w, cov):
    sigma = portfolio_vol(w, cov)
    return w * (cov @ w) / sigma

def sse_budgeted(w, cov, b):
    rc = risk_contributions(w, cov)
    rc_adj = rc / b
    mean_rc_adj = np.mean(rc_adj)
    return np.sum((rc_adj - mean_rc_adj) ** 2)

w0 = np.ones(n) / n
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = [(0, None)] * n

result = minimize(sse_budgeted, w0, args=(cov, b), method='SLSQP',
                  bounds=bounds, constraints=constraints,
                  options={'ftol': 1e-12, 'maxiter': 1000})

weights = result.x
print(pd.DataFrame({'W': weights}).to_string(index=False))

expected = pd.read_csv('data/testout10_2.csv')['W'].values
print(f"\nMatches expected: {np.allclose(weights, expected, atol=1e-6)}")