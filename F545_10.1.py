import numpy as np
import pandas as pd
from scipy.optimize import minimize

cov = pd.read_csv('data/test5_2.csv').values
n = len(cov)

def portfolio_vol(w, cov):
    return np.sqrt(w @ cov @ w)

def risk_contributions(w, cov):
    sigma = portfolio_vol(w, cov)
    # gradient of vol w.r.t. weights, then element-wise times w
    return w * (cov @ w) / sigma

def sse(w, cov):
    rc = risk_contributions(w, cov)
    mean_rc = np.mean(rc)
    return np.sum((rc - mean_rc) ** 2)

w0 = np.ones(n) / n  # start equal weight
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = [(0, None)] * n

result = minimize(sse, w0, args=(cov,), method='SLSQP',
                  bounds=bounds, constraints=constraints,
                  options={'ftol': 1e-12, 'maxiter': 1000})

weights = result.x
print(pd.DataFrame({'W': weights}).to_string(index=False))