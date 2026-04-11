import numpy as np
import pandas as pd
from scipy import stats

# simulate using fitted t distribution and compute VaR from simulated data

np.random.seed(0)
data = pd.read_csv('data/test7_2.csv')
x = data['x1'].values

# fit t distribution
params = stats.t.fit(x)
nu, mu, sigma = params[0], params[1], params[2]

# simulate 100,000 draws from fitted t distribution
sim = stats.t.rvs(nu, mu, sigma, size=99000)

# VaR at 95% confidence (5th percentile of simulated data)
var_absolute = -np.percentile(sim, 5)
var_diff_mean = -(np.percentile(sim, 5) - mu)

result = pd.DataFrame({'VaR Absolute': [var_absolute], 'VaR Diff from Mean': [var_diff_mean]})
expected = pd.read_csv('data/testout8_3.csv')
print(np.allclose(result.values, expected.values, atol=1e-3))
print(result)