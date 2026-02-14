import numpy as np
import pandas as pd
from scipy import stats

# In 8.1, we fit a normal distribution to the data and compute VaR
# In 8.2, we fit a t distribution to the data and compute VaR
# Note, the t distribution has heavier tails, so we expect higher VaR values than the normal case

data = pd.read_csv('data/test7_2.csv')
x = data['x1'].values

# fit t distribution
params = stats.t.fit(x)
nu, mu, sigma = params[0], params[1], params[2]

# VaR at 95% confidence (5th percentile)
var_absolute = -stats.t.ppf(0.05, nu, mu, sigma)
var_diff_mean = -stats.t.ppf(0.05, nu, 0, sigma)

result = pd.DataFrame({'VaR Absolute': [var_absolute], 'VaR Diff from Mean': [var_diff_mean]})
expected = pd.read_csv('data/testout8_2.csv')
print(np.allclose(result.values, expected.values))
print(result)