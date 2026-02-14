import numpy as np
import pandas as pd
from scipy import stats

data = pd.read_csv('data/test7_1.csv')
x = data['x1'].values

# fit normal distribution
mu = np.mean(x)
sigma = np.std(x, ddof=1)

# VaR at 95% confidence (5th percentile)
var_absolute = -stats.norm.ppf(0.05, mu, sigma)  # absolute loss
var_diff_mean = -stats.norm.ppf(0.05, 0, sigma)  # diff from mean

result = pd.DataFrame({'VaR Absolute': [var_absolute], 'VaR Diff from Mean': [var_diff_mean]})
expected = pd.read_csv('data/testout8_1.csv')
print(np.allclose(result.values, expected.values))
print(result)