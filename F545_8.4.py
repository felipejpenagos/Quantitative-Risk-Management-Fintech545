import numpy as np
import pandas as pd
from scipy import stats

data = pd.read_csv('data/test7_1.csv')
x = data['x1'].values

# fit normal distribution
mu = np.mean(x)
sigma = np.std(x, ddof=1)

# ES at 95% confidence from normal distribution
alpha = 0.05
z_alpha = stats.norm.ppf(alpha)  # -1.645
phi_z = stats.norm.pdf(z_alpha)   # PDF at that point

# ES formula for normal distribution
es_diff_mean = sigma * phi_z / alpha
es_absolute = -mu + sigma * phi_z / alpha

result = pd.DataFrame({
    'ES Absolute': [es_absolute], 
    'ES Diff from Mean': [es_diff_mean]
})

expected = pd.read_csv('data/testout8_4.csv')
print(np.allclose(result.values, expected.values))