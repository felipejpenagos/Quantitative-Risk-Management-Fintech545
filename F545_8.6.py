import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(0)
data = pd.read_csv('data/test7_2.csv')
x = data['x1'].values

# fit t-distribution
params = stats.t.fit(x)
nu, mu, scale = params

# simulate 100k draws
sim = stats.t.rvs(nu, mu, scale, size=100000)

# calculate ES from simulation
alpha = 0.05

# ES = average of all losses in the left tail (below 5th percentile)
threshold = np.percentile(sim, alpha * 100)
losses_in_tail = sim[sim <= threshold]
es_absolute = -losses_in_tail.mean()

# ES diff from mean assumes 0 mean
sim_zero_mean = stats.t.rvs(nu, 0, scale, size=100000)
threshold_zero = np.percentile(sim_zero_mean, alpha * 100)
losses_zero = sim_zero_mean[sim_zero_mean <= threshold_zero]
es_diff_mean = -losses_zero.mean()

result = pd.DataFrame({
    'ES Absolute': [es_absolute],
    'ES Diff from Mean': [es_diff_mean]
})

expected = pd.read_csv('data/testout8_6.csv')
print(np.allclose(result.values, expected.values, atol=1e-2))