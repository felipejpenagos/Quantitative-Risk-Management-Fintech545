import numpy as np
import pandas as pd
from scipy import stats

data = pd.read_csv('data/test7_2.csv')
x = data['x1'].values

# fit t-distribution
params = stats.t.fit(x)
nu, mu, scale = params

# ES for t-distribution at 95% confidence
alpha = 0.05
t_alpha = stats.t.ppf(alpha, nu)
pdf_t_alpha = stats.t.pdf(t_alpha, nu)

# ES formula for t-distribution
es_diff_mean = scale * (nu + t_alpha**2) / (nu - 1) * pdf_t_alpha / alpha
es_absolute = -mu + es_diff_mean

result = pd.DataFrame({
    'ES Absolute': [es_absolute],
    'ES Diff from Mean': [es_diff_mean]
})

expected = pd.read_csv('data/testout8_5.csv')
print(np.allclose(result.values, expected.values))