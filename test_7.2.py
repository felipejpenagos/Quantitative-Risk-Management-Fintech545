import pandas as pd
import numpy as np
from scipy import stats

# Load data
data = pd.read_csv('data/test7_2.csv')
x = data['x1'].values

# Fit t-distribution using MLE
# scipy.stats.t.fit returns (df, loc, scale) where df=nu, loc=mu, scale=sigma
params = stats.t.fit(x)
nu = params[0]   # degrees of freedom
mu = params[1]   # location parameter
sigma = params[2]  # scale parameter

print(f"mu\tsigma\tnu")
print(f"{mu}\t{sigma}\t{nu}")
