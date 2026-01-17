import pandas as pd
import numpy as np
from scipy import stats

# Load data
data = pd.read_csv('data/test7_1.csv')
x = data['x1'].values

# Maximum likelihood estimation for normal distribution
# MLE estimators: mu = sample mean, sigma = sample std (biased estimator with n denominator)
mu = np.mean(x)
sigma = np.std(x, ddof=1)  # ddof=1 for MLE (unbiased estimator)

print(f"mu\tsigma")
print(f"{mu}\t{sigma}")