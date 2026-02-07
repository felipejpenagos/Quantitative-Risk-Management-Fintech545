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
                           # mathematcially, this line sigma = np.std(x, ddof=1) refers to 
                           # this formula: σ = sqrt(1/n-1 * Σ(xi - μ)²). Which is the unbiased estimator
                           # for the standard deviation of a sample.

print(f"mu\tsigma")
print(f"{mu}\t{sigma}")