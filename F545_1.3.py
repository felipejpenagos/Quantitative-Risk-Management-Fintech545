import numpy as np
import pandas as pd

# pairwise covariance - uses all available data for each pair
def covariance_pairwise(filepath):
    df = pd.read_csv(filepath)
    # pairwise deletion - calculates cov for each pair using their shared valid rows
    return df.cov(min_periods=1)

# calculate pairwise covariance
result = covariance_pairwise('data/test1.csv')

# check if it matches
expected = pd.read_csv('data/testout_1.3.csv')
print(np.allclose(result.values, expected.values))