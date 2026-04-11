import numpy as np
import pandas as pd

# pairwise correlation - uses all available data for each pair
def correlation_pairwise(filepath):
    df = pd.read_csv(filepath)
    # pairwise deletion for correlation
    return df.corr(min_periods=1)

# calculate pairwise correlation
result = correlation_pairwise('data/test1.csv')

# check if it matches
expected = pd.read_csv('data/testout_1.4.csv')
print(np.allclose(result.values, expected.values))