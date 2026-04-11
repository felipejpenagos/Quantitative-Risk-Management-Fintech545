import numpy as np
import pandas as pd

# function to calculate correlation, skipping rows with missing data
# (not covariance, which was in the previous function)
def correlation_skip_missing(filepath):
    df = pd.read_csv(filepath)
    clean_df = df.dropna()  # remove any rows with NaN
    return clean_df.corr()

# calculate correlation
result = correlation_skip_missing('data/test1.csv')

# check if it matches
expected = pd.read_csv('data/testout_1.2.csv')
print(np.allclose(result.values, expected.values))