import numpy as np
import pandas as pd

# function to calculate covariance, skipping rows with missing data
def covariance_skip_missing(filepath):
    df = pd.read_csv(filepath)
    clean_df = df.dropna()  # remove any rows with NaN
    return clean_df.cov()

# calculate covariance
result = covariance_skip_missing('data/test1.csv')

# check if it matches
expected = pd.read_csv('data/testout_1.1.csv')
print(np.allclose(result.values, expected.values))


# Output should be: "True" 
# it worked.