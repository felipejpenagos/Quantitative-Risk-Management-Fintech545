import pandas as pd
import numpy as np

# read price data
df = pd.read_csv('data/test6.csv')

# calculate arithmetic returns for each column except Date
returns_df = df.copy()
for col in df.columns[1:]:  # skip Date column
    returns_df[col] = df[col].pct_change()

# drop first row (NaN values)
returns_df = returns_df.iloc[1:]

# check if it matches
expected = pd.read_csv('data/testout6_1.csv')
print(np.allclose(returns_df.iloc[:, 1:].values, expected.iloc[:, 1:].values))