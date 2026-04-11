import pandas as pd
import numpy as np

# read price data
df = pd.read_csv('data/test6.csv')

# calculate log returns for each column except Date
# log return = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
returns_df = df.copy()
for col in df.columns[1:]:  # skip Date column
    returns_df[col] = np.log(df[col] / df[col].shift(1))

# drop first row (NaN values)
returns_df = returns_df.iloc[1:]

# check if it matches
expected = pd.read_csv('data/testout6_2.csv')
print(np.allclose(returns_df.iloc[:, 1:].values, expected.iloc[:, 1:].values))