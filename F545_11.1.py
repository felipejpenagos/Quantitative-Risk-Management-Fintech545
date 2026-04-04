import numpy as np
import pandas as pd
from scipy import stats

returns = pd.read_csv('data/test11_1_returns.csv')   # T x n asset returns
weights = pd.read_csv('data/test11_1_weights.csv')['W'].values  # initial weights
n = len(weights)
assets = returns.columns.tolist()

# --- step 1: drift weights through time ---
w = weights.copy()
port_returns = []
all_weights = []

for t in range(len(returns)):
    r = returns.iloc[t].values
    all_weights.append(w.copy())
    w_star = w * (1 + r)
    port_ret = np.sum(w_star) - 1
    port_returns.append(port_ret)
    w = w_star / (1 + port_ret)  # normalize back to sum=1

all_weights = np.array(all_weights)   # T x n
port_returns = np.array(port_returns) # T

# --- step 2: carino K scaling factors ---
R = np.prod(1 + port_returns) - 1          # total arithmetic return
GR = np.log(1 + R)                          # total geometric return
K = GR / R                                  # global K
k_t = np.log(1 + port_returns) / (K * port_returns)  # per-period k

# --- step 3: return attribution ---
# contribution each period = k_t * w_it * r_it, summed over time
r_attr = np.sum(k_t[:, None] * all_weights * returns.values, axis=0)

# total return per asset
total_returns = np.prod(1 + returns.values, axis=0) - 1

# --- step 4: risk attribution (regression method) ---
# regress weighted returns on portfolio return, ra_i = beta_i * sigma_p
port_sigma = np.std(port_returns, ddof=1)
vol_attr = []
for i in range(n):
    y = all_weights[:, i] * returns.values[:, i]
    beta, _, _, _, _ = stats.linregress(port_returns, y)
    vol_attr.append(beta * port_sigma)

vol_attr = np.array(vol_attr)

# --- build output ---
port_total_return = R
port_r_attr = np.sum(r_attr)
port_vol_attr = np.sum(vol_attr)

out = pd.DataFrame({
    'Value': ['TotalReturn', 'Return Attribution', 'Vol Attribution'],
    **{assets[i]: [total_returns[i], r_attr[i], vol_attr[i]] for i in range(n)},
    'Portfolio': [port_total_return, port_r_attr, port_vol_attr]
})

print(out.to_string(index=False))

# --- compare ---
expected = pd.read_csv('data/testout11_1.csv')
numeric_cols = assets + ['Portfolio']
print(f"\nMatches expected: {np.allclose(out[numeric_cols].values, expected[numeric_cols].values, atol=1e-6)}")