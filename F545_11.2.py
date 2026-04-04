import numpy as np
import pandas as pd
from scipy import stats

factor_returns = pd.read_csv('data/test11_2_factor_returns.csv')   # T x m
stock_returns  = pd.read_csv('data/test11_2_stock_returns.csv')    # T x n
beta_df        = pd.read_csv('data/test11_2_beta.csv', index_col=0) # n x m
weights        = pd.read_csv('data/test11_2_weights.csv')['W'].values

factors = factor_returns.columns.tolist()
T = len(factor_returns)

# --- step 1: compute factor weights and alpha each period ---
# factor_weight_j = sum_i(w_i * beta_i_j)  -- this drifts with weights
w = weights.copy()
port_returns = []
all_factor_weights = []
all_alpha = []

for t in range(T):
    r_s = stock_returns.iloc[t].values   # stock returns this period
    r_f = factor_returns.iloc[t].values  # factor returns this period

    # factor weights = w' @ beta  (shape: m)
    fw = w @ beta_df.values
    all_factor_weights.append(fw.copy())

    # portfolio return
    w_star = w * (1 + r_s)
    port_ret = np.sum(w_star) - 1
    port_returns.append(port_ret)

    # alpha = port_ret - factor_weights . factor_returns
    alpha_t = port_ret - fw @ r_f
    all_alpha.append(alpha_t)

    # update weights
    w = w_star / (1 + port_ret)

all_factor_weights = np.array(all_factor_weights)  # T x m
port_returns = np.array(port_returns)               # T
all_alpha = np.array(all_alpha)                     # T

# --- step 2: carino K ---
R = np.prod(1 + port_returns) - 1
GR = np.log(1 + R)
K = GR / R
k_t = np.log(1 + port_returns) / (K * port_returns)

# --- step 3: return attribution per factor ---
# contribution of factor j = sum_t(k_t * fw_jt * F_jt)
r_attr_factors = np.sum(k_t[:, None] * all_factor_weights * factor_returns.values, axis=0)
r_attr_alpha   = np.sum(k_t * all_alpha)

# total returns per factor (geometric)
total_factor_returns = np.prod(1 + factor_returns.values, axis=0) - 1
total_alpha = np.prod(1 + all_alpha) - 1

# --- step 4: risk attribution (regression) ---
port_sigma = np.std(port_returns, ddof=1)

# stack factor contributions + alpha as "assets" for regression
# weighted factor series: fw_jt * F_jt
factor_series = all_factor_weights * factor_returns.values  # T x m
alpha_series  = all_alpha.reshape(-1, 1)                    # T x 1
all_series    = np.hstack([factor_series, alpha_series])    # T x (m+1)

vol_attr = []
for i in range(all_series.shape[1]):
    beta, _, _, _, _ = stats.linregress(port_returns, all_series[:, i])
    vol_attr.append(beta * port_sigma)

vol_attr = np.array(vol_attr)

# --- build output ---
col_names = factors + ['Alpha']
out = pd.DataFrame({
    'Value': ['TotalReturn', 'Return Attribution', 'Vol Attribution'],
    **{col_names[j]: [
        total_factor_returns[j] if j < len(factors) else total_alpha,
        r_attr_factors[j] if j < len(factors) else r_attr_alpha,
        vol_attr[j]
    ] for j in range(len(col_names))},
    'Portfolio': [R, np.sum(r_attr_factors) + r_attr_alpha, np.sum(vol_attr)]
})

print(out.to_string(index=False))

# --- compare ---
expected = pd.read_csv('data/testout11_2.csv')
numeric_cols = col_names + ['Portfolio']
print(f"\nMatches expected: {np.allclose(out[numeric_cols].values, expected[numeric_cols].values, atol=1e-6)}")