import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

np.random.seed(0)

# load data
portfolio = pd.read_csv('data/test9_1_portfolio.csv')
returns = pd.read_csv('data/test9_1_returns.csv')

# fit distributions to each stock
# Stock A: Normal
params_A = stats.norm.fit(returns['A'])
mu_A, sigma_A = params_A

# Stock B: T-distribution
params_B = stats.t.fit(returns['B'])
nu_B, mu_B, scale_B = params_B

# Step 1: Transform to uniform using CDFs (this is the U matrix from lecture)
U_A = stats.norm.cdf(returns['A'], mu_A, sigma_A)
U_B = stats.t.cdf(returns['B'], nu_B, mu_B, scale_B)

# Step 2: Calculate Spearman correlation directly on U (lecture says we can skip Z transform)
U = np.column_stack([U_A, U_B])
rho_spearman = spearmanr(U[:, 0], U[:, 1])[0]
corr = np.array([[1.0, rho_spearman], [rho_spearman, 1.0]])

# Step 3: Simulate from Gaussian copula
n_sim = 100000
L = np.linalg.cholesky(corr)
Z_sim = (L @ np.random.normal(0, 1, (2, n_sim))).T

# Step 4: Transform Z to uniform using standard normal CDF
U_sim = stats.norm.cdf(Z_sim)

# Step 5: Transform uniform back to original distributions using quantile functions
returns_sim_A = stats.norm.ppf(U_sim[:, 0], mu_A, sigma_A)
returns_sim_B = stats.t.ppf(U_sim[:, 1], nu_B, mu_B, scale_B)

# Calculate P&L
pnl_A = 100 * 20 * returns_sim_A
pnl_B = 100 * 30 * returns_sim_B
pnl_total = pnl_A + pnl_B

# VaR and ES
alpha = 0.05

def calc_var_es(pnl):
    var = -np.percentile(pnl, alpha * 100)
    es = -pnl[pnl <= -var].mean()
    return var, es

var_A, es_A = calc_var_es(pnl_A)
var_B, es_B = calc_var_es(pnl_B)
var_total, es_total = calc_var_es(pnl_total)

# percentages
var_A_pct = var_A / 2000
es_A_pct = es_A / 2000
var_B_pct = var_B / 3000
es_B_pct = es_B / 3000
var_total_pct = var_total / 5000
es_total_pct = es_total / 5000

result = pd.DataFrame({
    'Stock': ['A', 'B', 'Total'],
    'VaR95': [var_A, var_B, var_total],
    'ES95': [es_A, es_B, es_total],
    'VaR95_Pct': [var_A_pct, var_B_pct, var_total_pct],
    'ES95_Pct': [es_A_pct, es_B_pct, es_total_pct]
})

expected = pd.read_csv('data/testout9_1.csv')
print(np.allclose(result.iloc[:, 1:].values, expected.iloc[:, 1:].values, atol=1))