import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize

# Load data
data = pd.read_csv('data/test7_3.csv')
X = data[['x1', 'x2', 'x3']].values
y = data['y'].values

def neg_log_likelihood(params):
    sigma = params[0]
    nu = params[1]
    alpha = params[2]
    beta = params[3:6]
    
    # Linear prediction with centered errors (mu=0)
    y_pred = alpha + X @ beta
    residuals = y - y_pred
    
    if sigma <= 0 or nu <= 0:
        return 1e10
    
    log_lik = np.sum(stats.t.logpdf(residuals / sigma, df=nu) - np.log(sigma))
    return -log_lik

# Initialize with OLS
X_with_intercept = np.column_stack([np.ones(len(X)), X])
beta_init = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
residuals_init = y - X_with_intercept @ beta_init

initial_params = [np.std(residuals_init), 5.0] + beta_init.tolist()

result = minimize(neg_log_likelihood, initial_params, method='Nelder-Mead', 
                  options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})

sigma, nu, alpha, b1, b2, b3 = result.x
mu = 0.0  # Fixed at 0

print(f"mu\tsigma\tnu\tAlpha\tB1\tB2\tB3")
print(f"{mu}\t{sigma}\t{nu}\t{alpha}\t{b1}\t{b2}\t{b3}")