import numpy as np
import pandas as pd
from scipy.stats import norm

def gbsm(is_call, S, X, T, r, b, sigma):
    d1 = (np.log(S/X) + (b + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    phi  = norm.pdf(d1)
    N1, N2   = norm.cdf(d1),  norm.cdf(d2)
    Nn1, Nn2 = norm.cdf(-d1), norm.cdf(-d2)

    if is_call:
        value = S*np.exp((b-r)*T)*N1 - X*np.exp(-r*T)*N2
        delta = np.exp((b-r)*T)*N1
        rho   = T*X*np.exp(-r*T)*N2
        theta = (-S*np.exp((b-r)*T)*phi*sigma/(2*np.sqrt(T))
                 - (b-r)*S*np.exp((b-r)*T)*N1
                 - r*X*np.exp(-r*T)*N2)
    else:
        value = X*np.exp(-r*T)*Nn2 - S*np.exp((b-r)*T)*Nn1
        delta = np.exp((b-r)*T)*(N1 - 1)
        rho   = -T*X*np.exp(-r*T)*Nn2
        theta = (-S*np.exp((b-r)*T)*phi*sigma/(2*np.sqrt(T))
                 + (b-r)*S*np.exp((b-r)*T)*Nn1
                 + r*X*np.exp(-r*T)*Nn2)

    gamma = phi*np.exp((b-r)*T) / (S*sigma*np.sqrt(T))
    vega  = S*np.exp((b-r)*T)*phi*np.sqrt(T)

    return value, delta, gamma, vega, rho, theta

df = pd.read_csv('data/test12_1.csv').dropna(subset=['ID'])

rows = []
for _, row in df.iterrows():
    is_call = row['Option Type'].strip().lower() == 'call'
    S, X  = float(row['Underlying']), float(row['Strike'])
    T     = float(row['DaysToMaturity']) / float(row['DayPerYear'])
    r, q  = float(row['RiskFreeRate']), float(row['DividendRate'])
    sigma = float(row['ImpliedVol'])
    b     = r - q  # cost of carry: b=r (no div), b=r-q (continuous div)

    v, d, g, ve, rh, th = gbsm(is_call, S, X, T, r, b, sigma)
    rows.append({'ID': int(row['ID']), 'Value': v, 'Delta': d, 'Gamma': g,
                 'Vega': ve, 'Rho': rh, 'Theta': th})

out = pd.DataFrame(rows)
print(out.to_string(index=False))