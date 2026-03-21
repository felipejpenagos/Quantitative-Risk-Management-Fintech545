import numpy as np
import pandas as pd

def american_binomial(is_call, S, X, T, r, b, sigma, N=500):
    """
    American option via recombining binomial tree with backward induction.
    b = cost of carry (r - q for continuous dividend yield q).
    Returns option value.
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(b * dt) - d) / (u - d)
    discount = np.exp(-r * dt)
    z = 1 if is_call else -1

    i = np.arange(N + 1)
    values = np.maximum(0.0, z * (S * (u**i) * (d**(N-i)) - X))

    for step in range(N - 1, -1, -1):
        ia = np.arange(step + 1)
        ps = S * (u**ia) * (d**(step - ia))
        hold = discount * (p * values[1:step+2] + (1 - p) * values[0:step+1])
        values = np.maximum(hold, np.maximum(0.0, z * (ps - X)))

    return values[0]


def compute_greeks(is_call, S, X, T, r, b, sigma, N=500):
    v0 = american_binomial(is_call, S, X, T, r, b, sigma, N)

    # delta — central FD on S, b fixed
    dS = 0.001
    delta = (american_binomial(is_call, S+dS, X, T, r, b, sigma, N) -
             american_binomial(is_call, S-dS, X, T, r, b, sigma, N)) / (2 * dS)

    # gamma — second derivative via $1 bump
    vp = american_binomial(is_call, S+1, X, T, r, b, sigma, N)
    vm = american_binomial(is_call, S-1, X, T, r, b, sigma, N)
    gamma = vp - 2*v0 + vm  # dS=1 so dS^2 = 1

    # vega — central FD on sigma
    ds = 0.0001
    vega = (american_binomial(is_call, S, X, T, r, b, sigma+ds, N) -
            american_binomial(is_call, S, X, T, r, b, sigma-ds, N)) / (2 * ds)

    # rho — FD on r only, b held fixed (only discount rate changes, not drift)
    dr = 0.0001
    rho = (american_binomial(is_call, S, X, T, r+dr, b, sigma, N) -
           american_binomial(is_call, S, X, T, r-dr, b, sigma, N)) / (2 * dr)

    # theta — value lost per year as 1 day passes
    theta = (american_binomial(is_call, S, X, T - 1/365, r, b, sigma, N) - v0) / (-1/365)

    return v0, delta, gamma, vega, rho, theta


df = pd.read_csv('data/test12_1.csv').dropna(subset=['ID'])
df['ID'] = df['ID'].astype(int)

results = []
for _, row in df.iterrows():
    is_call = str(row['Option Type']).strip().lower() == 'call'
    S     = float(row['Underlying'])
    X     = float(row['Strike'])
    T     = float(row['DaysToMaturity']) / float(row['DayPerYear'])
    r     = float(row['RiskFreeRate'])
    q     = float(row['DividendRate'])
    sigma = float(row['ImpliedVol'])
    b     = r - q  # Merton 1973: cost of carry = r - continuous dividend yield

    v, d, g, ve, rh, th = compute_greeks(is_call, S, X, T, r, b, sigma)
    results.append({
        'ID': int(row['ID']),
        'Value': v, 'Delta': d, 'Gamma': g,
        'Vega': ve, 'Rho': rh, 'Theta': th
    })

out = pd.DataFrame(results)
print(out.to_string(index=False))