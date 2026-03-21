import numpy as np
import pandas as pd

def american_discrete_div(is_call, S, X, T, r, sigma, div_times, div_amts, N=500):
    b = r
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(b * dt) - d) / (u - d)
    discount = np.exp(-r * dt)
    z = 1 if is_call else -1

    active = sorted([(t, a) for t, a in zip(div_times, div_amts) if 0 < t <= T])

    if not active:
        # standard recombining tree
        i = np.arange(N + 1)
        values = np.maximum(0.0, z * (S * (u**i) * (d**(N-i)) - X))
        for step in range(N - 1, -1, -1):
            ia = np.arange(step + 1)
            ps = S * (u**ia) * (d**(step - ia))
            hold = discount * (p * values[1:step+2] + (1-p) * values[0:step+1])
            values = np.maximum(hold, np.maximum(0.0, z * (ps - X)))
        return values[0]

    # first dividend splits the tree — non-recombining from here
    t_div, amt = active[0]
    remaining  = active[1:]
    n_before   = min(max(1, round(t_div / dt)), N - 1)
    T_after    = T - n_before * dt
    N_after    = N - n_before

    # stock prices at the dividend node
    ia      = np.arange(n_before + 1)
    S_nodes = S * (u**ia) * (d**(n_before - ia))

    # for each node: compare holding (recurse forward) vs exercising now
    node_vals = np.zeros(n_before + 1)
    for i in range(n_before + 1):
        s_pre  = S_nodes[i]
        s_post = max(s_pre - amt, 0.0)   # stock drops by dividend amount
        rem_t  = [t - n_before * dt for t, _ in remaining]
        rem_a  = [a for _, a in remaining]
        v_hold = american_discrete_div(is_call, s_post, X, T_after, r, sigma,
                                       rem_t, rem_a, N_after)
        v_exer = max(0.0, z * (s_pre - X))  # exercise before ex-div
        node_vals[i] = max(v_hold, v_exer)

    # backward induction from dividend date back to t=0
    values = node_vals
    for step in range(n_before - 1, -1, -1):
        ia = np.arange(step + 1)
        ps = S * (u**ia) * (d**(step - ia))
        hold = discount * (p * values[1:step+2] + (1-p) * values[0:step+1])
        values = np.maximum(hold, np.maximum(0.0, z * (ps - X)))

    return values[0]


df = pd.read_csv('data/test12_3.csv').dropna(subset=['ID'])
df['ID'] = df['ID'].astype(int)

results = []
for _, row in df.iterrows():
    is_call   = row['Option Type'].strip().lower() == 'call'
    S         = float(row['Underlying'])
    X         = float(row['Strike'])
    dpy       = float(row['DayPerYear'])
    T         = float(row['DaysToMaturity']) / dpy
    r         = float(row['RiskFreeRate'])
    sigma     = float(row['ImpliedVol'])
    div_days  = [float(x) for x in str(row['DividendDates']).strip('"').split(',')]
    div_amts  = [float(x) for x in str(row['DividendAmts']).strip('"').split(',')]
    div_times = [d / dpy for d in div_days]

    val = american_discrete_div(is_call, S, X, T, r, sigma, div_times, div_amts)
    results.append({'ID': int(row['ID']), 'Value': val})

print(pd.DataFrame(results).to_string(index=False))
