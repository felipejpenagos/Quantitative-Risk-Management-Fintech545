# FINTECH 545 — Quantitative Risk Management

**Duke University · Pratt School of Engineering · Spring 2026**  
**Instructor:** Prof. Dominic J. Pazzula, Head of Engineering and Risk at DUMAC

---

This repository contains all coursework for FINTECH 545. The course covers the mathematical and computational foundations of modern risk — from statistical modeling and simulation to options pricing, portfolio construction, and performance attribution. All implementations are in **Python**.

---

## Notebooks

| Notebook | Coverage |
|---|---|
| `F545_QRM_Notebook1.ipynb` | Weeks 1–7 · Midterm material |
| `F545_QRM_Notebook2.ipynb` | Weeks 7–13 · Final material |

Both notebooks contain mathematical derivations, commented Python implementations, worked examples, and exam-style practice problems with full solutions.

---

## Assignments (Code folder)

### Statistical Foundations

| Test | Description |
|---|---|
| 6.1 | Calculate arithmetic returns |
| 6.2 | Calculate log returns |
| 7.1 | Fit a Normal Distribution via MLE |
| 7.2 | Fit a T-Distribution via MLE |
| 7.3 | T-Distribution Regression (robust regression with fat-tailed errors) |

### Covariance & Correlation Estimation

| Test | Description |
|---|---|
| 1.1 | Covariance with missing data — skip missing rows |
| 1.2 | Correlation with missing data — skip missing rows |
| 1.3 | Covariance with missing data — pairwise |
| 1.4 | Correlation with missing data — pairwise |
| 2.1 | Exponentially Weighted Covariance (λ = 0.97) |
| 2.2 | Exponentially Weighted Correlation (λ = 0.94) |
| 2.3 | Combined EW Covariance — EW variance (λ = 0.97) + EW correlation (λ = 0.94) |

### PSD Matrix Repair

| Test | Description |
|---|---|
| 3.1 | Near PSD covariance (Rebonato & Jäckel) |
| 3.2 | Near PSD correlation (Rebonato & Jäckel) |
| 3.3 | Nearest PSD covariance (Higham's algorithm) |
| 3.4 | Nearest PSD correlation (Higham's algorithm) |
| 4.1 | Cholesky decomposition of PSD matrix |

### Simulation

| Test | Description |
|---|---|
| 5.1 | Normal simulation — PD input, 0 mean, 100k draws |
| 5.2 | Normal simulation — PSD input, 0 mean, 100k draws |
| 5.3 | Normal simulation — non-PSD input, near_psd fix, 100k draws |
| 5.4 | Normal simulation — non-PSD input, Higham fix, 100k draws |
| 5.5 | PCA simulation — 99% explained variance, 0 mean, 100k draws |

### Risk Measures (VaR & ES)

| Test | Description |
|---|---|
| 8.1 | VaR from Normal Distribution |
| 8.2 | VaR from T-Distribution |
| 8.3 | VaR from Simulation (validated against 8.2) |
| 8.4 | ES from Normal Distribution |
| 8.5 | ES from T-Distribution |
| 8.6 | ES from Simulation (validated against 8.5) |
| 9.1 | VaR & ES at 2 confidence levels from Gaussian Copula simulation |

### Portfolio Optimization

| Test | Description |
|---|---|
| 10.1 | Risk Parity portfolio — normal assumption |
| 10.2 | Risk Parity portfolio — ½ risk budget on X5 |
| 10.3 | Maximum Sharpe Ratio portfolio — normal assumption, w ≥ 0 |
| 10.4 | Maximum Sharpe Ratio portfolio — normal assumption, 0.1 ≤ w ≤ 0.5 |

### Risk & Return Attribution

| Test | Description |
|---|---|
| 11.1 | Ex-post return and risk attribution per stock (Cariño linking) |
| 11.2 | Ex-post return and risk attribution to Fama-French factors |

### Options Pricing

| Test | Description |
|---|---|
| 12.1 | European options via Generalized BSM including all Greeks |
| 12.2 | American options with continuous dividends including Greeks |
| 12.3 | American options with discrete dividends (non-recombining binomial tree) |

---

## Repository Structure

```
.
├── F545_QRM_Notebook1.ipynb     # Master notebook — Weeks 1–7
├── F545_QRM_Notebook2.ipynb     # Master notebook — Weeks 7–13
├── F545_7.1.py  …  F545_12.3.py # Individual assignment scripts
└── data/                         # Input CSVs and expected output files
```

---

## Dependencies

```bash
pip install numpy pandas scipy matplotlib
```

---

**Felipe Jaramillo Penagos** · M.S. Quantitative Management · Duke University  
[github.com/felipejpenagos](https://github.com/felipejpenagos)
