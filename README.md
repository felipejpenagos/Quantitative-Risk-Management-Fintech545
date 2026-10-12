<div align="center">

# Quantitative Risk Management

### FINTECH 545 · Duke University · Spring 2026

[![Duke](https://img.shields.io/badge/Duke-FINTECH%20545-00539B?style=flat-square)](#about)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](#structure)
[![Related paper](https://img.shields.io/badge/Related%20paper-MVO%20vs.%20Risk%20Parity-B31B1B?style=flat-square)](https://github.com/felipejpenagos/Portfolio-Optimization-Theory)

</div>

---

Python implementations of the core toolkit of quantitative risk management: covariance repair, simulation, value at risk and expected shortfall, portfolio construction, attribution, and option pricing. Each module is checked against reference outputs in `data/`.

The course's final paper, comparing mean-variance optimization with risk parity, lives in its own repository: **[Portfolio-Optimization-Theory](https://github.com/felipejpenagos/Portfolio-Optimization-Theory)**.

## Notebooks

| Notebook | Coverage |
|---|---|
| `F545_QRM_Notebook1.ipynb` | Weeks 1–7: derivations, implementations, and worked problems |
| `F545_QRM_Notebook2.ipynb` | Weeks 7–13: derivations, implementations, and worked problems |

## Modules

**Covariance estimation**

| Test | Method |
|---|---|
| 1.1 · 1.2 | Covariance and correlation, skipping rows with missing data |
| 1.3 · 1.4 | Pairwise covariance and correlation using all available data |
| 2.1 · 2.2 | Exponentially weighted covariance and correlation |
| 2.3 | Hybrid: EW variance (λ = 0.97) with EW correlation (λ = 0.94) |

**Covariance repair**

| Test | Method |
|---|---|
| 3.1 | Near-PSD covariance (Rebonato and Jäckel) |
| 3.2 | Near-PSD correlation (Rebonato and Jäckel) |
| 3.3 | Nearest PSD covariance (Higham) |
| 3.4 | Nearest PSD correlation (Higham) |
| 4.1 | Cholesky factorization of a PSD matrix |

**Simulation**

| Test | Method |
|---|---|
| 5.1 | Multivariate normal, positive-definite input, 100k draws |
| 5.2 | Multivariate normal, PSD input, 100k draws |
| 5.3 | Multivariate normal, non-PSD input repaired with near-PSD |
| 5.4 | Multivariate normal, non-PSD input repaired with Higham |
| 5.5 | PCA simulation at 99% explained variance |

**Returns and distribution fitting**

| Test | Method |
|---|---|
| 6.1 · 6.2 | Arithmetic and log returns from prices |
| 7.1 · 7.2 | Normal and t-distribution fits by maximum likelihood |
| 7.3 | Regression with t-distributed errors, initialized from OLS |

**Value at risk and expected shortfall**

| Test | Method |
|---|---|
| 8.1 · 8.4 | VaR and ES under a normal distribution |
| 8.2 · 8.5 | VaR and ES under a fitted t-distribution |
| 8.3 · 8.6 | VaR and ES from simulation, validated against the t results |
| 9.1 | Portfolio VaR and ES at two confidence levels via Gaussian copula |

**Portfolio construction**

| Test | Method |
|---|---|
| 10.1 | Risk parity under normality |
| 10.2 | Risk budgeting with half the budget on one asset |
| 10.3 | Maximum Sharpe ratio, long-only |
| 10.4 | Maximum Sharpe ratio with weight bounds 0.1 ≤ w ≤ 0.5 |

**Attribution**

| Test | Method |
|---|---|
| 11.1 | Ex-post return and risk attribution by asset (Cariño linking) |
| 11.2 | Ex-post return and risk attribution to Fama-French factors |

**Option pricing**

| Test | Method |
|---|---|
| 12.1 | European options under generalized Black-Scholes-Merton, with all Greeks |
| 12.2 | American options with continuous dividends, with Greeks |
| 12.3 | American options with discrete dividends (non-recombining binomial tree) |

## Structure

```
.
├── F545_QRM_Notebook1.ipynb    weeks 1–7
├── F545_QRM_Notebook2.ipynb    weeks 7–13
├── Code/                       one script per module (F545_1.1.py … F545_12.3.py)
└── data/                       inputs and reference outputs
```

## Run

```bash
pip install numpy pandas scipy statsmodels matplotlib
python Code/F545_8.2.py
```

## About

Coursework for FINTECH 545, Quantitative Risk Management, taught by Prof. Dominic J. Pazzula (Head of Engineering and Risk, DUMAC) at Duke University's Pratt School of Engineering.
