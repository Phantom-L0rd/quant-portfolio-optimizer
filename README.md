# 📈 Stochastic Portfolio Optimizer & Market Stress-Tester

> A quantitative finance tool bridging computational physics and portfolio theory.
> Simulates portfolio performance using Monte Carlo methods, optimises asset allocation
> via the Efficient Frontier, and stress-tests against market crashes using the
> Merton Jump-Diffusion model.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?logo=streamlit)](https://your-app-link.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Live Demo

🚀 **[Launch App →](https://your-app-link.streamlit.app)**

---

## What It Does

Most retail investors pick stocks based on gut feel. This tool replaces that with
quantitative analysis — pulling historical price data for a basket of JSE and global
assets, computing optimal portfolio weights, and simulating thousands of possible futures
to quantify both upside potential and crash risk.

| Feature | Description |
|---|---|
| **Efficient Frontier** | Simulates 5,000 random weight combinations to find the Max-Sharpe and Min-Volatility portfolios |
| **Monte Carlo Projections** | Projects portfolio value over 1–30 years using Geometric Brownian Motion across up to 5,000 paths |
| **Merton Jump-Diffusion** | Extends GBM with a compound Poisson process to model sudden market crashes |
| **Value at Risk (VaR)** | Computes the loss threshold exceeded with 5% probability over the investment horizon |
| **SQL Data Pipeline** | Fetches from Yahoo Finance, stores in SQLite using `LAG()` window functions for return computation |

---

## Screenshots

<!-- Add screenshots after deployment -->
> _Dashboard screenshots coming after deployment_

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Data | `yfinance`, `SQLite`, `pandas` |
| Analytics | `NumPy`, `SciPy` |
| Visualisation | `Plotly`, `Streamlit` |
| Database | SQLite with SQL window functions (`LAG`, `PARTITION BY`) |

---

## Asset Universe

The app covers 10 assets across JSE and global markets:

**JSE (ZAR):** Naspers (NPN.JO), Sasol (SOL.JO), Standard Bank (SBK.JO), Shoprite (SHP.JO), MTN (MTN.JO)

**Global ETFs (USD):** S&P 500 (SPY), Nasdaq 100 (QQQ), Gold (GLD), Emerging Markets (EEM), US 20Y Bonds (TLT)

> **Note on currency:** JSE tickers are ZAR-denominated, global ETFs are USD-denominated.
> The model uses relative log returns, which are comparable across currencies for
> covariance and weight optimisation purposes. Absolute return comparisons should
> account for the ZAR/USD exchange rate.

---

## Project Structure

```
quant-portfolio-optimizer/
├── src/
│   ├── data_pipeline.py    # yfinance fetch → SQLite storage + LAG() queries
│   ├── analytics.py        # portfolio return, volatility, Sharpe ratio
│   ├── optimization.py     # Efficient Frontier via Monte Carlo weight simulation
│   └── monte_carlo.py      # GBM and Merton Jump-Diffusion simulation
├── app.py                  # Streamlit multi-tab dashboard
├── MATH_AND_METHODOLOGY.md # Full mathematical derivations
├── requirements.txt
└── README.md
```

---

## The Models

### Geometric Brownian Motion (GBM)
Standard continuous-time model for asset prices. The log-price follows:

```
S(t+dt) = S(t) · exp( (μ - σ²/2)·dt + σ·√dt·Z ),  Z ~ N(0,1)
```

The `σ²/2` term is the Itô correction, ensuring E[S(t)] grows at rate μ.

### Merton Jump-Diffusion
Extends GBM with a compound Poisson process to model sudden market crashes:

```
dS = μS dt + σS dW + S(e^J - 1) dN
```

Where `N ~ Poisson(λ·dt)` counts crash events and `J ~ N(μ_J, σ_J²)` is the log jump size.
The drift is compensated by `-λk` (where `k = E[e^J - 1]`) to preserve the expected return.

See [MATH_AND_METHODOLOGY.md](MATH_AND_METHODOLOGY.md) for full derivations.

---

## Installation

```bash
git clone https://github.com/Phantom-L0rd/quant-portfolio-optimizer.git
cd quant-portfolio-optimizer
pip install -r requirements.txt
streamlit run app.py
```

---

## How the SQL Pipeline Works

Rather than computing returns purely in Python, this project uses SQLite's `LAG()` window
function to compute daily returns directly in the database:

```sql
WITH lagged AS (
    SELECT ticker, date, adj_close,
           LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) AS prev_close
    FROM prices
    WHERE ticker IN (...)
      AND date BETWEEN ? AND ?
)
SELECT ticker, date,
       LOG(adj_close / prev_close) AS log_return
FROM lagged
WHERE prev_close IS NOT NULL;
```

`LOG()` is registered as a Python UDF since SQLite has no built-in logarithm function,
demonstrating Python ↔ SQL integration.

---

## Limitations & Future Work

- **Currency mixing:** JSE (ZAR) and global (USD) assets are treated as same-currency.
  A production version would apply a ZAR/USD exchange rate adjustment.
- **Long-only constraint:** The optimiser uses Dirichlet-sampled weights (no short selling).
  Future work: add scipy.optimize for exact frontier computation with short constraints.
- **Static parameters:** μ and σ are estimated from historical data and assumed constant.
  Future work: GARCH model for time-varying volatility.
- **No transaction costs:** Real portfolio rebalancing incurs brokerage fees not modelled here.

---

## References

- Merton, R.C. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics*, 3(1–2), 125–144.
- Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77–91.
- Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637–654.

---

*Built by [Arop Kuol](https://github.com/Phantom-L0rd) · BSc Physical Sciences, SMU Pretoria*