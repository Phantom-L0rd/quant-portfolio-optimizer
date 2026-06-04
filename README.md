# Stochastic Portfolio Optimizer & Crash Stress-Tester

A quantitative financial engineering dashboard built with Python and Streamlit. This application applies computational physics principles—specifically Geometric Brownian Motion and the Merton Jump-Diffusion model—to optimize asset allocation and stress-test portfolios against sudden market crashes.

## Overview
Asset managers and quantitative analysts rely heavily on stochastic modeling to understand risk. This project bridges data engineering, modern portfolio theory, and advanced mathematical analysis to provide a full-stack risk assessment tool. It ingests historical market data, calculates the Efficient Frontier, and runs Monte Carlo simulations to project long-term Value at Risk (VaR).

## Interactive Dashboard Features

* **Portfolio Overview:** Visualizes current asset weights, historical cumulative returns, and summary statistics including mean return, volatility, and individual asset Sharpe ratios.
* **The Efficient Frontier:** An interactive scatter plot of thousands of simulated portfolios. It highlights the Max Sharpe Ratio and Minimum Volatility portfolios, plotting the current allocation directly on the curve.
* **Monte Carlo Projections:** A fan chart displaying 1,000 simulated portfolio paths over a selected multi-year horizon, highlighting the 5th, 50th, and 95th percentile bands alongside a Value at Risk (VaR) distribution histogram.
* **Crash Stress-Testing:** A side-by-side comparative analysis of a normal market projection versus a sudden crash scenario, calculating the exact probability of severe loss using jump-diffusion mechanics.

## The Quantitative Engine

This application relies on a robust mathematical backend to simulate market realities.

### Log Returns & Diversification
Raw adjusted close prices are converted to time-additive, symmetric log returns using $r_t = \ln(P_t / P_{t-1})$. The optimizer calculates portfolio variance using the covariance matrix $\Sigma$, capturing the correlation between assets. Portfolio volatility is defined as:
$$\sigma_p = \sqrt{w^T \Sigma w}$$

### Objective Function
The Scipy optimization engine seeks the weight vector $w$ that maximizes the Sharpe Ratio, utilizing the South African 10-year government bond yield as the baseline risk-free rate ($r_f$):
$$S = \frac{E[R_p] - r_f}{\sigma_p}$$

### Stochastic Differential Equations (SDE)
Future portfolio states are simulated using Geometric Brownian Motion (GBM) for normal market drift and continuous diffusion:
$$dS = \mu S dt + \sigma S dW$$

To simulate black-swan events and market crashes, the engine implements the **Merton Jump-Diffusion Model**, injecting a compound Poisson jump process into the SDE:
$$dS = \mu S dt + \sigma S dW + S(e^J - 1) dN$$
Where $dN$ is a Poisson process dictating crash frequency ($\lambda$) and $J$ dictates average crash severity.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Engineering:** SQL (SQLite), Pandas, yfinance
* **Mathematics & Optimization:** NumPy, SciPy
* **Frontend UI:** Streamlit
* **Visualization:** Plotly / D3.js

## Local Installation & Execution

Clone the repository and install the required dependencies to run the simulation locally.

```bash
# 1. Clone the repository
git clone [https://github.com/yourusername/quant-portfolio-optimizer.git](https://github.com/yourusername/quant-portfolio-optimizer.git)
cd quant-portfolio-optimizer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build the local SQLite database (Fetches historical data)
python scripts/build_db.py

# 4. Launch the Streamlit application
streamlit run app.py
```
