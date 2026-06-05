"""
optimization.py
===============
Efficient Frontier via Monte Carlo weight simulation.
Finds the Max Sharpe Ratio and Min Volatility portfolios.

We use random Dirichlet-sampled weights (long-only, sums to 1) to trace
the frontier rather than scipy.optimize, which makes the scatter plot
intuitive — every dot is a real investable portfolio.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from src.analytics import portfolio_return, portfolio_volatility


def generate_random_portfolios(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    n_portfolios: int = 5000,
    risk_free_rate: float = 0.085,
) -> pd.DataFrame:
    """
    Simulate random weight vectors and compute risk/return for each.

    Args:
        mean_returns:  Annualised mean returns array, shape (n_assets,)
        cov_matrix:    Annualised covariance matrix, shape (n_assets, n_assets)
        n_portfolios:  Number of random portfolios
        risk_free_rate: Used for Sharpe calculation

    Returns:
        DataFrame with columns:
          return, volatility, sharpe, w_0, w_1, ..., w_(n-1)
    """
    n_assets = len(mean_returns)
    out = np.zeros((n_portfolios, 3 + n_assets))

    for i in range(n_portfolios):
        # Dirichlet(1,1,...) samples uniformly from the simplex
        w = np.random.dirichlet(np.ones(n_assets))

        ret = portfolio_return(w, mean_returns)
        vol = portfolio_volatility(w, cov_matrix)
        sr  = (ret - risk_free_rate) / vol if vol > 0 else 0.0

        out[i, 0] = ret
        out[i, 1] = vol
        out[i, 2] = sr
        out[i, 3:] = w

    weight_cols = [f"w_{i}" for i in range(n_assets)]
    return pd.DataFrame(out, columns=["return", "volatility", "sharpe"] + weight_cols)


def find_max_sharpe(portfolios: pd.DataFrame) -> pd.Series:
    """Row with the highest Sharpe ratio."""
    return portfolios.loc[portfolios["sharpe"].idxmax()]


def find_min_volatility(portfolios: pd.DataFrame) -> pd.Series:
    """Row with the lowest volatility."""
    return portfolios.loc[portfolios["volatility"].idxmin()]


def equal_weight_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    tickers: list,
    risk_free_rate: float = 0.085,
) -> Dict:
    """Stats for the naive 1/N equal-weight portfolio."""
    n = len(mean_returns)
    w = np.ones(n) / n
    ret = portfolio_return(w, mean_returns)
    vol = portfolio_volatility(w, cov_matrix)
    sr  = (ret - risk_free_rate) / vol if vol > 0 else 0.0
    return {
        "weights": dict(zip(tickers, w)),
        "return":     ret,
        "volatility": vol,
        "sharpe":     sr,
    }


def extract_weights(portfolio_row: pd.Series, tickers: list) -> Dict[str, float]:
    """Pull weight columns from a portfolio row into a {ticker: weight} dict."""
    return {
        ticker: float(portfolio_row[f"w_{i}"])
        for i, ticker in enumerate(tickers)
    }