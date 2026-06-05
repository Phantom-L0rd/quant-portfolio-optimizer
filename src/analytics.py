"""
analytics.py
============
Portfolio statistics: returns, volatility, Sharpe ratio, correlation matrix.
All annualisation uses 252 trading days.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ── PER-ASSET STATS ───────────────────────────────────────────────────────────

def annualize_returns(log_returns: pd.DataFrame) -> pd.Series:
    """Expected annual return per asset (μ = mean daily log-return × 252)."""
    return log_returns.mean() * TRADING_DAYS


def annualize_volatility(log_returns: pd.DataFrame) -> pd.Series:
    """Annual volatility per asset (σ = daily std × √252)."""
    return log_returns.std() * np.sqrt(TRADING_DAYS)


def covariance_matrix(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Annualised covariance matrix (Σ = daily cov × 252)."""
    return log_returns.cov() * TRADING_DAYS


def correlation_matrix(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of daily log returns."""
    return log_returns.corr()


# ── PORTFOLIO STATS ───────────────────────────────────────────────────────────

def portfolio_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    """E[R_p] = wᵀμ"""
    return float(np.dot(weights, mean_returns))


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """σ_p = √(wᵀΣw)"""
    return float(np.sqrt(weights @ cov_matrix @ weights))


def sharpe_ratio(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.085,
) -> float:
    """S = (E[R_p] - r_f) / σ_p"""
    ret = portfolio_return(weights, mean_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    return (ret - risk_free_rate) / vol if vol > 0 else 0.0


# ── DERIVED SERIES ─────────────────────────────────────────────────────────────

def cumulative_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Normalise prices to 1.0 at start date."""
    return prices / prices.iloc[0]


def rolling_volatility(log_returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Rolling annualised volatility (default: 21-day ≈ 1 month)."""
    return log_returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────

def asset_summary_table(
    log_returns: pd.DataFrame,
    risk_free_rate: float = 0.085,
) -> pd.DataFrame:
    """Single-row-per-asset summary for Tab 1 display."""
    ann_ret = annualize_returns(log_returns)
    ann_vol = annualize_volatility(log_returns)
    sharpe  = (ann_ret - risk_free_rate) / ann_vol

    return pd.DataFrame({
        "Annual Return (%)":    (ann_ret * 100).round(2),
        "Annual Volatility (%)": (ann_vol * 100).round(2),
        "Sharpe Ratio":          sharpe.round(3),
    })