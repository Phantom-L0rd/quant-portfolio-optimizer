"""
monte_carlo.py
==============
Portfolio path simulation using two models:

  1. Geometric Brownian Motion (GBM)
     Standard continuous-time model. Good for normal markets.

  2. Merton Jump-Diffusion (JD)
     Extends GBM with a compound Poisson jump process — models sudden
     crashes (negative jumps) caused by market shocks, crises, etc.

Both functions are fully vectorised with NumPy for speed.
Weekly time steps (52/year) used by default — fine for visualisation
and keeps memory manageable on Streamlit Cloud.
"""

import numpy as np
from typing import Tuple

STEPS_PER_YEAR = 52   # weekly steps — good balance of accuracy vs memory


# ── GBM ──────────────────────────────────────────────────────────────────────

def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_sims: int,
    steps_per_year: int = STEPS_PER_YEAR,
    seed: int = None,
) -> np.ndarray:
    """
    Simulate portfolio paths under Geometric Brownian Motion.

    Discretisation (Euler-Maruyama of the log-price SDE):
        S(t+dt) = S(t) · exp( (μ - σ²/2)·dt  +  σ·√dt·Z )
        where Z ~ N(0,1)

    Args:
        S0:             Initial portfolio value (e.g. investment in Rands)
        mu:             Annual drift (expected portfolio return)
        sigma:          Annual volatility
        T:              Investment horizon in years
        n_sims:         Number of simulation paths
        steps_per_year: Time resolution (default: 52 = weekly)
        seed:           Optional random seed

    Returns:
        paths: np.ndarray of shape (n_sims, n_steps+1)
    """
    if seed is not None:
        np.random.seed(seed)

    n_steps = int(T * steps_per_year)
    dt = T / n_steps

    # All random numbers generated at once — fully vectorised
    Z = np.random.standard_normal((n_sims, n_steps))
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    paths = np.zeros((n_sims, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(np.cumsum(log_increments, axis=1))
    return paths


# ── MERTON JUMP-DIFFUSION ─────────────────────────────────────────────────────

def simulate_merton_jd(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_sims: int,
    lambda_j: float = 1.0,
    mu_j: float = -0.10,
    sigma_j: float = 0.05,
    steps_per_year: int = STEPS_PER_YEAR,
    seed: int = None,
) -> np.ndarray:
    """
    Simulate portfolio paths under Merton Jump-Diffusion.

    Model:
        dS = μS dt + σS dW + S(e^J - 1) dN

    Discretisation:
        S(t+dt) = S(t) · exp( drift·dt + σ·√dt·Z + N·J )

    Where:
        drift = (μ - λk - σ²/2)·dt      [compensated for expected jump impact]
        k     = E[e^J - 1] = exp(μ_J + σ_J²/2) - 1
        N     ~ Poisson(λ·dt)             [jump count per step]
        J     ~ N(μ_J, σ_J²)             [log jump size, negative = crash]

    Since λ·dt << 1 (e.g. 1 × 1/52 ≈ 0.02), P(N > 1) per step is negligible,
    so N·J accurately approximates the compound Poisson sum.

    Args:
        lambda_j:  Jump intensity — expected jumps per year (set by UI slider)
        mu_j:      Mean log jump size (e.g. -0.15 = avg 15% crash per event)
        sigma_j:   Std dev of log jump size

    Returns:
        paths: np.ndarray of shape (n_sims, n_steps+1)
    """
    if seed is not None:
        np.random.seed(seed)

    n_steps = int(T * steps_per_year)
    dt = T / n_steps

    # Jump compensation: keeps E[return] = mu regardless of jump parameters
    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    drift = (mu - lambda_j * k - 0.5 * sigma**2) * dt

    # Vectorised: generate all random variables at once
    Z = np.random.standard_normal((n_sims, n_steps))            # diffusion noise
    N = np.random.poisson(lambda_j * dt, (n_sims, n_steps))     # jump counts
    J = np.random.normal(mu_j, sigma_j, (n_sims, n_steps))      # jump sizes

    # N·J ≈ compound Poisson when lambda·dt is small
    log_increments = drift + sigma * np.sqrt(dt) * Z + N * J

    paths = np.zeros((n_sims, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(np.cumsum(log_increments, axis=1))
    return paths


# ── RISK METRICS ─────────────────────────────────────────────────────────────

def compute_percentile_bands(
    paths: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 5th, 50th, 95th percentile bands across simulations.
    Used for the fan chart in Tab 3.
    """
    p05 = np.percentile(paths,  5, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    return p05, p50, p95


def compute_var(
    final_values: np.ndarray,
    initial_value: float,
    confidence: float = 0.05,
) -> float:
    """
    Value at Risk at the given confidence level.

    VaR(5%) = R X  means: 5% probability of losing more than R X over the horizon.
    Returns the loss as a positive number.
    """
    pnl = final_values - initial_value
    return float(-np.percentile(pnl, confidence * 100))


def prob_of_loss(
    final_values: np.ndarray,
    initial_value: float,
    loss_fraction: float,
) -> float:
    """
    P(final value < initial_value × (1 - loss_fraction)).
    E.g. prob_of_loss(vals, 100_000, 0.20) = probability of losing > 20%.
    """
    threshold = initial_value * (1 - loss_fraction)
    return float(np.mean(final_values < threshold))