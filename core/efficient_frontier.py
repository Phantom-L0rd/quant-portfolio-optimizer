import sys
import os
import numpy as np
import pandas as pd
import scipy.optimize as sco

# Dynamically add the root directory to the path so we can import our script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.query_engine import get_portfolio_returns

def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    """
    Calculates the annualized expected return, volatility, and Sharpe Ratio 
    for a given set of portfolio weights using linear algebra.
    """
    # Expected Return: w^T * mu (dot product of weights and mean returns)
    returns = np.sum(mean_returns * weights) * 252
    
    # Portfolio Volatility: sqrt(w^T * Sigma * w)
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    
    # Sharpe Ratio: (Return - Risk Free Rate) / Volatility
    sharpe_ratio = (returns - risk_free_rate) / std_dev
    
    return returns, std_dev, sharpe_ratio

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    """
    The objective function for the SciPy optimizer. We want to maximize the Sharpe ratio,
    but SciPy only has a 'minimize' function, so we minimize the negative Sharpe ratio.
    """
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def optimize_portfolio():
    """
    The main engine that fetches data, calculates covariance, and runs the optimizer.
    """
    print("Fetching data matrix from SQL engine...")
    returns_df = get_portfolio_returns()
    
    # Calculate the building blocks of Modern Portfolio Theory
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    num_assets = len(returns_df.columns)
    tickers = returns_df.columns.tolist()

    # Constraints and Bounds for the Optimizer
    # 1. Weights must sum to 1 (100% of capital)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # 2. No short selling (bounds for each weight between 0.0 and 1.0)
    bounds = tuple((0.0, 1.0) for asset in range(num_assets))
    
    # 3. Initial Guess (Equal weighting: e.g., 20% in each of the 5 assets)
    initial_guess = num_assets * [1. / num_assets,]

    print("\nRunning SciPy Optimizer to find the Maximum Sharpe Ratio...")
    optimal_result = sco.minimize(
        negative_sharpe_ratio, 
        initial_guess, 
        args=(mean_returns, cov_matrix), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )

    if not optimal_result.success:
        raise Exception("Optimization failed to converge.")

    # Extract the winning weights
    optimal_weights = optimal_result.x
    opt_return, opt_std, opt_sharpe = portfolio_performance(optimal_weights, mean_returns, cov_matrix)

    # Format the results into a clean dictionary
    allocation = {tickers[i]: round(optimal_weights[i] * 100, 2) for i in range(num_assets)}

    print("\n=== OPTIMAL PORTFOLIO ALLOCATION ===")
    for ticker, weight in allocation.items():
        print(f"{ticker}: {weight}%")
    
    print("\n=== EXPECTED PERFORMANCE (Annualized) ===")
    print(f"Expected Return: {round(opt_return * 100, 2)}%")
    print(f"Expected Volatility (Risk): {round(opt_std * 100, 2)}%")
    print(f"Sharpe Ratio: {round(opt_sharpe, 2)}")

    return allocation, opt_return, opt_std

if __name__ == "__main__":
    optimize_portfolio()