import os
import sqlite3
import pandas as pd

def get_portfolio_returns():
    """
    Extracts historical prices from SQLite, calculates daily returns 
    using an SQL Window function, and returns a pivoted 'wide' DataFrame.
    """

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, 'data', 'portfolio.db')
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}. Please run build_db.py first.")

    conn = sqlite3.connect(db_path)

    sql_query = """
    WITH RankedPrices AS (
        SELECT 
            ticker,
            Date,
            price,
            LAG(price, 1) OVER (PARTITION BY ticker ORDER BY Date) AS prev_price
        FROM historical_prices
    )
    SELECT 
        ticker,
        Date,
        CASE 
            WHEN prev_price IS NOT NULL THEN (price - prev_price) / prev_price
            ELSE NULL 
        END AS daily_return
    FROM RankedPrices
    WHERE prev_price IS NOT NULL;
    """

    print("Executing SQL window function to compute daily returns...")
    df_returns = pd.read_sql_query(sql_query, conn)
    conn.close()

    print("Pivoting data structure into feature matrix...")
    df_pivoted = df_returns.pivot(index='Date', columns='ticker', values='daily_return')
    
    df_pivoted = df_pivoted.dropna()
    
    print(f"Matrix successfully prepared. Shape: {df_pivoted.shape}")
    return df_pivoted

if __name__ == "__main__":
    # Test execution to verify the pipeline works locally
    matrix = get_portfolio_returns()
    print("\nSample Data Matrix (First 5 rows):")
    print(matrix.head())