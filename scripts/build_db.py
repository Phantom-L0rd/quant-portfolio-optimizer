import os
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def build_database():
    # 1. Define our diversified global asset basket
    # SPY (US Equities), GLD (Gold), BTC-USD (Crypto), TLT (US Bonds), EZA (SA Equities)
    tickers = ["SPY", "GLD", "BTC-USD", "TLT", "EZA"]

    # 2. Set the timeframe (last 10 years)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 10)

    print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")

    # 3. Download the data (Auto-adjust handles stock splits and dividends)
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

    # 4. Data Transformation: Isolate 'Close' prices and unpivot from Wide to Long format
    closing_prices = data['Close'].reset_index()
    df_long = pd.melt(closing_prices, id_vars=['Date'], var_name='ticker', value_name='price')
    
    # Drop rows where markets were closed (NaNs)
    df_long = df_long.dropna()

    # 5. Database Connection: Dynamically locate the /data folder
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, 'data', 'portfolio.db')

    print(f"Connecting to database at: {db_path}")
    conn = sqlite3.connect(db_path)

    # 6. Load data into SQLite
    print("Writing data to the 'historical_prices' table...")
    df_long.to_sql('historical_prices', conn, if_exists='replace', index=False)

    # 7. Verification
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM historical_prices")
    count = cursor.fetchone()[0]
    print(f"Success! {count} rows inserted into the database.")

    conn.close()

if __name__ == "__main__":
    build_database()