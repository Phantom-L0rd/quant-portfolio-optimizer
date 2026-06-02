import os
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def build_database():
    tickers = ["SPY", "GLD", "BTC-USD", "TLT", "EZA"]


    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 10)

    print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")

    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

    closing_prices = data['Close'].reset_index()
    df_long = pd.melt(closing_prices, id_vars=['Date'], var_name='ticker', value_name='price')

    df_long = df_long.dropna()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, 'data', 'portfolio.db')

    print(f"Connecting to database at: {db_path}")
    conn = sqlite3.connect(db_path)

    print("Writing data to the 'historical_prices' table...")
    df_long.to_sql('historical_prices', conn, if_exists='replace', index=False)

    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM historical_prices")
    count = cursor.fetchone()[0]
    print(f"Success! {count} rows inserted into the database.")

    conn.close()

if __name__ == "__main__":
    build_database()