"""
data_pipeline.py
================
Fetches price data from Yahoo Finance and persists it in SQLite.
SQL window functions (LAG) compute daily returns directly in the DB.

SQL is used for:
  - Storing raw OHLCV price data with PRIMARY KEY deduplication
  - Computing log/simple returns via LAG() window functions
  - Efficient date-range filtered queries with an index
  - LOG() registered as a Python UDF since SQLite lacks it natively
"""

import sqlite3
import math
import logging
import os
import tempfile
from typing import List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── SQL SCHEMA ────────────────────────────────────────────────────────────────

DDL_ASSETS = """
CREATE TABLE IF NOT EXISTS assets (
    ticker       TEXT PRIMARY KEY,
    name         TEXT,
    asset_class  TEXT,
    currency     TEXT,
    last_updated TEXT
);
"""

DDL_PRICES = """
CREATE TABLE IF NOT EXISTS prices (
    ticker      TEXT NOT NULL,
    date        TEXT NOT NULL,
    open_price  REAL,
    high_price  REAL,
    low_price   REAL,
    close_price REAL,
    adj_close   REAL NOT NULL,
    volume      INTEGER,
    PRIMARY KEY (ticker, date)
);
"""

DDL_INDEX = """
CREATE INDEX IF NOT EXISTS idx_prices_ticker_date
ON prices (ticker, date);
"""

# ── SQL QUERIES ────────────────────────────────────────────────────────────────

# LAG() partitioned by ticker so each ticker's returns are independent.
# LOG() is a registered Python UDF — shows SQL + Python integration.
SQL_RETURNS = """
WITH lagged AS (
    SELECT
        ticker,
        date,
        adj_close,
        LAG(adj_close) OVER (
            PARTITION BY ticker
            ORDER BY date
        ) AS prev_close
    FROM prices
    WHERE ticker IN ({placeholders})
      AND date BETWEEN ? AND ?
)
SELECT
    ticker,
    date,
    adj_close,
    CASE
        WHEN prev_close IS NOT NULL AND prev_close > 0
        THEN (adj_close - prev_close) / prev_close
        ELSE NULL
    END AS simple_return,
    CASE
        WHEN prev_close IS NOT NULL AND prev_close > 0
        THEN LOG(adj_close / prev_close)
        ELSE NULL
    END AS log_return
FROM lagged
WHERE prev_close IS NOT NULL
ORDER BY date, ticker;
"""

SQL_PRICES = """
SELECT ticker, date, adj_close
FROM prices
WHERE ticker IN ({placeholders})
  AND date BETWEEN ? AND ?
ORDER BY date, ticker;
"""

SQL_ROW_COUNT = """
SELECT COUNT(*) AS cnt
FROM prices
WHERE ticker = ? AND date BETWEEN ? AND ?;
"""

# ── ASSET REGISTRY ─────────────────────────────────────────────────────────────
# 10 tickers: 5 JSE + 5 global ETFs. Good balance for frontier visualisation.
# JSE tickers use .JO suffix for yfinance. Returns are in local currency.

ASSET_METADATA = {
    # JSE — ZAR denominated
    "NPN.JO": ("Naspers",         "SA Equity",     "ZAR"),
    "SOL.JO": ("Sasol",           "SA Energy",     "ZAR"),
    "SBK.JO": ("Standard Bank",   "SA Financials", "ZAR"),
    "SHP.JO": ("Shoprite",        "SA Retail",     "ZAR"),
    "MTN.JO": ("MTN Group",       "SA Telecom",    "ZAR"),
    # Global ETFs — USD denominated
    "SPY":    ("S&P 500 ETF",     "Global Equity", "USD"),
    "QQQ":    ("Nasdaq 100 ETF",  "Global Tech",   "USD"),
    "GLD":    ("Gold ETF",        "Commodity",     "USD"),
    "EEM":    ("Emerging Mkts",   "EM Equity",     "USD"),
    "TLT":    ("US 20Y Bond ETF", "Bond",          "USD"),
}

ALL_TICKERS = list(ASSET_METADATA.keys())

# ── CONNECTION ─────────────────────────────────────────────────────────────────

def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Return a SQLite connection with LOG() registered as a Python UDF.

    Args:
        db_path: Path to .db file. Defaults to /tmp/portfolio.db so it works
                 on both local machines and Streamlit Community Cloud.
    """
    if db_path is None:
        db_path = os.path.join(tempfile.gettempdir(), "portfolio.db")

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # Register Python's math.log as SQL LOG() — SQLite has no built-in LOG
    conn.create_function("LOG", 1, math.log)

    # WAL mode: faster concurrent reads
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    """Create tables and index if they don't exist."""
    cur = conn.cursor()
    cur.execute(DDL_ASSETS)
    cur.execute(DDL_PRICES)
    cur.execute(DDL_INDEX)
    conn.commit()


# ── FETCH & STORE ─────────────────────────────────────────────────────────────

def fetch_and_store(
    conn: sqlite3.Connection,
    tickers: List[str],
    start_date: str,
    end_date: str,
    force_refresh: bool = False,
) -> dict:
    """
    Download price data from Yahoo Finance and INSERT OR REPLACE into SQLite.
    Skips tickers already cached for the requested date range.

    Returns:
        {"success": [...], "failed": [...], "skipped": [...]}
    """
    cur = conn.cursor()
    results = {"success": [], "failed": [], "skipped": []}

    # Upsert asset metadata
    for ticker in tickers:
        if ticker in ASSET_METADATA:
            name, asset_class, currency = ASSET_METADATA[ticker]
            cur.execute(
                """INSERT OR REPLACE INTO assets
                   (ticker, name, asset_class, currency, last_updated)
                   VALUES (?, ?, ?, ?, datetime('now'))""",
                (ticker, name, asset_class, currency),
            )

    # Decide which tickers need fetching
    to_fetch = []
    for ticker in tickers:
        if not force_refresh:
            cur.execute(SQL_ROW_COUNT, (ticker, start_date, end_date))
            if cur.fetchone()["cnt"] > 5:
                results["skipped"].append(ticker)
                continue
        to_fetch.append(ticker)

    if not to_fetch:
        conn.commit()
        return results

    # Batch download
    try:
        raw = yf.download(
            to_fetch,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
    except Exception as exc:
        logger.error(f"yfinance download error: {exc}")
        results["failed"].extend(to_fetch)
        conn.commit()
        return results

    # Parse and insert per ticker
    for ticker in to_fetch:
        try:
            df = raw[ticker].copy() if len(to_fetch) > 1 else raw.copy()
            df = df.dropna(subset=["Close"])
            df.index = pd.to_datetime(df.index)

            rows = [
                (
                    ticker,
                    idx.strftime("%Y-%m-%d"),
                    float(row.get("Open",   row["Close"])),
                    float(row.get("High",   row["Close"])),
                    float(row.get("Low",    row["Close"])),
                    float(row["Close"]),
                    float(row["Close"]),            # adj_close = Close (auto_adjust=True)
                    int(row.get("Volume", 0) or 0),
                )
                for idx, row in df.iterrows()
            ]

            cur.executemany(
                """INSERT OR REPLACE INTO prices
                   (ticker, date, open_price, high_price, low_price,
                    close_price, adj_close, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            results["success"].append(ticker)
            logger.info(f"{ticker}: stored {len(rows)} rows")

        except Exception as exc:
            logger.error(f"{ticker} failed: {exc}")
            results["failed"].append(ticker)

    conn.commit()
    return results


# ── RETRIEVAL ─────────────────────────────────────────────────────────────────

def load_returns(
    conn: sqlite3.Connection,
    tickers: List[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Execute the LAG() window-function query and return a wide DataFrame.
    Index = date, columns = tickers, values = log_return.
    """
    ph = ",".join("?" * len(tickers))
    query = SQL_RETURNS.format(placeholders=ph)
    params = tickers + [start_date, end_date]

    df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        raise ValueError(f"No data for {tickers} in [{start_date}, {end_date}]")

    wide = df.pivot(index="date", columns="ticker", values="log_return")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.dropna()
    return wide


def load_prices(
    conn: sqlite3.Connection,
    tickers: List[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Return wide-format adjusted close prices. Index = date, columns = tickers."""
    ph = ",".join("?" * len(tickers))
    query = SQL_PRICES.format(placeholders=ph)
    params = tickers + [start_date, end_date]

    df = pd.read_sql_query(query, conn, params=params)
    wide = df.pivot(index="date", columns="ticker", values="adj_close")
    wide.index = pd.to_datetime(wide.index)
    return wide
