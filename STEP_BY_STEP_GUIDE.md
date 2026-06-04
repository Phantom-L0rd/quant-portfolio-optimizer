# Step-by-Step Build Guide
## Stochastic Portfolio Optimizer & Market Stress-Tester

---

## Phase 0: Setup (Day 1, ~30 min)

### 0.1 Create project structure
```bash
mkdir quant-portfolio-optimizer
cd quant-portfolio-optimizer
mkdir src data
touch src/__init__.py
```

### 0.2 Copy in all source files
Place these files exactly as given:
```
quant-portfolio-optimizer/
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py
│   ├── analytics.py
│   ├── optimization.py
│   └── monte_carlo.py
├── app.py
├── requirements.txt
├── README.md
└── MATH_AND_METHODOLOGY.md
```

### 0.3 Install dependencies
```bash
pip install -r requirements.txt
```

### 0.4 Verify installs
```bash
python -c "import streamlit, yfinance, pandas, numpy, plotly, scipy; print('All good')"
```
Expected: `All good`

---

## Phase 1: Data Pipeline (Day 1, ~1 hour)

**Goal:** Fetch real price data from Yahoo Finance and store in SQLite using SQL window functions.

### 1.1 Run the pipeline test
```bash
python test_pipeline.py
```

Create `test_pipeline.py`:
```python
from src.data_pipeline import get_connection, create_schema, fetch_and_store, load_returns, load_prices

conn = get_connection()
create_schema(conn)

result = fetch_and_store(conn, ["SPY", "GLD"], "2022-01-01", "2024-01-01")
print("Fetch result:", result)

returns = load_returns(conn, ["SPY", "GLD"], "2022-01-01", "2024-01-01")
print("Returns shape:", returns.shape)
print(returns.head())

prices = load_prices(conn, ["SPY", "GLD"], "2022-01-01", "2024-01-01")
print("Prices shape:", prices.shape)
print(prices.head())
```

### Expected output:
```
Fetch result: {'success': ['SPY', 'GLD'], 'failed': [], 'skipped': []}
Returns shape: (502, 2)
ticker        GLD       SPY
date
2022-01-04  0.0039   0.0008
...
```

### 1.2 Verify SQL window functions work
```bash
python -c "
from src.data_pipeline import get_connection
import sqlite3
conn = get_connection()
conn.execute('SELECT * FROM prices LIMIT 3').fetchall()
print('SQL query works')
"
```

### 1.3 Test caching behaviour
Run `test_pipeline.py` again. The second run should show `skipped` instead of `success` — data is being read from the DB, not re-downloaded.

### 1.4 Common issues
- **`KeyError: ticker`** — yfinance API changed. Update: `yfinance>=0.2.40`
- **Empty returns** — check your date range has trading days (no weekends)
- **JSE tickers fail** — JSE (.JO) data sometimes has gaps. Test with `NPN.JO` first. If it returns empty, the ticker may be delisted or renamed.

---

## Phase 2: Analytics (Day 1, ~1 hour)

**Goal:** Compute portfolio statistics from the returns DataFrame.

### 2.1 Run analytics test
```python
# test_analytics.py
from src.data_pipeline import get_connection, create_schema, fetch_and_store, load_returns
from src.analytics import (
    annualize_returns, annualize_volatility, covariance_matrix,
    correlation_matrix, asset_summary_table
)

conn = get_connection()
create_schema(conn)
fetch_and_store(conn, ["SPY", "QQQ", "GLD"], "2021-01-01", "2024-01-01")
returns = load_returns(conn, ["SPY", "QQQ", "GLD"], "2021-01-01", "2024-01-01")

print("=== Annual Returns ===")
print(annualize_returns(returns))

print("\n=== Annual Volatility ===")
print(annualize_volatility(returns))

print("\n=== Correlation Matrix ===")
print(correlation_matrix(returns).round(3))

print("\n=== Summary Table ===")
print(asset_summary_table(returns))
```

### Expected output:
```
=== Annual Returns ===
ticker
GLD    0.04...
QQQ    0.12...
SPY    0.10...

=== Correlation Matrix ===
ticker   GLD   QQQ   SPY
ticker
GLD     1.000 -0.1x -0.0x
QQQ    -0.1x  1.000  0.9x
SPY    -0.0x  0.9x  1.000
```

### 2.2 Sense-check results
- SPY and QQQ should be highly correlated (>0.85) — both US equity
- GLD should have low/negative correlation with equities — diversifier
- Annual volatility should be 15–25% for equity ETFs

---

## Phase 3: Efficient Frontier (Day 2, ~1.5 hours)

**Goal:** Simulate random portfolios and find the Max Sharpe + Min Vol points.

### 3.1 Run optimization test
```python
# test_optimization.py
import numpy as np
from src.data_pipeline import get_connection, create_schema, fetch_and_store, load_returns
from src.analytics import annualize_returns, covariance_matrix
from src.optimization import (
    generate_random_portfolios, find_max_sharpe, find_min_volatility
)

conn = get_connection()
create_schema(conn)
fetch_and_store(conn, ["SPY", "QQQ", "GLD", "TLT"], "2020-01-01", "2024-01-01")
returns = load_returns(conn, ["SPY", "QQQ", "GLD", "TLT"], "2020-01-01", "2024-01-01")

mean_ret = annualize_returns(returns).values
cov_mat  = covariance_matrix(returns).values

port_df = generate_random_portfolios(mean_ret, cov_mat, n_portfolios=5000)
max_sr  = find_max_sharpe(port_df)
min_vol = find_min_volatility(port_df)

print(f"Max Sharpe:     {max_sr['sharpe']:.3f}")
print(f"  Return:       {max_sr['return']*100:.2f}%")
print(f"  Volatility:   {max_sr['volatility']*100:.2f}%")
print(f"\nMin Volatility: {min_vol['volatility']*100:.2f}%")
print(f"  Return:       {min_vol['return']*100:.2f}%")
```

### Expected output (values vary by date range):
```
Max Sharpe:     1.2xx
  Return:       14.xx%
  Volatility:   11.xx%

Min Volatility: 7.xx%
  Return:       6.xx%
```

### 3.2 Quick visual check (optional)
```python
import matplotlib.pyplot as plt
plt.scatter(port_df["volatility"]*100, port_df["return"]*100,
            c=port_df["sharpe"], cmap="viridis", s=2, alpha=0.5)
plt.colorbar(label="Sharpe")
plt.scatter(max_sr["volatility"]*100, max_sr["return"]*100,
            color="gold", s=100, marker="*", label="Max Sharpe")
plt.xlabel("Volatility (%)"); plt.ylabel("Return (%)"); plt.legend()
plt.savefig("frontier_test.png")
print("Saved frontier_test.png")
```

---

## Phase 4: Monte Carlo (Day 2, ~1 hour)

**Goal:** Simulate portfolio value paths using GBM and Merton Jump-Diffusion.

### 4.1 Test GBM simulation
```python
# test_mc.py
import numpy as np
from src.monte_carlo import (
    simulate_gbm, simulate_merton_jd,
    compute_percentile_bands, compute_var, prob_of_loss
)

S0    = 100_000    # R100,000 initial
mu    = 0.12       # 12% annual return
sigma = 0.18       # 18% annual volatility
T     = 10         # 10 years
n_sims = 1000

# GBM test
gbm = simulate_gbm(S0, mu, sigma, T, n_sims)
p05, p50, p95 = compute_percentile_bands(gbm)
final_vals = gbm[:, -1]

print("=== GBM Results ===")
print(f"Shape:          {gbm.shape}")
print(f"Median (Y10):   R {np.median(final_vals):,.0f}")
print(f"5th pct (Y10):  R {np.percentile(final_vals, 5):,.0f}")
print(f"95th pct (Y10): R {np.percentile(final_vals, 95):,.0f}")
print(f"VaR(5%):        R {compute_var(final_vals, S0):,.0f}")
print(f"P(loss>20%):    {prob_of_loss(final_vals, S0, 0.20)*100:.1f}%")

# Merton JD test
jd = simulate_merton_jd(S0, mu, sigma, T, n_sims, lambda_j=2.0, mu_j=-0.15)
final_jd = jd[:, -1]

print("\n=== Merton JD Results (λ=2, μ_J=-0.15) ===")
print(f"Median (Y10):   R {np.median(final_jd):,.0f}")
print(f"5th pct (Y10):  R {np.percentile(final_jd, 5):,.0f}")
print(f"P(loss>20%):    {prob_of_loss(final_jd, S0, 0.20)*100:.1f}%")
```

### Expected output:
```
=== GBM Results ===
Shape:          (1000, 521)
Median (Y10):   R 3xx,xxx
5th pct (Y10):  R 1xx,xxx
P(loss>20%):    x.x%

=== Merton JD Results ===
Median (Y10):   R 2xx,xxx     ← lower median due to jumps
5th pct (Y10):  R xx,xxx      ← fatter downside tail
P(loss>20%):    xx.x%         ← higher crash probability
```

The JD model should show: lower median, lower 5th percentile, and higher loss probability than GBM. If all three numbers are the same, something is wrong with the jump parameters.

---

## Phase 5: Streamlit App (Day 3, ~2 hours)

**Goal:** Wire all modules into the Streamlit app and verify every tab works.

### 5.1 Launch the app
```bash
streamlit run app.py
```

### 5.2 Tab-by-tab testing checklist

**Sidebar:**
- [ ] Select 3–4 assets including at least 1 JSE and 1 global
- [ ] Set date range: 2021-01-01 to 2024-01-01
- [ ] Investment horizon: 10 years
- [ ] Initial investment: R100,000
- [ ] Simulations: 1000 (fast for testing)
- [ ] Click ▶ Run Analysis

**Tab 1 — Portfolio Overview:**
- [ ] Pie chart renders (equal weight)
- [ ] Cumulative returns line chart shows all selected tickers
- [ ] Stats table shows return, vol, Sharpe per ticker
- [ ] Correlation heatmap renders correctly

**Tab 2 — Efficient Frontier:**
- [ ] Scatter plot shows cloud of portfolios
- [ ] Gold star visible (Max Sharpe)
- [ ] Green diamond visible (Min Volatility)
- [ ] Cyan circle visible (Equal Weight)
- [ ] Weight table shows reasonable weights (no single asset >90%)

**Tab 3 — Monte Carlo:**
- [ ] Fan chart shows spreading paths over time
- [ ] Blue band visible (5th–95th)
- [ ] Orange dashed line at initial investment
- [ ] Histogram shows right-skewed distribution
- [ ] VaR line visible on histogram

**Tab 4 — Stress Test:**
- [ ] Two side-by-side charts render
- [ ] JD chart shows more spread/lower median than GBM chart
- [ ] Risk comparison table shows JD probability > GBM probability
- [ ] Key metric at bottom updates with slider changes

### 5.3 Common issues

**`ModuleNotFoundError: No module named 'src'`**
Run from the project root: `streamlit run app.py`, not from inside `src/`.

**Tab 2 shows only one cluster of dots (no frontier shape)**
Your date range may be too short (<1 year). Use 2+ years of data.

**JD and GBM results identical**
Check `crash_lambda > 0` in sidebar. If λ=0, no jumps occur.

**Memory error with large n_sims + long horizon**
Reduce n_sims to 1000 or horizon to 5 years for testing.

---

## Phase 6: GitHub Cleanup (Day 3, ~30 min)

```bash
# Initialise repo
git init
git add .
git commit -m "feat: initial portfolio optimizer - GBM, Merton JD, Efficient Frontier"

# Create .gitignore
echo "portfolio.db
/tmp/*.db
__pycache__/
*.pyc
.env
data/*.csv" > .gitignore

git add .gitignore
git commit -m "chore: add gitignore"

# Push to GitHub
git remote add origin https://github.com/Phantom-L0rd/quant-portfolio-optimizer.git
git push -u origin main
```

**Write the README before pushing.** Recruiters see the README first.

---

## Phase 7: Deploy to Streamlit Community Cloud (Day 3, ~20 min)

1. Push final code to GitHub (Phase 6 done)
2. Go to https://share.streamlit.io
3. Click **New app**
4. Select your repo → branch: `main` → file: `app.py`
5. Click **Deploy**

**Important:** The app uses `/tmp/portfolio.db` by default (set in `get_connection()`). This works on Streamlit Cloud — the DB is recreated fresh each session from yfinance.

**After deployment:**
- Copy the live URL (e.g. `https://phantom-l0rd-portfolio.streamlit.app`)
- Add it to your GitHub repo description
- Add it to your CV next to the project name
- Add a "Live Demo" badge to your README

---

## Phase 8: CV Update

Once deployed, update the project bullet on your CV:

```
Stochastic Portfolio Optimizer | Python, SQL, SQLite, Streamlit | Live: [URL]
• Engineered a financial data pipeline using SQLite and SQL window functions 
  (LAG, PARTITION BY) to compute historical returns and volatility for 10 assets
  spanning JSE and global ETFs.
• Implemented Merton's Jump-Diffusion model to simulate Monte Carlo projections
  of portfolio value under crash scenarios, computing VaR and crash probability
  metrics across 1,000–5,000 simulated paths.
• Built an interactive Streamlit dashboard featuring an Efficient Frontier
  visualiser, Monte Carlo fan chart, and side-by-side GBM vs Jump-Diffusion
  stress-test comparison.
```

---

## Summary Timeline

| Day | Phase | Output |
|-----|-------|--------|
| 1 AM | Phase 0–1 | Environment set up, data pipeline working |
| 1 PM | Phase 2–3 | Analytics + Efficient Frontier tested |
| 2 AM | Phase 4   | Monte Carlo (GBM + JD) tested |
| 2 PM | Phase 5   | Streamlit app running locally |
| 3    | Phase 6–8 | GitHub pushed, deployed, CV updated |
