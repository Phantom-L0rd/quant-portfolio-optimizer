# Phase 1: The Data Flow (SQL to Pandas)

**1. The SQL Extraction (`scripts/query_engine.py`)**
Your database currently stores prices in a "long" format. To do matrix math, we need to calculate the daily returns and pivot this into a "wide" format (dates as rows, tickers as columns).

* **The Math:** Daily return $R_{i,t}$ for asset $i$ at time $t$ is calculated as:

$$R_{i,t} = \frac{P_{i,t} - P_{i,t-1}}{P_{i,t-1}}$$


* **The Code Flow:** You will use Python's `sqlite3` to execute a query using the `LAG()` window function to calculate $R_{i,t}$. You will then load this directly into a pandas DataFrame and use `df.pivot(index='Date', columns='ticker', values='daily_return')` to create your feature matrix. Drop the `NaN` values from the first row.

---

# Phase 2: The Optimization Engine (Modern Portfolio Theory)

**1. Calculating Expected Returns and Risk (`core/efficient_frontier.py`)**
Once you have your matrix of daily returns, you need to calculate the historical drift (mean return) and the volatility (variance).

* **Annualized Mean Return ($\mu$):** We assume 252 trading days in a year.

$$\mu_i = \bar{R}_i \times 252$$



*In Python:* `mu = returns_df.mean() * 252`
* **The Covariance Matrix ($\Sigma$):** This represents how the assets move in relation to one another. This is crucial for diversification; you want assets that do not perfectly correlate.

$$\Sigma_{i,j} = \text{Cov}(R_i, R_j) \times 252$$



*In Python:* `cov_matrix = returns_df.cov() * 252`

**2. Portfolio Math**
For a portfolio with weights $w$ (where $\sum w_i = 1$), the expected return $E(R_p)$ and portfolio variance $\sigma_p^2$ are calculated using linear algebra:

* **Expected Return:** $E(R_p) = w^T \mu$
* **Portfolio Risk (Volatility):** $\sigma_p = \sqrt{w^T \Sigma w}$

**3. The Objective Function (Sharpe Ratio)**
The goal of the optimizer is to find the exact weights ($w$) that maximize the Sharpe Ratio (return per unit of risk). Assuming a risk-free rate ($R_f$) of 0 for simplicity:

* **Sharpe Ratio:** $SR = \frac{E(R_p)}{\sigma_p}$
* **The Code Flow:** You will use `scipy.optimize.minimize`. Because Scipy only *minimizes*, you will instruct it to minimize the *negative* Sharpe Ratio ($-SR$). You will pass constraints: the sum of weights must equal $1$, and bounds must be between $0$ and $1$ (no short-selling).

---

# Phase 3: The Stochastic Engine (Crash Simulator)

Now that the optimizer has given you the optimal weights, you know your portfolio's overall $\mu_p$ (drift) and $\sigma_p$ (volatility). You will plug these into **Merton's Jump-Diffusion Model** to project the portfolio's value over time.

**1. The Physics: Geometric Brownian Motion (GBM)**
Normal market movement is modeled like a particle in a fluid, experiencing continuous drift and random shocks (Brownian motion).

* **The Continuous Equation:**

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$



Where $dW_t$ is a Wiener process (standard Brownian motion).

**2. Adding the Crash (Poisson Jump Process)**
We add a discrete jump component to simulate sudden market crashes that normal GBM fails to capture.

* **The Full Equation:**

$$dS_t = S_t (\mu dt + \sigma dW_t + (J-1) dN_t)$$



Where $dN_t$ is a Poisson process (dictating *when* a crash happens) and $J$ represents the magnitude of the jump.

**3. The Simulation Code (`core/jump_diffusion.py`)**
To code this, we discretize the equation using the Euler-Maruyama method. This is what you will write inside a `for` loop to generate thousands of Monte Carlo paths.

For a time step $\Delta t$ (e.g., 1 day = $1/252$), the value of the portfolio at the next step is:


$$S_{t+\Delta t} = S_t \exp\left( \left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma \sqrt{\Delta t} Z + \ln(J) Y \right)$$

* **The Variables in Python:**
* $Z$: A random draw from a standard normal distribution, `np.random.standard_normal()`.
* $Y$: A binary trigger. For your Streamlit app, you can let the user trigger the crash manually. If the "Crash" button is unclicked, $Y=0$. If clicked at time $t$, $Y=1$.
* $J$: The crash magnitude. If the user simulates a 30% crash, $J = 0.70$.



By running this discretized equation inside a nested loop (1,000 paths $\times$ 252 days $\times$ 5 years), you generate the data matrix required to plot the beautiful projection charts in Streamlit.



