# Math & Methodology

## 1. Data: Log Returns

Raw price data is stored as adjusted close prices. All analytics use **log returns**:

```
r_t = ln(P_t / P_{t-1})
```

**Why log returns?**
- Time-additive: multi-period return = sum of daily log returns
- Symmetric: a 10% gain and 10% loss are equal in magnitude
- Normally distributed assumption holds better than simple returns
- Prevents negative prices in simulation (exponential transform)

**Annualisation:** Daily log returns are scaled to annual using 252 trading days:
- Annual mean: `μ = mean(r_t) × 252`
- Annual volatility: `σ = std(r_t) × √252`

---

## 2. Portfolio Statistics

For a portfolio with weight vector **w** (sums to 1, all weights ≥ 0):

**Expected Return:**
```
E[R_p] = wᵀμ
```
where `μ` is the vector of annualised mean returns per asset.

**Portfolio Variance:**
```
σ²_p = wᵀΣw
```
where `Σ` is the annualised covariance matrix. The off-diagonal terms
capture correlation between assets — the source of diversification benefit.

**Portfolio Volatility (standard deviation):**
```
σ_p = √(wᵀΣw)
```

---

## 3. Sharpe Ratio

Measures risk-adjusted return — return earned per unit of volatility:

```
S = (E[R_p] - r_f) / σ_p
```

- `r_f` is the risk-free rate (SA 10-year government bond, ~8.5%)
- S > 1 is considered good; S > 2 is excellent
- Higher Sharpe = better return for the risk taken

---

## 4. Covariance Matrix and Diversification

The covariance matrix `Σ` captures how assets move together:

```
Σᵢⱼ = Cov(rᵢ, rⱼ) = ρᵢⱼ · σᵢ · σⱼ
```

where `ρᵢⱼ` is the Pearson correlation coefficient.

**Diversification:** If assets are perfectly correlated (ρ = 1), portfolio
volatility = weighted average of individual volatilities. As correlation
decreases, portfolio volatility falls *below* the weighted average — this
reduction is the free lunch of diversification.

---

## 5. Efficient Frontier

The efficient frontier is the set of portfolios that maximise return for
a given level of risk (or equivalently, minimise risk for a given return).

**How we find it (Monte Carlo weight simulation):**
1. Sample N random weight vectors from a Dirichlet(1,1,...,1) distribution
   — this gives uniform coverage of the weight simplex (long-only, sums to 1)
2. For each weight vector, compute portfolio return, volatility, and Sharpe ratio
3. Plot all portfolios in return-vs-volatility space
4. The upper-left boundary of the scatter = efficient frontier

**Key portfolios highlighted:**
- **Max Sharpe Ratio:** optimal portfolio for a risk-averse investor with
  access to a risk-free asset (Tobin separation theorem)
- **Min Volatility:** lowest risk portfolio; preferred when return forecasts
  are uncertain

---

## 6. Geometric Brownian Motion (GBM)

The standard continuous-time model for asset prices:

**Stochastic Differential Equation (SDE):**
```
dS = μS dt + σS dW
```
where `dW` is a Wiener process increment (standard Brownian motion).

**Applying Itô's Lemma** to `ln(S)` gives the exact solution:
```
S(t) = S(0) · exp( (μ - σ²/2)t + σW(t) )
```

The `σ²/2` term is the **Itô correction** — it adjusts the drift so that
the *expected* price (not log-price) grows at rate `μ`.

**Discrete simulation (Euler-Maruyama):**
```
S(t+dt) = S(t) · exp( (μ - σ²/2)dt + σ√dt · Z )
where Z ~ N(0,1)
```

All N paths are generated simultaneously as a (n_sims × n_steps) matrix.

---

## 7. Merton Jump-Diffusion Model

GBM produces smooth paths — it cannot model sudden market crashes.
Robert Merton (1976) extended GBM with a **compound Poisson jump process**:

**SDE:**
```
dS = μS dt + σS dW + S(e^J - 1) dN
```

**Components:**
- `σS dW` — continuous diffusion (same as GBM)
- `dN` — Poisson process with intensity `λ` (expected jumps per year)
- `J ~ N(μ_J, σ_J²)` — log jump size (negative μ_J models crashes)

**Jump compensator:** The drift is adjusted by `-λk` to keep the expected
return equal to `μ`, regardless of jump frequency:
```
k = E[e^J - 1] = exp(μ_J + σ_J²/2) - 1
drift = (μ - λk - σ²/2) dt
```

**Discretisation:**
```
S(t+dt) = S(t) · exp( drift + σ√dt·Z + N·J )
where:
  N ~ Poisson(λ·dt)   — number of jumps in interval dt
  J ~ N(μ_J, σ_J²)   — size of each jump
```

Since `λ·dt` is small (e.g. 1 × 1/52 ≈ 0.02), P(N > 1) per step is
negligible, so `N·J` accurately approximates the compound sum.

**Parameters and their meaning:**
| Parameter | Interpretation | Typical value |
|-----------|----------------|---------------|
| λ | Crash frequency (jumps/year) | 1–3 normal, 3–5 stressed |
| μ_J | Average crash severity (log) | -0.10 to -0.20 |
| σ_J | Uncertainty in crash size | 0.03 to 0.08 |

---

## 8. Value at Risk (VaR)

**Definition:** VaR at confidence level α is the loss that will not be
exceeded with probability (1 - α).

```
VaR(α) = -Q_α(PnL)
```

where `Q_α` is the α-quantile of the profit/loss distribution.

**Example:** VaR(5%) = R80,000 means:
- 95% of the time, final loss will be less than R80,000
- 5% of the time, final loss exceeds R80,000

We estimate VaR empirically from the Monte Carlo final-value distribution:
```
VaR(5%) = initial_investment - 5th_percentile(final_values)
```

---

## 9. Monte Carlo Method

Monte Carlo simulation estimates quantities that are analytically
intractable by drawing random samples and aggregating outcomes.

**For portfolio projection:**
1. Simulate N paths of portfolio value over T years
2. Record the final value of each path
3. The distribution of final values gives:
   - Expected outcome (median path)
   - Downside risk (5th percentile, VaR)
   - Upside potential (95th percentile)

**Why weekly steps?**
Daily steps (252/year) would use ~5× more memory with negligible
improvement in path-level accuracy for long-horizon projections.
Weekly steps (52/year) balance accuracy and performance.

**Convergence:** As N → ∞, Monte Carlo estimates converge to the true
distribution. N = 1,000–5,000 gives stable percentile estimates for
portfolio-level analysis.
