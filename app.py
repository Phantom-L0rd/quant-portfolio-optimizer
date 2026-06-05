"""
app.py
======
Stochastic Portfolio Optimizer & Market Stress-Tester
Streamlit front-end — 4 analytical tabs.
"""

import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_pipeline import (
    get_connection, create_schema, fetch_and_store,
    load_returns, load_prices,
    ALL_TICKERS, ASSET_METADATA,
)
from src.analytics import (
    annualize_returns, annualize_volatility, covariance_matrix,
    cumulative_returns, asset_summary_table, correlation_matrix,
    portfolio_return, portfolio_volatility,
)
from src.optimization import (
    generate_random_portfolios, find_max_sharpe,
    find_min_volatility, equal_weight_portfolio, extract_weights,
)
from src.monte_carlo import (
    simulate_gbm, simulate_merton_jd,
    compute_percentile_bands, compute_var, prob_of_loss,
    STEPS_PER_YEAR,
)

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SESSION STATE ──────────────────────────────────────────────────────────────
for _k in ["returns", "prices", "tickers", "ready"]:
    if _k not in st.session_state:
        st.session_state[_k] = None
if "ready" not in st.session_state:
    st.session_state.ready = False


# ── DATABASE (one connection per session) ─────────────────────────────────────
@st.cache_resource
def get_db():
    conn = get_connection()   # uses /tmp/portfolio.db by default
    create_schema(conn)
    return conn


conn = get_db()

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Portfolio Settings")
    st.markdown("---")

    # ── Asset selection ──
    label_map = {t: f"{t}  —  {ASSET_METADATA[t][0]}" for t in ALL_TICKERS}
    selected = st.multiselect(
        "Select Assets (2–10)",
        options=ALL_TICKERS,
        default=["NPN.JO", "SBK.JO", "SHP.JO", "SPY", "GLD", "QQQ"],
        format_func=lambda t: label_map[t],
    )

    st.markdown("---")

    # ── Historical data range ──
    st.subheader("📅 Historical Data Range")
    c1, c2 = st.columns(2)
    start_date = c1.date_input("From", value=date.today() - timedelta(days=3 * 365))
    end_date   = c2.date_input("To",   value=date.today())

    # ── Investment horizon (Monte Carlo forward) ──
    st.subheader("🔭 Investment Horizon")
    horizon_years = st.slider("Years to project", 1, 30, 10)

    st.markdown("---")

    # ── Simulation settings ──
    st.subheader("💰 Simulation Settings")
    initial_investment = st.number_input(
        "Initial Investment (R)",
        min_value=1_000,
        max_value=10_000_000,
        value=100_000,
        step=5_000,
    )
    n_sims = st.slider("Simulations", 500, 5_000, 1_000, step=500)

    st.markdown("---")

    # ── Stress-test controls ──
    st.subheader("⚡ Crash Parameters")
    crash_lambda = st.slider(
        "Crash Intensity λ (jumps/year)",
        0.0, 5.0, 1.0, step=0.5,
        help="Expected number of crash events per year.",
    )
    mu_j = st.slider(
        "Mean Jump Size μ_J (log)",
        -0.50, -0.05, -0.15, step=0.05,
        help="Average log loss per crash. -0.15 ≈ 15% drop per event.",
    )
    loss_pct = st.slider("Loss Threshold for Risk Metric (%)", 5, 50, 20, step=5)

    st.markdown("---")

    risk_free = st.number_input(
        "Risk-Free Rate (%)",
        min_value=0.0, max_value=20.0,
        value=8.5, step=0.5,
        help="SA 10-year government bond yield.",
    ) / 100

    st.markdown("---")

    run = st.button("▶  Run Analysis", type="primary", use_container_width=True)

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.title("📊 Stochastic Portfolio Optimizer & Market Stress-Tester")
st.caption(
    "Efficient Frontier  ·  Monte Carlo (GBM)  ·  Merton Jump-Diffusion  ·  Value at Risk"
)

if len(selected) < 2:
    st.warning("Select at least 2 assets in the sidebar.")
    st.stop()

# ── RUN ANALYSIS ───────────────────────────────────────────────────────────────
if run:
    sd = start_date.strftime("%Y-%m-%d")
    ed = end_date.strftime("%Y-%m-%d")

    with st.spinner("Fetching data from Yahoo Finance → SQLite …"):
        res = fetch_and_store(conn, selected, sd, ed)
        if res["success"]:
            st.sidebar.success(f"Downloaded: {', '.join(res['success'])}")
        if res["skipped"]:
            st.sidebar.info(f"Cached: {', '.join(res['skipped'])}")
        if res["failed"]:
            st.sidebar.error(f"Failed: {', '.join(res['failed'])}")

    with st.spinner("Loading returns from SQL window functions …"):
        try:
            ret_df    = load_returns(conn, selected, sd, ed)
            prices_df = load_prices(conn, selected, sd, ed)
        except ValueError as e:
            st.error(str(e))
            st.stop()

    st.session_state.returns = ret_df
    st.session_state.prices  = prices_df
    st.session_state.tickers = selected
    st.session_state.ready   = True

if not st.session_state.ready:
    st.info("Configure your portfolio in the sidebar and click **▶ Run Analysis**.")
    st.stop()

# ── UNPACK STATE ───────────────────────────────────────────────────────────────
ret_df    = st.session_state.returns
prices_df = st.session_state.prices
tickers   = st.session_state.tickers

mean_ret = annualize_returns(ret_df).values
cov_mat  = covariance_matrix(ret_df).values

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📦 Portfolio Overview",
    "🎯 Efficient Frontier",
    "📈 Monte Carlo Projections",
    "⚡ Stress Test",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — PORTFOLIO OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Portfolio Overview")

    col_l, col_r = st.columns([1, 2])

    with col_l:
        eq_w = np.ones(len(tickers)) / len(tickers)
        fig_pie = px.pie(
            values=eq_w,
            names=tickers,
            title="Asset Allocation (Equal Weight)",
            hole=0.4,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        cum = cumulative_returns(prices_df)
        fig_cum = go.Figure()
        for t in tickers:
            if t in cum.columns:
                fig_cum.add_trace(
                    go.Scatter(x=cum.index, y=cum[t], name=t, mode="lines")
                )
        fig_cum.update_layout(
            title="Cumulative Returns (Normalised to 1.0 at start)",
            xaxis_title="Date",
            yaxis_title="Growth of R1",
            legend=dict(orientation="h", y=-0.25),
            hovermode="x unified",
        )
        st.plotly_chart(fig_cum, use_container_width=True)

    st.subheader("Asset Statistics")
    summary = asset_summary_table(ret_df, risk_free)
    st.dataframe(
        summary.style.format({
            "Annual Return (%)":    "{:.2f}",
            "Annual Volatility (%)": "{:.2f}",
            "Sharpe Ratio":         "{:.3f}",
        }),
        use_container_width=True,
    )

    st.subheader("Correlation Matrix")
    corr = correlation_matrix(ret_df)
    fig_corr = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Return Correlations",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — EFFICIENT FRONTIER
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Efficient Frontier")

    with st.spinner("Simulating 5,000 random portfolios …"):
        port_df   = generate_random_portfolios(mean_ret, cov_mat, 5_000, risk_free)
        max_sr    = find_max_sharpe(port_df)
        min_v     = find_min_volatility(port_df)
        eq_port   = equal_weight_portfolio(mean_ret, cov_mat, tickers, risk_free)

    fig_ef = go.Figure()

    # All simulated portfolios
    fig_ef.add_trace(go.Scatter(
        x=port_df["volatility"] * 100,
        y=port_df["return"] * 100,
        mode="markers",
        marker=dict(
            color=port_df["sharpe"],
            colorscale="Viridis",
            size=3,
            opacity=0.55,
            colorbar=dict(title="Sharpe"),
        ),
        name="Simulated Portfolios",
        hovertemplate="Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>",
    ))

    # Max Sharpe
    fig_ef.add_trace(go.Scatter(
        x=[max_sr["volatility"] * 100],
        y=[max_sr["return"] * 100],
        mode="markers+text",
        marker=dict(color="gold", size=16, symbol="star"),
        text=[f"Max Sharpe ({max_sr['sharpe']:.2f})"],
        textposition="top center",
        name="Max Sharpe",
    ))

    # Min Volatility
    fig_ef.add_trace(go.Scatter(
        x=[min_v["volatility"] * 100],
        y=[min_v["return"] * 100],
        mode="markers+text",
        marker=dict(color="limegreen", size=14, symbol="diamond"),
        text=[f"Min Vol ({min_v['volatility']*100:.1f}%)"],
        textposition="top center",
        name="Min Volatility",
    ))

    # Equal weight
    fig_ef.add_trace(go.Scatter(
        x=[eq_port["volatility"] * 100],
        y=[eq_port["return"] * 100],
        mode="markers+text",
        marker=dict(color="cyan", size=12, symbol="circle"),
        text=["Equal Weight (1/N)"],
        textposition="bottom center",
        name="Equal Weight",
    ))

    fig_ef.update_layout(
        title="Efficient Frontier — Risk vs Return",
        xaxis_title="Annual Volatility (%)",
        yaxis_title="Annual Return (%)",
        height=560,
    )
    st.plotly_chart(fig_ef, use_container_width=True)

    # Key metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Max Sharpe Ratio",  f"{max_sr['sharpe']:.3f}")
    c1.metric("Return at Max SR",  f"{max_sr['return']*100:.2f}%")
    c1.metric("Vol at Max SR",     f"{max_sr['volatility']*100:.2f}%")
    c2.metric("Min Volatility",    f"{min_v['volatility']*100:.2f}%")
    c2.metric("Return at Min Vol", f"{min_v['return']*100:.2f}%")
    c2.metric("Sharpe at Min Vol", f"{min_v['sharpe']:.3f}")
    c3.metric("Equal-Weight Sharpe", f"{eq_port['sharpe']:.3f}")
    c3.metric("Equal-Weight Return", f"{eq_port['return']*100:.2f}%")
    c3.metric("Equal-Weight Vol",    f"{eq_port['volatility']*100:.2f}%")

    # Optimal weight tables
    st.subheader("Optimal Weights")
    w_ms = extract_weights(max_sr, tickers)
    w_mv = extract_weights(min_v, tickers)
    wt_df = pd.DataFrame({
        "Max Sharpe (%)":    [f"{w_ms[t]*100:.1f}" for t in tickers],
        "Min Volatility (%)": [f"{w_mv[t]*100:.1f}" for t in tickers],
    }, index=tickers)
    st.dataframe(wt_df, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — MONTE CARLO PROJECTIONS
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"Monte Carlo Projections — {horizon_years}-Year Horizon")
    st.caption("Using Max-Sharpe portfolio weights · Geometric Brownian Motion")

    ms_weights = np.array([max_sr[f"w_{i}"] for i in range(len(tickers))])
    port_mu    = portfolio_return(ms_weights, mean_ret)
    port_sigma = portfolio_volatility(ms_weights, cov_mat)

    with st.spinner(f"Running {n_sims:,} GBM paths …"):
        gbm_paths = simulate_gbm(
            S0=initial_investment,
            mu=port_mu, sigma=port_sigma,
            T=horizon_years, n_sims=n_sims,
        )

    p05, p50, p95 = compute_percentile_bands(gbm_paths)
    n_steps = gbm_paths.shape[1] - 1
    t_axis  = np.linspace(0, horizon_years, n_steps + 1)

    # Fan chart
    fig_mc = go.Figure()

    # Background sample paths
    sample_n = min(150, n_sims)
    for i in range(sample_n):
        fig_mc.add_trace(go.Scatter(
            x=t_axis, y=gbm_paths[i],
            mode="lines",
            line=dict(color="rgba(100,149,237,0.06)"),
            showlegend=False, hoverinfo="skip",
        ))

    # Percentile band fill
    fig_mc.add_trace(go.Scatter(
        x=np.concatenate([t_axis, t_axis[::-1]]),
        y=np.concatenate([p95, p05[::-1]]),
        fill="toself",
        fillcolor="rgba(100,149,237,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="5th–95th Percentile Band",
    ))

    fig_mc.add_trace(go.Scatter(
        x=t_axis, y=p50, mode="lines",
        line=dict(color="royalblue", width=3),
        name="Median (50th pct)",
    ))
    fig_mc.add_trace(go.Scatter(
        x=t_axis, y=p05, mode="lines",
        line=dict(color="crimson", width=1.5, dash="dash"),
        name="5th Percentile",
    ))
    fig_mc.add_trace(go.Scatter(
        x=t_axis, y=p95, mode="lines",
        line=dict(color="seagreen", width=1.5, dash="dash"),
        name="95th Percentile",
    ))
    fig_mc.add_hline(
        y=initial_investment, line_dash="dot",
        line_color="orange", annotation_text="Initial Investment",
    )

    fig_mc.update_layout(
        title=f"Portfolio Projection — {n_sims:,} Simulations over {horizon_years} Years",
        xaxis_title="Years",
        yaxis_title="Portfolio Value (R)",
        hovermode="x unified", height=500,
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    # Final distribution histogram
    final_vals = gbm_paths[:, -1]
    var_5 = compute_var(final_vals, initial_investment)

    fig_hist = px.histogram(
        x=final_vals, nbins=80,
        title=f"Final Portfolio Value Distribution at Year {horizon_years}",
        labels={"x": "Final Value (R)"},
        color_discrete_sequence=["royalblue"],
    )
    fig_hist.add_vline(
        x=initial_investment, line_dash="dash", line_color="orange",
        annotation_text="Initial Investment",
    )
    fig_hist.add_vline(
        x=initial_investment - var_5, line_dash="dash", line_color="crimson",
        annotation_text=f"5% VaR  −R{var_5:,.0f}",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median Final Value",  f"R {np.median(final_vals):,.0f}")
    c2.metric("5th Percentile",      f"R {np.percentile(final_vals,5):,.0f}")
    c3.metric("95th Percentile",     f"R {np.percentile(final_vals,95):,.0f}")
    c4.metric("Value at Risk (5%)",  f"R {var_5:,.0f}",
              delta=f"−{var_5/initial_investment*100:.1f}%", delta_color="inverse")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — STRESS TEST
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Stress Test — Merton Jump-Diffusion vs Standard GBM")
    st.caption(
        f"λ = {crash_lambda} jumps/year  ·  μ_J = {mu_j}  ·  σ_J = 0.05  |  "
        f"Loss threshold: {loss_pct}%"
    )

    with st.spinner("Running Jump-Diffusion simulation …"):
        jd_paths = simulate_merton_jd(
            S0=initial_investment,
            mu=port_mu, sigma=port_sigma,
            T=horizon_years, n_sims=n_sims,
            lambda_j=crash_lambda,
            mu_j=mu_j,
            sigma_j=0.05,
        )

    p05_jd, p50_jd, p95_jd = compute_percentile_bands(jd_paths)
    final_jd = jd_paths[:, -1]
    var_jd   = compute_var(final_jd, initial_investment)

    # Side-by-side charts
    c_gbm, c_jd = st.columns(2)

    def percentile_chart(p05_, p50_, p95_, t_axis_, title_):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.concatenate([t_axis_, t_axis_[::-1]]),
            y=np.concatenate([p95_, p05_[::-1]]),
            fill="toself", fillcolor="rgba(100,149,237,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="5–95th Band",
        ))
        fig.add_trace(go.Scatter(x=t_axis_, y=p50_, mode="lines",
                                  line=dict(width=3), name="Median"))
        fig.add_trace(go.Scatter(x=t_axis_, y=p05_, mode="lines",
                                  line=dict(color="crimson", dash="dash"), name="5th pct"))
        fig.add_trace(go.Scatter(x=t_axis_, y=p95_, mode="lines",
                                  line=dict(color="seagreen", dash="dash"), name="95th pct"))
        fig.add_hline(y=initial_investment, line_dash="dot", line_color="orange")
        fig.update_layout(title=title_, xaxis_title="Years",
                          yaxis_title="Value (R)", height=420,
                          showlegend=False)
        return fig

    c_gbm.plotly_chart(
        percentile_chart(p05, p50, p95, t_axis, "Standard GBM (No Crashes)"),
        use_container_width=True,
    )
    c_jd.plotly_chart(
        percentile_chart(p05_jd, p50_jd, p95_jd, t_axis,
                         f"Merton JD (λ={crash_lambda}, μ_J={mu_j})"),
        use_container_width=True,
    )

    # Risk comparison table
    st.subheader("Risk Metrics: GBM vs Jump-Diffusion")

    p_gbm = prob_of_loss(final_vals, initial_investment, loss_pct / 100)
    p_jd  = prob_of_loss(final_jd,  initial_investment, loss_pct / 100)

    comp = pd.DataFrame({
        "Metric": [
            f"P(loss > {loss_pct}%)",
            "Median Final Value",
            "5th Percentile Final",
            "VaR (5%)",
        ],
        "Standard GBM": [
            f"{p_gbm*100:.1f}%",
            f"R {np.median(final_vals):,.0f}",
            f"R {np.percentile(final_vals,5):,.0f}",
            f"R {var_5:,.0f}",
        ],
        "Merton Jump-Diffusion": [
            f"{p_jd*100:.1f}%",
            f"R {np.median(final_jd):,.0f}",
            f"R {np.percentile(final_jd,5):,.0f}",
            f"R {var_jd:,.0f}",
        ],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)

    st.metric(
        label=f"Probability of Losing >{loss_pct}% — Crash Scenario",
        value=f"{p_jd*100:.1f}%",
        delta=f"+{(p_jd - p_gbm)*100:.1f}% vs standard model",
        delta_color="inverse",
    )