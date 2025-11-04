# app.py
import os
import numpy as np
import pandas as pd
import streamlit as st

from dataloader import DataLoader
from features import DataFeatures
from rl_model import RLModel

st.set_page_config(page_title="Bank Volatility RL", layout="wide")

# Shared constants
BANK_TICKERS = ["AXISBANK.NS","BANKBARODA.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","^NSEBANK"]
TICKER_SHORT = {
    "AXISBANK.NS":"AXISBANK", "BANKBARODA.NS":"BANKBARODA", "HDFCBANK.NS":"HDFCBANK",
    "ICICIBANK.NS":"ICICIBANK", "SBIN.NS":"SBIN", "^NSEBANK":"NSEBANK"
}

# Folders
CACHE_DIR = "data_cache-10"
FEATURE_DIR = "rl_vol_features"
STRAT_A_DIR = "strategy-A-model"
STRAT_B_DIR = "strategy-B-model"  # reserved for future

# Instantiate utilities
loader = DataLoader(tickers=BANK_TICKERS, cache_dir=CACHE_DIR, options_file="bank_options.csv")
featurer = DataFeatures(out_dir=FEATURE_DIR)
rl = RLModel(feature_dir=FEATURE_DIR, out_root=STRAT_A_DIR)

# Initialize session flags
if "trained_strategyA" not in st.session_state:
    st.session_state["trained_strategyA"] = False
if "strategyA_summary" not in st.session_state:
    st.session_state["strategyA_summary"] = None

tab1, tab2 = st.tabs(["🛢 Data Pipeline", "👾 RL Model"])

# -------------------------- TAB 1: Data Pipeline --------------------------
with tab1:
    st.header("Data Pipeline")

    # ---- Source data section ----
    st.subheader("Source Data")
    days = st.number_input("Duration (days)", min_value=30, max_value=36500, value=3650, step=30)
    colA, colB = st.columns([1,1])
    with colA:
        get_data = st.button("Get Data", type="primary")
    with colB:
        show_source = st.button("Show Source (plots)")

    if get_data:
        loader.days = int(days)
        prog = st.progress(0)
        st.info("Checking cache and downloading (if needed)…")
        prices = loader.fetch_or_load(verbose=False)
        prog.progress(40)
        st.info("Computing IV, HV, and GARCH…")
        priced = loader.compute_vol_metrics(prices, verbose=False)
        prog.progress(100)
        st.success("Source data ready. You can now view plots or move to Feature Engineering.")
        # store in session for 'Show Source'
        st.session_state["priced"] = {k: v.to_json(date_format="iso", orient="table") for k,v in priced.items()}

    if show_source:
        priced_json = st.session_state.get("priced")
        if not priced_json:
            st.warning("No in-memory data. Click 'Get Data' first (or plots will be minimal).")
        else:
            for t, js in priced_json.items():
                df = pd.read_json(js, orient="table")
                st.write(f"**{t}** — Close")
                st.line_chart(df.set_index("Date")["Close"])

    st.markdown("---")

    # ---- Feature Engineering section ----
    st.subheader("Feature Engineering")
    gen_feat = st.button("Generate Features")
    tick = st.selectbox("Select ticker to view features", options=[TICKER_SHORT[t] for t in BANK_TICKERS])
    view_feat = st.button("View Features (plots)")

    if gen_feat:
        st.info("Building features for all tickers…")
        prog = st.progress(0)
        built = 0
        for i, t in enumerate(BANK_TICKERS, start=1):
            csv_path = os.path.join(CACHE_DIR, f"{t.replace('^','')}.csv")
            if not os.path.exists(csv_path):
                st.warning(f"Missing cache for {t}. Please run 'Get Data' first.")
                continue
            df = pd.read_csv(csv_path, parse_dates=["Date"])
            feats_path = featurer.build_one(TICKER_SHORT[t], df)
            featurer.plot_one(TICKER_SHORT[t], feats_path)
            built += 1
            prog.progress(int(100 * i / len(BANK_TICKERS)))
        if built:
            st.success(f"Features generated for {built} tickers. See '{FEATURE_DIR}'.")
        else:
            st.error("No features generated. Ensure data exists in cache.")

    if view_feat:
        pdir = os.path.join(FEATURE_DIR, "plots")
        tshort = tick
        cols = st.columns(3)
        with cols[0]:
            st.image(os.path.join(pdir, f"{tshort}_close.png"))
            st.image(os.path.join(pdir, f"{tshort}_returns.png"))
        with cols[1]:
            st.image(os.path.join(pdir, f"{tshort}_zscore.png"))
            st.image(os.path.join(pdir, f"{tshort}_hv.png"))
        with cols[2]:
            st.image(os.path.join(pdir, f"{tshort}_garch_iv.png"))
            st.image(os.path.join(pdir, f"{tshort}_ivhv.png"))

# -------------------------- TAB 2: RL Model --------------------------
with tab2:
    st.header("RL Model (Strategy A for now)")

    # Hyperparameters
    st.subheader("Train/Validation Split & Hyperparameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        split = st.slider("Train fraction", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
        rl.train_split = float(split)
        epochs = st.number_input("Epochs", min_value=1, max_value=500, value=8, step=1)
        rl.epochs = int(epochs)
    with col2:
        alpha = st.number_input("Learning rate (alpha)", min_value=0.01, max_value=1.0, value=0.2, step=0.01, format="%.2f")
        rl.alpha = float(alpha)
        gamma = st.number_input("Discount (gamma)", min_value=0.5, max_value=0.999, value=0.97, step=0.01, format="%.3f")
        rl.gamma = float(gamma)
    with col3:
        epsilon = st.number_input("Epsilon (exploration)", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.2f")
        rl.epsilon = float(epsilon)
        cost = st.number_input("Transaction cost", min_value=0.0, max_value=0.01, value=0.001, step=0.0005, format="%.4f")
        rl.cost = float(cost)

    train_btn = st.button("Train (Strategy A: pooled tickers)", type="primary")
    if train_btn:
        st.info("Training… this may take a moment.")
        prog = st.progress(0)
        bins, Qq, Qs, summary = rl.train_pooled(label="strategyA")
        prog.progress(100)
        st.success(f"Training complete. Models and plots saved under '{rl.out_root}'.")
        st.dataframe(summary.sort_values(["Ticker","Strategy"]).reset_index(drop=True))
        # Stash in session for commentary reuse and set trained flag
        st.session_state["strategyA_summary"] = summary.copy()
        st.session_state["trained_strategyA"] = True

    # === Commentary block (now gated behind training) ===
    if st.session_state.get("trained_strategyA", False) and st.session_state.get("strategyA_summary") is not None:
        st.markdown("### Results Commentary")
        _summ = st.session_state["strategyA_summary"]

        # Column explanations
        with st.expander("How to read the table parameters"):
            st.markdown(
                """
- **TotalReward / AvgReward**: RL training reward based on **directional correctness** and **transaction cost** — **not actual PnL**. Negative values mean the policy often aligned with unfavorable moves or paid frequent switching costs.
- **N_Steps**: Number of evaluated timesteps (validation rows), **not** number of trades.
- **SharpeRatio**: Annualized Sharpe computed on the **per-step reward series**. Useful for stability comparison, still **not portfolio Sharpe**.
- **MaxDrawdownPct**: Drawdown computed on **cumulative reward** (treated like equity). This can look very large (e.g., -800%) or even `-inf` if the series starts at 0 and trends down — it's a limitation of using reward instead of a real equity curve.
- **WinRatePct**: % of times the **non-HOLD** position matched next-step price direction. Higher is better, but it doesn’t guarantee profitability without a proper backtest.
- **N_Trades**: Number of **position changes** (including the first non-zero). High values imply more turnover and higher costs.
> **Important**: These signals are **indicative only** — there is **no real PnL** here. For financial metrics (Return, portfolio Sharpe, realistic Drawdown), add a **1-share backtest** that marks to market.
                """
            )

        # Per-bank takeaways (choose best by TotalReward, break ties with Sharpe)
        st.markdown("#### Per-Bank Takeaways")
        msgs = []
        for tkr in sorted(_summ["Ticker"].unique()):
            sub = _summ[_summ["Ticker"] == tkr].copy()
            if sub.empty:
                continue
            sub["__rank_key__"] = list(
                zip(sub["TotalReward"].fillna(-np.inf), sub["SharpeRatio"].fillna(-np.inf))
            )
            best_idx = sub["__rank_key__"].idxmax()
            best = sub.loc[best_idx]
            tr = best.get("TotalReward", np.nan)
            sh = best.get("SharpeRatio", np.nan)
            wr = best.get("WinRatePct", np.nan)
            dd = best.get("MaxDrawdownPct", np.nan)

            dd_flag = " (note: very large because computed on reward, not equity)" if pd.notna(dd) and dd < -100 else ""
            inf_flag = " (degenerate drawdown from non-recovering cumulative reward)" if (isinstance(dd, str) and "inf" in str(dd).lower()) else ""

            msgs.append(
                f"- **{tkr}** → **{best['Strategy']}** looks better by `TotalReward` (≈ {tr:,.2f})"
                f" and Sharpe (≈ {sh:,.2f}); WinRate ≈ {wr:,.1f}% ; MaxDD ≈ {dd}{'%' if pd.notna(dd) else ''}{dd_flag}{inf_flag}"
            )

        if msgs:
            st.markdown("\n".join(msgs))
        else:
            st.info("No per-bank comments available.")

        st.warning(
            "These are **indicative trading signals**. Metrics are computed on **RL rewards**, not on a backtested portfolio. "
            "Expect unusually large/negative drawdowns when computed on rewards. "
            "For realistic Return/Sharpe/Drawdown, enable a **1-share backtest** with mark-to-market equity."
        )
    # === End commentary block ===

    st.markdown("---")

    st.subheader("View Validation Results")
    tsel = st.selectbox("Select ticker", options=[TICKER_SHORT[t] for t in BANK_TICKERS])
    view_btn = st.button("View result (combined Q vs SARSA)")
    if view_btn:
        p = os.path.join(rl.out_root, "plots", f"{tsel}_QvS_strategyA.png")
        if os.path.exists(p):
            st.image(p, caption=os.path.basename(p), use_container_width=True)
        else:
            st.warning("Plot not found. Train first, or ticker may be missing enough data.")
