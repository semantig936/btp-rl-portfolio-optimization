# rl_model.py
import os
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class RLModel:
    feature_dir: str = "rl_vol_features"
    out_root: str = "strategy-A-model"  # Strategy A folder; Strategy B can use another root
    train_split: float = 0.7
    actions: np.ndarray = field(default_factory=lambda: np.array([-1,0,1], dtype=int))
    features: List[str] = field(default_factory=lambda: [
        "R_t-1","R_t-5","R_t-20","z_price_vs_SMA20","HV_20","GARCH_ann","IV_minus_HV"
    ])
    n_bins: int = 3
    alpha: float = 0.2
    gamma: float = 0.97
    epsilon: float = 0.1
    cost: float = 0.001
    epochs: int = 8

    def _ensure(self):
        os.makedirs(self.out_root, exist_ok=True)
        os.makedirs(os.path.join(self.out_root, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.out_root, "plots"), exist_ok=True)

    # ------------ data ------------
    def _load_feature_files(self, restrict_to: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        paths = {}
        for fn in os.listdir(self.feature_dir):
            if not fn.endswith("_features.csv"): continue
            t = fn.replace("_features.csv","")
            if restrict_to and t not in restrict_to: continue
            paths[t] = os.path.join(self.feature_dir, fn)

        data = {}
        for t,p in paths.items():
            df = pd.read_csv(p, parse_dates=["Date"])
            need = ["Date","Close"] + self.features
            missing = [c for c in need if c not in df.columns]
            if missing: 
                print(f"[WARN] {t} missing {missing}, skipping.")
                continue
            data[t] = df[need].dropna().sort_values("Date").reset_index(drop=True)
        return data

    def _split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        k = int(math.floor(len(df) * self.train_split))
        return df.iloc[:k].reset_index(drop=True), df.iloc[k:].reset_index(drop=True)

    # ------------ discretization ------------
    def _fit_bins(self, train_frames: List[pd.DataFrame]) -> Dict[str, np.ndarray]:
        pooled = pd.concat([f[self.features] for f in train_frames], axis=0, ignore_index=True)
        bins = {}
        for f in self.features:
            qs = np.linspace(0,1,self.n_bins+1)
            edges = np.unique(np.quantile(pooled[f].values, qs))
            if len(edges) <= 2 and np.allclose(edges.max(), edges.min()):
                v = edges[0]
                edges = np.array([v-1e-9, v, v+1e-9])
            bins[f] = edges
        return bins

    def _digitize(self, row: pd.Series, bins: Dict[str, np.ndarray]) -> Tuple[int,...]:
        idxs = []
        for f in self.features:
            e = bins[f]
            b = np.digitize([row[f]], e[1:-1])[0]  # 0..n_bins-1
            idxs.append(int(b))
        return tuple(idxs)

    def _tuple_to_state(self, tpl: Tuple[int,...]) -> int:
        s, m = 0, 1
        for v in reversed(tpl):
            s += v*m
            m *= self.n_bins
        return int(s)

    # ------------ training ------------
    def _rollout(self, df: pd.DataFrame, bins: Dict[str,np.ndarray], q: np.ndarray,
                 on_policy: bool, train: bool) -> float:
        if len(df) < 2: return 0.0
        states = [ self._tuple_to_state(self._digitize(df.iloc[t], bins)) for t in range(len(df)-1) ]
        prices = df["Close"].values
        pos_prev = 0
        total = 0.0

        def choose_action(s: int) -> int:
            if train and np.random.rand() < self.epsilon:
                return np.random.randint(len(self.actions))
            return int(np.argmax(q[s]))

        for t in range(len(states)):
            s = states[t]
            a = choose_action(s)
            pos = int(self.actions[a])
            pnl = (prices[t+1] - prices[t]) * pos
            cst = self.cost * abs(pos - pos_prev)
            r = pnl - cst
            total += r

            if train:
                if t < len(states)-1:
                    s_next = states[t+1]
                    if on_policy:  # SARSA
                        a_next = choose_action(s_next)
                        td_target = r + self.gamma*q[s_next, a_next]
                    else:          # Q-learning
                        td_target = r + self.gamma*np.max(q[s_next])
                else:
                    td_target = r
                q[s, a] += self.alpha*(td_target - q[s, a])

            pos_prev = pos
        return total

    # ------------ metrics helpers (NEW) ------------
    @staticmethod
    def _sharpe_from_rewards(reward_series: pd.Series, periods_per_year: int = 252) -> float:
        """Annualized Sharpe computed directly on per-step reward series."""
        r = pd.Series(reward_series).dropna()
        if len(r) < 2:
            return 0.0
        mu = r.mean()
        sd = r.std(ddof=1)
        if sd == 0 or np.isnan(sd):
            return 0.0
        return float((mu / sd) * np.sqrt(periods_per_year))

    @staticmethod
    def _max_drawdown_pct_from_rewards(reward_series: pd.Series) -> float:
        """
        Treat cumulative rewards as an equity-like curve and compute max drawdown.
        Returns percentage (negative numbers, e.g., -12.3 means -12.3%).
        """
        eq = pd.Series(reward_series).cumsum()
        if eq.empty:
            return 0.0
        roll_max = eq.cummax()
        dd = (eq / roll_max) - 1.0
        return float(dd.min() * 100.0)

    @staticmethod
    def _win_rate_pct_from_positions_and_prices(positions: pd.Series, close: pd.Series) -> float:
        """
        Directional correctness (HOLD excluded):
          win if position_t * (Close_{t+1} - Close_t) > 0
        Returns percentage in [0, 100].
        """
        pos = pd.Series(positions).astype(int)
        px = pd.Series(close).astype(float)
        chg = px.shift(-1) - px
        mask = (pos != 0) & chg.notna()
        if mask.sum() == 0:
            return 0.0
        wins = ((pos[mask] * chg[mask]) > 0).sum()
        return float(100.0 * wins / mask.sum())

    @staticmethod
    def _count_trades_from_positions(positions: pd.Series) -> int:
        """Count # of times the position changes (including first non-zero as a trade)."""
        pos = pd.Series(positions).astype(int)
        changes = pos.shift(1).fillna(0) != pos
        return int(changes.sum())

    def train_pooled(self, restrict_to: Optional[List[str]] = None, label: str = "strategyA"):
        """Train pooled across tickers in self.feature_dir (or restricted subset)."""
        self._ensure()
        data = self._load_feature_files(restrict_to)
        if not data:
            raise RuntimeError("No feature files found to train on.")

        splits = {t: self._split(df) for t,df in data.items() if len(df) >= 60}
        if not splits:
            raise RuntimeError("Not enough rows for any ticker to split.")

        trains = [tr for tr,_ in splits.values()]
        bins = self._fit_bins(trains)

        n_states = self.n_bins ** len(self.features)
        n_actions = len(self.actions)
        Q_q = np.zeros((n_states, n_actions), dtype=float)
        Q_s = np.zeros((n_states, n_actions), dtype=float)

        tickers = list(splits.keys())
        for ep in range(self.epochs):
            for t in np.random.permutation(tickers):
                tr, _ = splits[t]
                self._rollout(tr, bins, Q_q, on_policy=False, train=True)
                self._rollout(tr, bins, Q_s, on_policy=True,  train=True)

        mdir = os.path.join(self.out_root, "models")
        np.savez(os.path.join(mdir, f"QLEARN_{label}.npz"),
                 q_table=Q_q, actions=self.actions, features=np.array(self.features, dtype=object),
                 n_bins=self.n_bins)
        np.savez(os.path.join(mdir, f"SARSA_{label}.npz"),
                 q_table=Q_s, actions=self.actions, features=np.array(self.features, dtype=object),
                 n_bins=self.n_bins)

        summary = []
        for t, (tr, te) in splits.items():
            if len(te) < 2: 
                continue

            # Inference creates per-ticker, per-model reward CSVs and plots
            df_q = self._inference(te, bins, Q_q, f"{t}_QLEARN_{label}")
            df_s = self._inference(te, bins, Q_s, f"{t}_SARSA_{label}")

            # Aggregate metrics row-by-row
            def _row_from_df(df_out: Optional[pd.DataFrame], strategy_name: str):
                if df_out is None or df_out.empty:
                    return None
                total_reward = float(df_out["reward"].sum())
                avg_reward   = float(df_out["reward"].mean())
                n_steps      = int(len(df_out))
                sharpe       = self._sharpe_from_rewards(df_out["reward"])
                mdd_pct      = self._max_drawdown_pct_from_rewards(df_out["reward"])
                win_rate_pct = self._win_rate_pct_from_positions_and_prices(df_out["position"], df_out["Close"])
                n_trades     = self._count_trades_from_positions(df_out["position"])
                return (t, strategy_name, total_reward, avg_reward, n_steps,
                        sharpe, mdd_pct, win_rate_pct, n_trades)

            row_q = _row_from_df(df_q, f"QLEARN_{label}")
            row_s = _row_from_df(df_s, f"SARSA_{label}")
            if row_q is not None: summary.append(row_q)
            if row_s is not None: summary.append(row_s)

            # Combined Q vs S plot (kept as-is)
            if df_q is not None and df_s is not None:
                self._plot_combined(t, df_q, df_s, f"{t}_QvS_{label}")

        # Updated summary with new columns
        summ = pd.DataFrame(summary, columns=[
            "Ticker","Strategy","TotalReward","AvgReward","N_Steps",
            "SharpeRatio","MaxDrawdownPct","WinRatePct","N_Trades"
        ])
        summ.to_csv(os.path.join(self.out_root, "models", f"{label}_backtest_summary.csv"), index=False)
        return bins, Q_q, Q_s, summ

    # ------------ inference / plotting ------------
    def _inference(self, te: pd.DataFrame, bins: Dict[str,np.ndarray], q: np.ndarray, name: str) -> Optional[pd.DataFrame]:
        if len(te) < 2: return None
        states = [ self._tuple_to_state(self._digitize(te.iloc[t], bins)) for t in range(len(te)-1) ]
        prices = te["Close"].values

        pos_prev = 0
        acts, poss, rews = [], [], []
        for t in range(len(states)):
            s = states[t]
            a = int(np.argmax(q[s]))
            pos = int(self.actions[a])
            pnl = (prices[t+1] - prices[t]) * pos
            cst = self.cost * abs(pos - pos_prev)
            r = pnl - cst
            acts.append(a); poss.append(pos); rews.append(r)
            pos_prev = pos

        out = te.iloc[:-1].copy().reset_index(drop=True)
        out["action_idx"] = acts
        out["position"]   = poss
        out["reward"]     = rews
        out["signal"]     = out["position"].map({-1:"SELL",0:"HOLD",1:"BUY"})
        out.to_csv(os.path.join(self.out_root, "models", f"{name}_rewards.csv"), index=False)

        self._plot_signals(out, os.path.join(self.out_root,"plots", f"{name}_signals.png"))
        return out

    def _plot_signals(self, df: pd.DataFrame, path: str):
        plt.figure(figsize=(12,4))
        plt.plot(df["Date"], df["Close"], color="black", linewidth=1.2, label="Close")
        buy  = df[df["position"]==1]
        sell = df[df["position"]==-1]
        if not buy.empty:
            plt.scatter(buy["Date"], buy["Close"], color="green", marker="^", label="BUY", alpha=0.7)
        if not sell.empty:
            plt.scatter(sell["Date"], sell["Close"], color="red", marker="v", label="SELL", alpha=0.7)
        plt.title("Validation signals")
        plt.xlabel("Date"); plt.ylabel("Price"); plt.legend()
        plt.tight_layout(); os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path); plt.close()

    def _plot_combined(self, ticker: str, df_q: pd.DataFrame, df_s: pd.DataFrame, name: str):
        """
        Make ONE figure with TWO stacked subplots:
          - Top: Q-learning signals
          - Bottom: SARSA signals
        Shared X-axis for clean comparison.
        """
        path = os.path.join(self.out_root, "plots", f"{name}.png")

        # Compute quick summaries for titles
        tot_q = df_q["reward"].sum()
        tot_s = df_s["reward"].sum()

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 7), sharex=True)

        # --- Q-learning (top) ---
        ax = axes[0]
        ax.plot(df_q["Date"], df_q["Close"], color="black", linewidth=1.2, label="Close")
        bq = df_q[df_q["position"]==1]
        sq = df_q[df_q["position"]==-1]
        if not bq.empty: ax.scatter(bq["Date"], bq["Close"], color="green", marker="^", label="BUY (Q)", alpha=0.7)
        if not sq.empty: ax.scatter(sq["Date"], sq["Close"], color="red",   marker="v", label="SELL (Q)", alpha=0.7)
        ax.set_title(f"{ticker} — Q-learning (validation)  |  Total reward: {tot_q:,.2f}")
        ax.set_ylabel("Price")
        ax.legend(loc="best", fontsize=8)

        # --- SARSA (bottom) ---
        ax = axes[1]
        ax.plot(df_s["Date"], df_s["Close"], color="black", linewidth=1.2, label="Close")
        bs = df_s[df_s["position"]==1]
        ss = df_s[df_s["position"]==-1]
        if not bs.empty: ax.scatter(bs["Date"], bs["Close"], color="green", marker="^", label="BUY (SARSA)", alpha=0.7)
        if not ss.empty: ax.scatter(ss["Date"], ss["Close"], color="red",   marker="v", label="SELL (SARSA)", alpha=0.7)
        ax.set_title(f"{ticker} — SARSA (validation)  |  Total reward: {tot_s:,.2f}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc="best", fontsize=8)

        fig.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
