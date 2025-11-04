# features.py
import os
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional GARCH dependency (fallback to EWMA if not available)
try:
    from arch import arch_model
    HAVE_ARCH = True
except Exception:
    HAVE_ARCH = False

# ----------------- helpers -----------------
def _coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace("\u00a0", "", regex=False)
         .str.replace(" ", "", regex=False)
         .str.replace("%", "", regex=False),
        errors="coerce"
    )

def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close.astype(float)).diff()

def hist_vol(logret: pd.Series, window: int, trading_days: int = 252) -> pd.Series:
    return logret.rolling(window).std() * math.sqrt(trading_days)

def sma(series: pd.Series, w: int) -> pd.Series:
    return series.rolling(w).mean()

def ewma_vol_daily(logret: pd.Series, lam=0.94) -> pd.Series:
    r = logret.dropna()
    if r.empty:
        return pd.Series(index=logret.index, dtype=float)
    prev_var = r.var() if r.var() > 0 else 1e-8
    vals = []
    for x in r:
        prev_var = lam*prev_var + (1-lam)*(x**2)
        vals.append(prev_var)
    out = pd.Series(vals, index=r.index).pow(0.5)
    return out.reindex(logret.index)

def garch_vol_ann(logret: pd.Series, trading_days=252) -> pd.Series:
    """Annualized daily conditional vol: GARCH(1,1) if available else EWMA proxy."""
    r_pct = logret.dropna() * 100.0
    out = pd.Series(index=logret.index, dtype=float)
    if HAVE_ARCH and len(r_pct) > 100:
        try:
            am = arch_model(r_pct, vol='GARCH', p=1, q=1, mean='zero', dist='normal')
            res = am.fit(disp='off')
            cond = (res.conditional_volatility / 100.0) * math.sqrt(trading_days)
            out.loc[cond.index] = cond.astype(float)
            return out
        except Exception:
            pass
    daily = ewma_vol_daily(logret)
    return daily * math.sqrt(trading_days)

# ----------------- DataFeatures -----------------
@dataclass
class DataFeatures:
    out_dir: str = "rl_vol_features"
    trading_days: int = 252

    def _ensure(self):
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "plots"), exist_ok=True)

    def build_one(self, t: str, df: pd.DataFrame) -> str:
        """
        Build feature panel and save CSV to out_dir/<ticker>_features.csv.
        Robust to missing IV_30d/GARCH_ann in the input df (recomputes if needed).
        """
        self._ensure()

        # --- Clean & ensure required columns ---
        d = df.copy()
        if "Date" not in d.columns:
            # try to find a likely date column
            for col in d.columns:
                parsed = pd.to_datetime(d[col], errors="coerce")
                if parsed.notna().mean() > 0.8:
                    d = d.rename(columns={col: "Date"})
                    break
        if "Date" not in d.columns:
            raise ValueError(f"{t}: no Date column in input data.")

        if "Close" not in d.columns:
            # fall back to AdjClose if present
            if "Adj Close" in d.columns:
                d = d.rename(columns={"Adj Close": "AdjClose"})
            if "AdjClose" in d.columns:
                d = d.rename(columns={"AdjClose": "Close"})
            else:
                # pick a numeric-looking column as a last resort
                candidates = [c for c in d.columns if c != "Date"]
                chosen = None
                for c in candidates:
                    if _coerce_numeric(d[c]).notna().mean() > 0.8:
                        chosen = c
                        break
                if chosen is None:
                    raise ValueError(f"{t}: no usable Close column in input data.")
                d = d.rename(columns={chosen: "Close"})

        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        d["Close"] = _coerce_numeric(d["Close"])
        d = d.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)

        # --- Base calculations ---
        x = d[["Date","Close"]].copy()
        x["log_ret"] = log_returns(x["Close"])

        # Return lags
        x["R_t-1"]  = x["log_ret"].shift(1)
        x["R_t-5"]  = x["log_ret"].rolling(5).sum().shift(1)
        x["R_t-20"] = x["log_ret"].rolling(20).sum().shift(1)

        # SMA & z-scores
        x["SMA20"]  = sma(x["Close"], 20)
        x["SMA60"]  = sma(x["Close"], 60)
        x["z_price_vs_SMA20"] = (x["Close"] - x["SMA20"]) / x["Close"].rolling(20).std()
        x["z_price_vs_SMA60"] = (x["Close"] - x["SMA60"]) / x["Close"].rolling(60).std()

        # HVs (annualized)
        x["HV_10"] = hist_vol(x["log_ret"], 10, self.trading_days)
        x["HV_20"] = hist_vol(x["log_ret"], 20, self.trading_days)
        x["HV_60"] = hist_vol(x["log_ret"], 60, self.trading_days)

        # GARCH_ann (use provided if exists; else compute)
        if "GARCH_ann" in d.columns:
            garch_series = pd.Series(d["GARCH_ann"].values, index=d.index)
            garch_series.index = d["Date"]
            garch_series = garch_series.reindex(x["Date"])
            x["GARCH_ann"] = garch_series.values
        else:
            x["GARCH_ann"] = garch_vol_ann(x["log_ret"], trading_days=self.trading_days)

        # IV_30d (use provided if exists; else proxy from smoothed GARCH)
        if "IV_30d" in d.columns:
            iv_series = pd.Series(d["IV_30d"].values, index=d.index)
            iv_series.index = d["Date"]
            iv_series = iv_series.reindex(x["Date"])
            x["IV_30d"] = iv_series.values
        else:
            garch_smooth = x["GARCH_ann"].ewm(span=10, adjust=False).mean() * 1.10
            x["IV_30d"] = garch_smooth.values

        # Spreads/ratios (guard against divide-by-zero)
        x["IV_minus_HV"] = x["IV_30d"] - x["HV_20"]
        with np.errstate(divide="ignore", invalid="ignore"):
            x["IV_div_HV"] = x["IV_30d"] / x["HV_20"]
        x["IV_div_HV"].replace([np.inf, -np.inf], np.nan, inplace=True)

        # Keep needed columns & dropna conservatively
        keep = ["Date","Close","R_t-1","R_t-5","R_t-20","z_price_vs_SMA20","z_price_vs_SMA60",
                "HV_10","HV_20","HV_60","GARCH_ann","IV_30d","IV_minus_HV","IV_div_HV"]
        x = x[keep].dropna().reset_index(drop=True)

        # Save CSV
        out_csv = os.path.join(self.out_dir, f"{t.replace('^','')}_features.csv")
        x.to_csv(out_csv, index=False)
        return out_csv

    def plot_one(self, t: str, feats_path: str):
        """Make the same plots as in your notebook."""
        df = pd.read_csv(feats_path, parse_dates=["Date"])

        pdir = os.path.join(self.out_dir, "plots")
        os.makedirs(pdir, exist_ok=True)

        # Close
        plt.figure(figsize=(10,4))
        plt.plot(df["Date"], df["Close"], color="black", linewidth=1.2)
        plt.title(f"{t} Close"); plt.xlabel("Date"); plt.ylabel("Price")
        plt.tight_layout(); plt.savefig(os.path.join(pdir, f"{t}_close.png")); plt.close()

        # Return lags
        plt.figure(figsize=(10,4))
        plt.plot(df["Date"], df["R_t-1"], label="R_t-1")
        plt.plot(df["Date"], df["R_t-5"], label="R_t-5")
        plt.plot(df["Date"], df["R_t-20"], label="R_t-20")
        plt.legend(); plt.title(f"{t} Return lags"); plt.xlabel("Date")
        plt.tight_layout(); plt.savefig(os.path.join(pdir, f"{t}_returns.png")); plt.close()

        # Z-scores
        plt.figure(figsize=(10,4))
        plt.plot(df["Date"], df["z_price_vs_SMA20"], label="z vs SMA20")
        plt.plot(df["Date"], df["z_price_vs_SMA60"], label="z vs SMA60")
        plt.legend(); plt.title(f"{t} Z-scored vs SMA")
        plt.tight_layout(); plt.savefig(os.path.join(pdir, f"{t}_zscore.png")); plt.close()

        # HVs
        plt.figure(figsize=(10,4))
        plt.plot(df["Date"], df["HV_10"], label="HV_10")
        plt.plot(df["Date"], df["HV_20"], label="HV_20")
        plt.plot(df["Date"], df["HV_60"], label="HV_60")
        plt.legend(); plt.title(f"{t} Historical Vol (annualized)")
        plt.tight_layout(); plt.savefig(os.path.join(pdir, f"{t}_hv.png")); plt.close()

        # GARCH vs IV
        plt.figure(figsize=(10,4))
        plt.plot(df["Date"], df["GARCH_ann"], label="GARCH_ann")
        plt.plot(df["Date"], df["IV_30d"], label="IV_30d")
        plt.legend(); plt.title(f"{t} GARCH vs IV_30d")
        plt.tight_layout(); plt.savefig(os.path.join(pdir, f"{t}_garch_iv.png")); plt.close()

        # IV-HV spread / ratio
        plt.figure(figsize=(10,4))
        plt.plot(df["Date"], df["IV_minus_HV"], label="IV - HV")
        plt.plot(df["Date"], df["IV_div_HV"], label="IV / HV")
        plt.legend(); plt.title(f"{t} IV-HV metrics")
        plt.tight_layout(); plt.savefig(os.path.join(pdir, f"{t}_ivhv.png")); plt.close()
