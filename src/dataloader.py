# dataloader.py
import os
import math
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# yfinance for data
try:
    import yfinance as yf
except Exception as e:
    raise ImportError("Please install yfinance: pip install yfinance") from e

# arch for GARCH (falls back to EWMA if not available)
try:
    from arch import arch_model
    HAVE_ARCH = True
except Exception:
    HAVE_ARCH = False
    warnings.warn("Package 'arch' not available. Using EWMA (RiskMetrics) as GARCH proxy.")

# scipy for IV; if missing, we’ll skip IV root solve
try:
    from math import log, sqrt, exp
    from scipy.stats import norm
    from scipy.optimize import brentq
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
    warnings.warn("Package 'scipy' not available. Implied vol will fall back to NaN/proxy.")

# ------------------- cleaning helpers -------------------
import re

def _coerce_numeric(s: pd.Series) -> pd.Series:
    """Coerce a Series to numeric: strip commas, spaces, %, NBSP; errors->NaN."""
    return pd.to_numeric(
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace("\u00a0", "", regex=False)  # NBSP
         .str.replace(" ", "", regex=False)
         .str.replace("%", "", regex=False),
        errors="coerce"
    )

def _likely_date_col(df: pd.DataFrame) -> Optional[str]:
    """Find a column that parses to dates for the majority of rows."""
    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() > 0.8:
            return col
    return None

def _drop_leading_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop 1–2 leading rows if they look like embedded headers (e.g., '^NSEBANK', 'Ticker', 'Date')."""
    d = df.copy()
    if d.empty:
        return d
    head = d.iloc[0].astype(str).str.upper().tolist()
    if any(v.startswith("^") or v == "TICKER" for v in head):
        d = d.iloc[1:].reset_index(drop=True)
    # Some Yahoo variants put a "Date" row next
    if d.shape[0] and str(d.iloc[0, 0]).strip().lower() in {"date", "dates"}:
        d = d.iloc[1:].reset_index(drop=True)
    return d

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize common column names for prices and options tables."""
    ren = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc in ("date", "dates", "timestamp", "trade_date"):
            ren[c] = "Date"
        elif lc in ("adj close", "adjclose"):
            ren[c] = "AdjClose"
        elif lc in ("close*", "close"):
            ren[c] = "Close"
        elif lc == "open":
            ren[c] = "Open"
        elif lc == "high":
            ren[c] = "High"
        elif lc == "low":
            ren[c] = "Low"
        elif lc == "volume":
            ren[c] = "Volume"
        elif lc == "price":
            ren[c] = "Price"
        elif lc in ("ticker","symbol"):
            ren[c] = "Ticker"
        elif lc in ("ltp", "last", "settle_price", "optionprice", "option_price", "premium", "option_premium"):
            ren[c] = "OptionPrice"
        else:
            ren[c] = c
    return df.rename(columns=ren)

def _clean_cached_dataframe(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Clean messy cached price CSVs:
    - Remove embedded header rows
    - Identify/rename Date column
    - Ensure a numeric Close column exists (fallback to AdjClose if needed)
    - Drop rows with non-numeric Close / non-date Date
    """
    d = _drop_leading_header_rows(df)
    d = _normalize_columns(d)

    # Identify Date
    if "Date" not in d.columns:
        if "Price" in d.columns and pd.to_datetime(d["Price"], errors="coerce").notna().mean() > 0.8:
            d = d.rename(columns={"Price": "Date"})
        else:
            guess = _likely_date_col(d)
            if guess is not None:
                d = d.rename(columns={guess: "Date"})
            else:
                raise ValueError("Could not identify a Date column in cached CSV.")

    # Choose Close
    if "Close" not in d.columns:
        if "AdjClose" in d.columns:
            d = d.rename(columns={"AdjClose": "Close"})
        else:
            candidates = [c for c in d.columns if c != "Date"]
            chosen = None
            for c in candidates:
                if _coerce_numeric(d[c]).notna().mean() > 0.8:
                    chosen = c
                    break
            if chosen is None:
                raise ValueError("Could not find a usable Close column in cached CSV.")
            d = d.rename(columns={chosen: "Close"})

    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["Close"] = _coerce_numeric(d["Close"])
    d = d.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)

    for col in ["Open", "High", "Low", "AdjClose", "Volume"]:
        if col in d.columns:
            d[col] = _coerce_numeric(d[col])

    return d

# ------------------- Black-Scholes helpers -------------------
def bs_call_price(S, K, T, r, sigma, q=0.0):
    """Black–Scholes call (continuous dividend q)."""
    if not HAVE_SCIPY:
        return np.nan
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    d1 = (log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return S*exp(-q*T)*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)

def implied_vol_call(price, S, K, T, r, q=0.0, lo=1e-4, hi=3.0):
    """Brent root-finding for call IV. Returns np.nan on failure or if scipy missing."""
    if not HAVE_SCIPY:
        return np.nan
    if price <= max(S - K*math.exp(-r*T), 0.0) or T <= 0 or S <= 0 or K <= 0:
        return np.nan
    try:
        f = lambda sigma: bs_call_price(S, K, T, r, sigma, q) - price
        return brentq(f, lo, hi, maxiter=200, disp=False)
    except Exception:
        return np.nan

# ------------------- Volatility helpers -------------------
def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close.astype(float)).diff()

def hist_vol(logret: pd.Series, window: int, trading_days: int = 252) -> pd.Series:
    return logret.rolling(window).std() * math.sqrt(trading_days)

def ewma_vol_daily(logret: pd.Series, lam=0.94) -> pd.Series:
    # RiskMetrics EWMA daily vol (not annualized)
    r = logret.dropna()
    if r.empty:
        return pd.Series(index=logret.index, dtype=float)
    prev_var = r.var() if r.var() > 0 else 1e-8
    vals = []
    for x in r:
        prev_var = lam*prev_var + (1-lam)*(x**2)
        vals.append(prev_var)
    out = pd.Series(vals, index=r.index).pow(0.5)  # daily vol
    return out.reindex(logret.index)

def garch_vol_ann(logret: pd.Series, trading_days=252) -> pd.Series:
    """Annualized GARCH(1,1) daily conditional vol if arch available, else EWMA proxy."""
    r_pct = logret.dropna() * 100.0  # arch expects percent units
    out = pd.Series(index=logret.index, dtype=float)
    if HAVE_ARCH and len(r_pct) > 100:
        try:
            am = arch_model(r_pct, vol='GARCH', p=1, q=1, mean='zero', dist='normal')
            res = am.fit(disp='off')
            cond = (res.conditional_volatility / 100.0) * math.sqrt(trading_days)
            out.loc[cond.index] = cond.astype(float)
            return out
        except Exception:
            warnings.warn("GARCH fit failed, falling back to EWMA.")
    daily = ewma_vol_daily(logret)
    return daily * math.sqrt(trading_days)

# ------------------- DataLoader -------------------
@dataclass
class DataLoader:
    tickers: List[str] = field(default_factory=lambda: ["AXISBANK.NS","BANKBARODA.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","^NSEBANK"])
    cache_dir: str = "data_cache-10"
    options_file: str = "bank_options.csv"
    risk_free_rate: float = 0.06
    option_T_days: int = 30
    banknifty_market_price: float = 100.0  # fallback premium for BankNifty if options file empty
    days: int = 3650  # default 10 years
    trading_days: int = 252

    def _ensure_dir(self, path: str):
        os.makedirs(path, exist_ok=True)

    def fetch_or_load(self, verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Ensure price CSVs exist for each ticker in cache_dir.
        Returns dict of {ticker: dataframe[Date, Open, High, Low, Close, Volume]}.
        """
        self._ensure_dir(self.cache_dir)
        end = pd.Timestamp.today(tz="Asia/Kolkata").normalize()
        start = end - pd.Timedelta(days=self.days)

        out = {}
        for t in self.tickers:
            csv_path = os.path.join(self.cache_dir, f"{t.replace('^','')}.csv")
            if os.path.exists(csv_path):
                raw = pd.read_csv(csv_path)
                try:
                    df = _clean_cached_dataframe(raw, t)
                except Exception:
                    raw = _normalize_columns(raw)
                    if "Date" in raw.columns:
                        raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
                    if "Close" in raw.columns:
                        raw["Close"] = _coerce_numeric(raw["Close"])
                    df = raw.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)
                df.to_csv(csv_path, index=False)
                out[t] = df
                if verbose: print(f"[cache+clean] {t} -> {csv_path} ({len(df)} rows)")
                continue

            # Download if missing
            if verbose: print(f"[download] {t} from yfinance...")
            yf_ticker = t
            data = yf.download(yf_ticker, start=start, end=end, progress=False, auto_adjust=False)
            if data.empty:
                print(f"[warn] no data for {t}, skipping.")
                continue
            data = data.reset_index()
            data.rename(columns=str, inplace=True)
            data = data.rename(columns={"Adj Close":"AdjClose"})
            data.to_csv(csv_path, index=False)
            out[t] = data
            if verbose: print(f"[saved] {t} -> {csv_path} ({len(data)} rows)")
        return out

    def _load_options_table(self) -> pd.DataFrame:
        """
        Load and normalize the options table.
        - Date column: accepts date/timestamp/trade_date
        - Ticker column: accepts ticker/symbol
        - Price column: accepts optionprice/option_price/premium/option_premium/ltp/last/close/price/settle_price
        If no usable price column is found, returns empty DataFrame.
        """
        try:
            df = pd.read_csv(self.options_file)
        except Exception:
            return pd.DataFrame(columns=["Date","Ticker","OptionPrice"])

        df = _normalize_columns(df)

        if "Date" not in df.columns:
            guess = _likely_date_col(df)
            if guess is not None:
                df = df.rename(columns={guess: "Date"})
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        if "OptionPrice" not in df.columns:
            price_like = [c for c in df.columns if c.lower() in
                          ("optionprice","option_price","premium","option_premium","ltp","last","close","price","settle_price")]
            if price_like:
                df["OptionPrice"] = _coerce_numeric(df[price_like[0]])
            else:
                return pd.DataFrame(columns=["Date","Ticker","OptionPrice"])
        else:
            df["OptionPrice"] = _coerce_numeric(df["OptionPrice"])

        if "Ticker" in df.columns:
            df["Ticker"] = df["Ticker"].astype(str)

        keep = [c for c in ["Date","Ticker","OptionPrice"] if c in df.columns]
        df = df[keep].dropna(subset=[c for c in keep if c != "Ticker"])
        return df

    def _implied_vol_series(self, price_df: pd.DataFrame, opt_df: pd.DataFrame, ticker: str) -> pd.Series:
        """
        Build a daily IV_30d series using Black–Scholes from option premiums when available.
        ATM approximation: K = Close_t, T = option_T_days/365, r = risk_free_rate.
        """
        if price_df.empty:
            return pd.Series(dtype=float)

        if opt_df is None or opt_df.empty:
            return pd.Series(index=price_df["Date"], dtype=float)

        if "Date" not in opt_df.columns or "OptionPrice" not in opt_df.columns:
            return pd.Series(index=price_df["Date"], dtype=float)

        opt = opt_df.copy()
        if "Ticker" in opt.columns:
            tick_key = ticker.replace("^","")
            mask = opt["Ticker"].str.upper().str.contains(tick_key.upper(), na=False)
            if mask.any():
                opt = opt[mask]

        if not HAVE_SCIPY:
            return pd.Series(index=price_df["Date"], dtype=float)

        if "Date" not in price_df.columns or "Close" not in price_df.columns:
            return pd.Series(dtype=float)

        m = pd.merge(price_df[["Date","Close"]], opt[["Date","OptionPrice"]], on="Date", how="left")

        T = self.option_T_days / 365.0
        r = self.risk_free_rate
        q = 0.0

        iv = pd.Series(index=price_df["Date"], dtype=float)
        for _, row in m.iterrows():
            date = row["Date"]
            S = row["Close"]
            prem = row["OptionPrice"]
            if pd.isna(S) or pd.isna(prem):
                continue
            S = float(S)
            K = S  # ATM
            iv_val = implied_vol_call(float(prem), S, K, T, r, q)
            iv.loc[date] = iv_val
        return iv

    def compute_vol_metrics(self, prices: Dict[str, pd.DataFrame], verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """
        For each ticker dataframe, compute:
        - log returns
        - HV_10/20/60 (annualized)
        - GARCH_ann (annualized)
        - IV_30d (from bank_options.csv if possible; else fallback for NSEBANK using banknifty_market_price)
        Returns updated dict {ticker: df}.
        """
        opt_df = self._load_options_table()
        out = {}

        for t, df in prices.items():
            df = _clean_cached_dataframe(df, t)

            # log returns
            df["log_ret"] = log_returns(df["Close"])

            # HV (annualized)
            df["HV_10"] = hist_vol(df["log_ret"], 10, self.trading_days)
            df["HV_20"] = hist_vol(df["log_ret"], 20, self.trading_days)
            df["HV_60"] = hist_vol(df["log_ret"], 60, self.trading_days)

            # GARCH annualized (fallback EWMA if needed)
            df["GARCH_ann"] = garch_vol_ann(df["log_ret"], trading_days=self.trading_days)

            # IV_30d from options table
            iv_series = self._implied_vol_series(df[["Date","Close"]], opt_df, t)

            # Special fallback for index if options missing
            if iv_series.isna().all():
                if "^NSEBANK" in t or "NSEBANK" in t:
                    T = self.option_T_days / 365.0
                    r = self.risk_free_rate
                    approx = []
                    for date, close in df[["Date","Close"]].itertuples(index=False):
                        S = float(close)
                        K = S
                        prem = self.banknifty_market_price
                        ivv = implied_vol_call(prem, S, K, T, r, 0.0) if HAVE_SCIPY else np.nan
                        approx.append((date, ivv))
                    iv_series = pd.Series({d:v for d,v in approx})

            # Align and fill missing IV with smoothed GARCH proxy (pass a SERIES, not .values)
            iv_full = iv_series.reindex(df["Date"]) if isinstance(iv_series, pd.Series) else pd.Series(index=df["Date"], dtype=float)
            garch_smooth = df["GARCH_ann"].ewm(span=10, adjust=False).mean() * 1.10
            df["IV_30d"] = iv_full.fillna(garch_smooth.reindex(iv_full.index))

            out[t] = df

            if verbose:
                nonnull = int(df["IV_30d"].notna().sum())
                print(f"[vol] {t}: rows={len(df)}, IV non-null={nonnull}")
        return out

    def plot_close(self, df: pd.DataFrame, title: str, out_path: Optional[str] = None):
        plt.figure(figsize=(10,4))
        plt.plot(df["Date"], df["Close"], color="black", linewidth=1.2)
        plt.title(title)
        plt.xlabel("Date"); plt.ylabel("Price")
        plt.tight_layout()
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path)
            plt.close()
        else:
            plt.show()
