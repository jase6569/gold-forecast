"""Data loaders for daily UK gold (GBP per troy ounce) and the drivers behind it.

Sources, tried in order:
  1. Stooq daily CSVs (long history, free, no key)
  2. Yahoo Finance chart API (free, no key)
  3. Local CSV files in data/ (for offline runs or a manual export)

Every series is returned as a pandas DataFrame indexed by date with columns
open, high, low, close (open/high/low may be missing for some sources).
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
UA = {"User-Agent": "Mozilla/5.0 (gold-forecast research)"}

# name -> (stooq symbol, yahoo symbol)
SYMBOLS = {
    "gold_gbp": ("xaugbp", None),          # Yahoo has no direct GBP gold, we derive it
    "gold_usd": ("xauusd", "GC=F"),
    "gbpusd": ("gbpusd", "GBPUSD=X"),
    "silver_usd": ("xagusd", "SI=F"),
    "dxy": ("dx.f", "DX-Y.NYB"),
    "us10y": ("10usy.b", "^TNX"),
}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"adj close": "adj_close"})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    keep = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors="coerce")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    return df


def from_stooq(symbol: str) -> pd.DataFrame | None:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    try:
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code != 200 or not r.text.lower().startswith("date"):
            return None
        return _clean(pd.read_csv(io.StringIO(r.text)))
    except requests.RequestException:
        return None


def from_yahoo(symbol: str) -> pd.DataFrame | None:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(symbol)}"
    params = {"period1": 0, "period2": int(time.time()), "interval": "1d", "events": "history"}
    try:
        r = requests.get(url, headers=UA, params=params, timeout=30)
        if r.status_code != 200:
            return None
        res = r.json()["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        df = pd.DataFrame(
            {"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"]},
            index=pd.to_datetime(res["timestamp"], unit="s"),
        )
        return _clean(df.rename_axis("date").reset_index())
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
        return None


def from_csv(name: str) -> pd.DataFrame | None:
    p = DATA_DIR / f"{name}.csv"
    if not p.exists():
        return None
    return _clean(pd.read_csv(p))


def load(name: str, use_cache: bool = True) -> pd.DataFrame | None:
    """Load one series by name, caching to data/<name>.csv."""
    stooq_sym, yahoo_sym = SYMBOLS.get(name, (None, None))
    df = None
    if stooq_sym:
        df = from_stooq(stooq_sym)
    if (df is None or len(df) < 500) and yahoo_sym:
        alt = from_yahoo(yahoo_sym)
        if alt is not None and (df is None or len(alt) > len(df)):
            df = alt
    if df is None and use_cache:
        df = from_csv(name)
    elif df is not None:
        DATA_DIR.mkdir(exist_ok=True)
        df.reset_index().rename(columns={"index": "date"}).to_csv(DATA_DIR / f"{name}.csv", index=False)
    return df


def load_all(start: str = "2000-01-01") -> dict[str, pd.DataFrame]:
    """Load gold in GBP plus every driver we can get. Gold in GBP is required."""
    out: dict[str, pd.DataFrame] = {}
    for name in ["gold_usd", "gbpusd", "silver_usd", "dxy", "us10y", "gold_gbp"]:
        df = load(name)
        if df is not None and len(df):
            out[name] = df[df.index >= start]

    # Derive gold in GBP from USD gold and GBPUSD when a direct series isn't there,
    # or when the derived one has longer history.
    if "gold_usd" in out and "gbpusd" in out:
        g, fx = out["gold_usd"], out["gbpusd"]["close"]
        fx = fx.reindex(g.index).ffill(limit=3)
        derived = g.div(fx, axis=0).dropna(subset=["close"])
        if "gold_gbp" not in out or len(derived) > 1.2 * len(out["gold_gbp"]):
            out["gold_gbp"] = derived

    if "gold_gbp" not in out:
        raise RuntimeError(
            "Could not load gold prices in GBP. Allow stooq.com / query1.finance.yahoo.com, "
            "or put gold_gbp.csv (date, open, high, low, close) in the data folder."
        )
    return out
