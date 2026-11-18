"""Daily indicators for UK gold.

Every value on row t uses only information known at the close of day t.
Targets (what happens next) are built separately and shifted forward, so
nothing from the future leaks into an indicator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_features(series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    g = series["gold_gbp"].copy()
    c = g["close"]
    has_hl = {"high", "low"}.issubset(g.columns) and g[["high", "low"]].notna().mean().min() > 0.9
    hi = g["high"].fillna(c) if has_hl else c
    lo = g["low"].fillna(c) if has_hl else c
    hi = np.maximum(hi, c)
    lo = np.minimum(lo, c)

    f = pd.DataFrame(index=g.index)
    f["close"] = c
    lr = np.log(c).diff()
    f["ret_1"] = lr

    # Momentum: returns over the last n days
    for n in [2, 3, 5, 10, 20, 60]:
        f[f"ret_{n}"] = np.log(c / c.shift(n))

    # Trend: distance from moving averages
    for n in [5, 10, 20, 50, 100, 200]:
        f[f"dist_sma{n}"] = c / c.rolling(n).mean() - 1
    f["sma50_over_sma200"] = (c.rolling(50).mean() > c.rolling(200).mean()).astype(float)
    f.loc[c.rolling(200).count() < 200, "sma50_over_sma200"] = np.nan

    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    f["macd_norm"] = macd / c
    f["macd_hist_norm"] = (macd - sig) / c

    # Oscillators
    f["rsi2"] = rsi(c, 2)
    f["rsi14"] = rsi(c, 14)
    ll14, hh14 = lo.rolling(14).min(), hi.rolling(14).max()
    f["stoch_k14"] = 100 * (c - ll14) / (hh14 - ll14).replace(0, np.nan)
    f["williams_r14"] = f["stoch_k14"] - 100

    # Bollinger and volatility
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    f["boll_pctb"] = (c - (m20 - 2 * s20)) / (4 * s20).replace(0, np.nan)
    f["vol_10"] = lr.rolling(10).std()
    f["vol_20"] = lr.rolling(20).std()
    f["vol_ratio"] = f["vol_10"] / lr.rolling(60).std()
    tr = pd.concat([hi - lo, (hi - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    f["atr14_pct"] = tr.rolling(14).mean() / c
    f["range_pos"] = (c - lo) / (hi - lo).replace(0, np.nan) if has_hl else np.nan

    # Breakouts and levels
    f["donchian20_pos"] = (c - lo.rolling(20).min()) / (hi.rolling(20).max() - lo.rolling(20).min()).replace(0, np.nan)
    f["dist_52w_high"] = c / hi.rolling(252).max() - 1
    f["dist_52w_low"] = c / lo.rolling(252).min() - 1
    f["new_20d_high"] = (c >= hi.rolling(20).max()).astype(float)
    f["new_20d_low"] = (c <= lo.rolling(20).min()).astype(float)
    down = (lr < 0).astype(int)
    f["down_streak"] = down.groupby((down != down.shift()).cumsum()).cumsum() * down
    up = (lr > 0).astype(int)
    f["up_streak"] = up.groupby((up != up.shift()).cumsum()).cumsum() * up

    # Calendar
    f["dow"] = f.index.dayofweek
    f["month"] = f.index.month

    # Drivers: what's moving the UK price
    def ext(name: str) -> pd.Series | None:
        if name not in series:
            return None
        return series[name]["close"].reindex(f.index).ffill(limit=3)

    gu, fx, si, dx, ty = ext("gold_usd"), ext("gbpusd"), ext("silver_usd"), ext("dxy"), ext("us10y")
    if gu is not None:
        f["usd_gold_ret_1"] = np.log(gu).diff()
        f["usd_gold_ret_5"] = np.log(gu / gu.shift(5))
    if fx is not None:
        f["gbpusd_ret_1"] = np.log(fx).diff()
        f["gbpusd_ret_5"] = np.log(fx / fx.shift(5))
        f["gbpusd_dist_sma50"] = fx / fx.rolling(50).mean() - 1
    if dx is not None:
        f["dxy_ret_1"] = np.log(dx).diff()
        f["dxy_ret_5"] = np.log(dx / dx.shift(5))
    if ty is not None:
        f["us10y_chg_1"] = ty.diff()
        f["us10y_chg_5"] = ty.diff(5)
    if si is not None and gu is not None:
        ratio = gu / si
        f["gold_silver_ratio_z"] = (ratio - ratio.rolling(100).mean()) / ratio.rolling(100).std()

    return f


def build_targets(close: pd.Series, horizons=(1, 5)) -> pd.DataFrame:
    """Forward returns. Row t holds what happens AFTER the close of day t."""
    t = pd.DataFrame(index=close.index)
    for h in horizons:
        t[f"fwd_ret_{h}"] = np.log(close.shift(-h) / close)
        t[f"up_{h}"] = (t[f"fwd_ret_{h}"] > 0).astype(float)
        t.loc[t[f"fwd_ret_{h}"].isna(), f"up_{h}"] = np.nan
    return t


FEATURE_EXCLUDE = {"close"}


def feature_columns(f: pd.DataFrame) -> list[str]:
    return [c for c in f.columns if c not in FEATURE_EXCLUDE and f[c].notna().mean() > 0.5]
