"""Calibrated price ranges: where gold is likely to be tomorrow and in a week.

Volatility is tracked with an exponentially weighted average of recent moves.
The range width comes from how big moves actually were relative to that
volatility over the previous four years, so fat tails are built in. Only
past data is used at every point, and the ranges are scored for how often
the real price landed inside.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAMBDA = 0.94
LOOKBACK = 1000
COVER = 0.75  # aim: the real price lands inside the range 3 times in 4


def ewma_vol(close: pd.Series) -> pd.Series:
    r = np.log(close).diff()
    return np.sqrt((r ** 2).ewm(alpha=1 - LAMBDA, adjust=False).mean())


def _z(close, vol, h):
    fwd = np.log(close.shift(-h) / close)
    return fwd / (vol * np.sqrt(h))


def backtest(close: pd.Series, horizons=(1, 5), start: str = "2008-01-01") -> dict:
    vol = ewma_vol(close)
    lo_q, hi_q = (1 - COVER) / 2, 1 - (1 - COVER) / 2
    out = {}
    for h in horizons:
        z = _z(close, vol, h)
        # Quantiles known at day t use only z values whose outcome was known by t
        known = z.shift(h)
        qlo = known.rolling(LOOKBACK, min_periods=500).quantile(lo_q)
        qhi = known.rolling(LOOKBACK, min_periods=500).quantile(hi_q)
        lo = close * np.exp(vol * np.sqrt(h) * qlo)
        hi = close * np.exp(vol * np.sqrt(h) * qhi)
        actual = close.shift(-h)
        m = (close.index >= start) & actual.notna() & lo.notna()
        inside = (actual[m] >= lo[m]) & (actual[m] <= hi[m])
        width = ((hi[m] - lo[m]) / close[m]).mean()
        out[f"h{h}"] = {"target": COVER, "hit": float(inside.mean()), "days": int(m.sum()),
                        "avg_width_pct": float(width * 100)}
    return out


def latest(close: pd.Series, horizons=(1, 5)) -> dict:
    vol = ewma_vol(close)
    lo_q, hi_q = (1 - COVER) / 2, 1 - (1 - COVER) / 2
    out = {}
    c, v = float(close.iloc[-1]), float(vol.iloc[-1])
    for h in horizons:
        z = _z(close, vol, h).dropna().iloc[-LOOKBACK:]
        qlo, qhi = float(z.quantile(lo_q)), float(z.quantile(hi_q))
        out[f"h{h}"] = {"lo": round(c * np.exp(v * np.sqrt(h) * qlo), 2), "hi": round(c * np.exp(v * np.sqrt(h) * qhi), 2)}
    return out
