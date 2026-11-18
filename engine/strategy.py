"""When to sell: lock the price in now, or wait?

A dealer has a batch of gold that must be sold within N trading days. Each
evening the choice is lock in at today's price, or wait a day. This module
tests rules for that choice on walk-forward predictions (the model never saw
the days it is scored on), tunes each rule on the first half of that period,
then reports how it did on the second half, which it never saw either.

All results are measured against two plain baselines:
  sell now      lock in on day one, whatever happens
  sell at end   always wait until the deadline
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

WINDOWS = (3, 5, 10)
THRESHOLDS = (None, 0.44, 0.46, 0.48, 0.50)   # lock if tomorrow's up-chance is below this
TAKE_PROFIT = (None, 0.5, 1.0, 1.5, 2.0)        # lock if price is up this many daily ranges from the start
PROTECT = (None, 1.5, 3.0)                      # lock if price is down this many daily ranges from the start


def _matrices(close: pd.Series, p1: pd.Series, atr: pd.Series, n: int):
    idx = p1.index
    pos = close.index.get_indexer(idx)
    ok = (pos >= 0) & (pos + n < len(close))
    pos = pos[ok]
    starts = idx[ok]
    c = close.values
    M = np.stack([c[pos + d] for d in range(n + 1)], axis=1)          # prices day 0..n
    pv = p1.reindex(close.index).values
    P = np.stack([pv[pos + d] for d in range(n + 1)], axis=1)         # tomorrow's up-chance, made each evening
    A = atr.reindex(close.index).values[pos]
    return starts, M, P, A


def _realised(M, P, A, t, k, s):
    n = M.shape[1] - 1
    lock = np.zeros_like(M, dtype=bool)
    if t is not None:
        lock |= np.nan_to_num(P, nan=1.0) < t
    rel = M / M[:, [0]] - 1
    if k is not None:
        lock |= rel >= (k * A)[:, None]
    if s is not None:
        lock |= rel <= -(s * A)[:, None]
    lock[:, n] = True  # deadline
    day = lock.argmax(axis=1)
    price = M[np.arange(len(M)), day]
    return np.log(price / M[:, 0]), day


def _summary(r, day, now_base=None):
    out = {"mean_bp": float(np.mean(r) * 1e4), "per_10k": float((np.exp(np.mean(r)) - 1) * 1e4),
           "worst_5pct_bp": float(np.percentile(r, 5) * 1e4), "avg_days_waited": float(np.mean(day)),
           "batches": int(len(r))}
    if now_base is not None:
        out["beats_sell_now"] = float(np.mean(r > now_base + 1e-12))
    return out


def evaluate(close: pd.Series, p1: pd.Series, atr_pct: pd.Series) -> dict:
    results = {}
    for n in WINDOWS:
        starts, M, P, A = _matrices(close, p1, atr_pct, n)
        half = len(starts) // 2
        first, second = slice(0, half), slice(half, None)
        zero = np.zeros(len(starts))
        end_r = np.log(M[:, -1] / M[:, 0])

        # Tune on the first half only
        best, best_score = None, -np.inf
        grid = []
        for t, k, s in itertools.product(THRESHOLDS, TAKE_PROFIT, PROTECT):
            r, day = _realised(M, P, A, t, k, s)
            score = r[first].mean()
            grid.append((t, k, s, score, r[second].mean()))
            if score > best_score:
                best, best_score = (t, k, s), score
        r, day = _realised(M, P, A, *best)

        # Model-only rule (no price targets), tuned the same way, so we can see
        # whether the forecast itself adds anything over plain waiting
        mbest = max((g for g in grid if g[1] is None and g[2] is None and g[0] is not None), key=lambda g: g[3])
        mr, mday = _realised(M, P, A, mbest[0], None, None)

        # Non-overlapping batches for a fair "is it real?" check on the unseen half
        step = slice(half, None, n + 1)
        diff = (r - end_r)[step]
        tstat = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))) if len(diff) > 5 else np.nan

        results[str(n)] = {
            "window_days": n,
            "tested_from": str(starts[half].date()), "tested_to": str(starts[-1].date()),
            "rule": {"lock_if_up_chance_below": best[0], "take_profit_ranges": best[1], "protect_ranges": best[2]},
            "rule_unseen": _summary(r[second], day[second], zero[second]),
            "model_only_rule": {"lock_if_up_chance_below": mbest[0], **_summary(mr[second], mday[second], zero[second])},
            "sell_now_unseen": _summary(zero[second], np.zeros(len(starts))[second]),
            "sell_at_end_unseen": _summary(end_r[second], np.full(len(starts), n)[second], zero[second]),
            "rule_vs_end_tstat": tstat,
        }
    return results


def firm_call_thresholds(p1: pd.Series, y: pd.Series, top: float = 0.10, min_hist: int = 250) -> dict:
    """Firm calls = the most confident nights. Thresholds come from past
    predictions only (expanding window), so hit rates are honest."""
    p = p1.dropna()
    hi = p.expanding(min_hist).quantile(1 - top).shift(1)
    lo = p.expanding(min_hist).quantile(top).shift(1)
    yy = y.reindex(p.index)
    up = p >= hi
    dn = p <= lo
    return {
        "top_share": top,
        "firm_up_days": int(up.sum()), "firm_up_right": float((yy[up] == 1).mean()) if up.any() else None,
        "firm_down_days": int(dn.sum()), "firm_down_right": float((yy[dn] == 0).mean()) if dn.any() else None,
        "all_days_up_rate": float(yy.mean()),
        "current_hi": float(p.quantile(1 - top)), "current_lo": float(p.quantile(top)),
    }
