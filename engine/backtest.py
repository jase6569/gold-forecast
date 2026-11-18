"""Backtesting: which indicators genuinely predicted UK gold, and how well a
combined model does when it is only ever trained on the past.

Three tests:
  1. Classic trader rules, one at a time (RSI, moving averages, MACD ...).
     Each must beat the plain base rate in BOTH halves of history to count.
  2. Information coefficient of every raw indicator, again split in halves.
  3. Walk-forward models (logistic regression + gradient boosting), retrained
     every year on everything before that year, never on the year itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 1. Classic rules. Each returns +1 (expect up), -1 (expect down) or 0 (no call)
# ---------------------------------------------------------------------------

def _sig(cond_up, cond_dn=None):
    s = pd.Series(0.0, index=cond_up.index)
    s[cond_up.fillna(False)] = 1.0
    if cond_dn is not None:
        s[cond_dn.fillna(False)] = -1.0
    return s


RULES: dict[str, tuple[str, str, callable]] = {
    # name: (family, plain description, function)
    "above_sma20": ("Trend", "Price above / below its 20-day average", lambda f: _sig(f.dist_sma20 > 0, f.dist_sma20 < 0)),
    "above_sma50": ("Trend", "Price above / below its 50-day average", lambda f: _sig(f.dist_sma50 > 0, f.dist_sma50 < 0)),
    "above_sma200": ("Trend", "Price above / below its 200-day average", lambda f: _sig(f.dist_sma200 > 0, f.dist_sma200 < 0)),
    "golden_cross": ("Trend", "50-day average above / below the 200-day", lambda f: _sig(f.sma50_over_sma200 == 1, f.sma50_over_sma200 == 0)),
    "macd_hist": ("Trend", "MACD above / below its signal line", lambda f: _sig(f.macd_hist_norm > 0, f.macd_hist_norm < 0)),
    "mom_1d": ("Momentum", "Yesterday's move continues", lambda f: _sig(f.ret_1 > 0, f.ret_1 < 0)),
    "mom_5d": ("Momentum", "Last week's move continues", lambda f: _sig(f.ret_5 > 0, f.ret_5 < 0)),
    "mom_20d": ("Momentum", "Last month's move continues", lambda f: _sig(f.ret_20 > 0, f.ret_20 < 0)),
    "breakout_20d": ("Breakout", "New 20-day high means up, new 20-day low means down", lambda f: _sig(f.new_20d_high == 1, f.new_20d_low == 1)),
    "rsi14_extreme": ("Reversal", "RSI(14) under 30 bounces, over 70 pulls back", lambda f: _sig(f.rsi14 < 30, f.rsi14 > 70)),
    "rsi2_extreme": ("Reversal", "RSI(2) under 10 bounces, over 90 pulls back", lambda f: _sig(f.rsi2 < 10, f.rsi2 > 90)),
    "bollinger_extreme": ("Reversal", "Below lower Bollinger band bounces, above upper pulls back", lambda f: _sig(f.boll_pctb < 0, f.boll_pctb > 1)),
    "stoch_extreme": ("Reversal", "Stochastic under 20 bounces, over 80 pulls back", lambda f: _sig(f.stoch_k14 < 20, f.stoch_k14 > 80)),
    "reversal_1d": ("Reversal", "A down day is followed by an up day (and vice versa)", lambda f: _sig(f.ret_1 < 0, f.ret_1 > 0)),
    "down_streak3": ("Reversal", "After 3+ down days in a row, expect a bounce", lambda f: _sig(f.down_streak >= 3, f.up_streak >= 3)),
    "dip_in_uptrend": ("Combo", "Down day while above the 200-day average: buy the dip", lambda f: _sig((f.ret_1 < 0) & (f.dist_sma200 > 0))),
    "rsi2_dip_uptrend": ("Combo", "RSI(2) under 10 while above the 200-day average", lambda f: _sig((f.rsi2 < 10) & (f.dist_sma200 > 0))),
    "monday": ("Calendar", "Mondays tend to rise", lambda f: _sig(f.dow == 0)),
    "friday": ("Calendar", "Fridays tend to rise", lambda f: _sig(f.dow == 4)),
}

DRIVER_RULES = {
    "sterling_fell": ("Drivers", "Pound fell against the dollar today (gold in GBP up tomorrow?)", lambda f: _sig(f.gbpusd_ret_1 < 0, f.gbpusd_ret_1 > 0), "gbpusd_ret_1"),
    "dollar_weak_5d": ("Drivers", "Dollar index fell over the last week", lambda f: _sig(f.dxy_ret_5 < 0, f.dxy_ret_5 > 0), "dxy_ret_5"),
    "yields_fell_5d": ("Drivers", "US 10-year yields fell over the last week", lambda f: _sig(f.us10y_chg_5 < 0, f.us10y_chg_5 > 0), "us10y_chg_5"),
}


@dataclass
class RuleResult:
    name: str
    family: str
    description: str
    rows: dict  # period -> metrics


def _rule_metrics(sig: pd.Series, fwd: pd.Series) -> dict:
    ok = sig.notna() & fwd.notna()
    sig, fwd = sig[ok], fwd[ok]
    up_rate = (fwd > 0).mean()
    calls = sig != 0
    n = int(calls.sum())
    if n < 30:
        return {"n": n, "hit": np.nan, "base": np.nan, "edge_bp": np.nan, "p": np.nan}
    s, r = sig[calls], fwd[calls]
    correct = int(((s > 0) & (r > 0)).sum() + ((s < 0) & (r < 0)).sum())
    # Base rate a coin-flipper would get making the same mix of up/down calls
    share_up = (s > 0).mean()
    base = share_up * up_rate + (1 - share_up) * (1 - up_rate)
    p = stats.binomtest(correct, n, base, alternative="greater").pvalue
    # Edge: average return captured per call, minus what the same calls would
    # get on average days (basis points, 1bp = 0.01%)
    uncond = fwd.mean()
    edge = (s * (r - uncond)).mean() * 1e4
    return {"n": n, "hit": correct / n, "base": base, "edge_bp": edge, "p": p}


def test_rules(f: pd.DataFrame, fwd: pd.Series, split_date: pd.Timestamp) -> list[RuleResult]:
    rules = dict(RULES)
    for k, (fam, desc, fn, col) in DRIVER_RULES.items():
        if col in f.columns:
            rules[k] = (fam, desc, fn)
    out = []
    for name, (fam, desc, fn) in rules.items():
        sig = fn(f)
        rows = {
            "first_half": _rule_metrics(sig[sig.index < split_date], fwd[fwd.index < split_date]),
            "second_half": _rule_metrics(sig[sig.index >= split_date], fwd[fwd.index >= split_date]),
            "all": _rule_metrics(sig, fwd),
        }
        out.append(RuleResult(name, fam, desc, rows))
    return out


def rule_verdict(r: RuleResult) -> str:
    a, b = r.rows["first_half"], r.rows["second_half"]
    if np.isnan(a["edge_bp"]) or np.isnan(b["edge_bp"]):
        return "Too rare to judge"
    if a["edge_bp"] > 0 and b["edge_bp"] > 0 and a["p"] < 0.1 and b["p"] < 0.1:
        return "Works, and held up"
    if a["edge_bp"] > 0 and b["edge_bp"] > 0:
        return "Slight edge both halves, not proven"
    if (a["edge_bp"] > 0) != (b["edge_bp"] > 0):
        return "Worked once, then stopped"
    return "Doesn't work"


# ---------------------------------------------------------------------------
# 2. Information coefficient per indicator
# ---------------------------------------------------------------------------

def information_coefficients(f: pd.DataFrame, cols: list[str], fwd: pd.Series, split_date) -> pd.DataFrame:
    rows = []
    for c in cols:
        out = {"feature": c}
        for label, mask in [("ic_first", f.index < split_date), ("ic_second", f.index >= split_date)]:
            x, y = f.loc[mask, c], fwd[mask]
            ok = x.notna() & y.notna()
            out[label] = stats.spearmanr(x[ok], y[ok]).statistic if ok.sum() > 100 else np.nan
        rows.append(out)
    df = pd.DataFrame(rows)
    df["consistent"] = np.sign(df.ic_first) == np.sign(df.ic_second)
    df["ic_min_abs"] = np.where(df.consistent, np.minimum(df.ic_first.abs(), df.ic_second.abs()), 0.0)
    return df.sort_values("ic_min_abs", ascending=False)


# ---------------------------------------------------------------------------
# 3. Walk-forward models
# ---------------------------------------------------------------------------

def _models():
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    gb = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.03, max_iter=200, min_samples_leaf=80, l2_regularization=1.0, random_state=0
    )
    return {"logistic": lr, "boosting": gb}


def select_features(train: pd.DataFrame, cols: list[str], max_features: int = 8) -> list[str]:
    """Keep only indicators that pointed the same way in both halves of the
    training window. Uses training data only, so the test year stays unseen."""
    mid = train.index[len(train) // 2]
    y = train["fwd"] if "fwd" in train.columns else train["y"]
    scores = []
    for c in cols:
        ics = []
        for m in (train.index < mid, train.index >= mid):
            x, yy = train.loc[m, c], y[m]
            ok = x.notna() & yy.notna()
            if ok.sum() < 200 or x[ok].nunique() < 2:
                ics.append(np.nan)
                continue
            ics.append(stats.spearmanr(x[ok], yy[ok]).statistic)
        a, b = ics
        if np.isnan(a) or np.isnan(b) or np.sign(a) != np.sign(b):
            continue
        s = min(abs(a), abs(b))
        # Must be clearly non-zero (1.5 standard errors) in BOTH halves
        if s >= 1.5 / np.sqrt(len(train) / 2):
            scores.append((s, c))
    scores.sort(reverse=True)
    chosen = [c for _, c in scores[:max_features]]
    return chosen


def walk_forward(f: pd.DataFrame, cols: list[str], y: pd.Series, fwd: pd.Series | None = None,
                 horizon: int = 1, min_train_years: int = 5) -> pd.DataFrame:
    data = f[cols].copy()
    data["y"] = y
    if fwd is not None:
        data["fwd"] = fwd
    data = data[data["y"].notna()]
    years = sorted(set(data.index.year))
    first_test_year = years[0] + min_train_years
    preds = []
    for yr in [y_ for y_ in years if y_ >= first_test_year]:
        test = data[data.index.year == yr]
        train = data[data.index < test.index[0]]
        # Purge: the last `horizon` training rows have targets that overlap the test year
        train = train.iloc[:-horizon] if horizon > 0 else train
        if len(train) < 500 or len(test) == 0:
            continue
        use = select_features(train, cols)
        out = pd.DataFrame(index=test.index)
        if not use:  # nothing proven: the honest forecast is the base rate
            out["logistic"] = out["boosting"] = train["y"].mean()
            out["n_features"] = 0
            out["train_up_rate"] = train["y"].mean()
            preds.append(out)
            continue
        for name, m in _models().items():
            m.fit(train[use], train["y"].astype(int))
            out[name] = m.predict_proba(test[use])[:, 1]
        out["n_features"] = len(use)
        out["train_up_rate"] = train["y"].mean()
        preds.append(out)
    p = pd.concat(preds)
    # Main model is the simple one: in testing, boosting chased noise and
    # diluted real but small edges. Boosting is kept as a challenger.
    p["ensemble"] = p["logistic"]
    p["y"] = data.loc[p.index, "y"]
    return p


def evaluate_predictions(p: pd.DataFrame, fwd: pd.Series, col: str = "ensemble") -> dict:
    y = p["y"].astype(int)
    prob = p[col]
    pred_up = prob > 0.5
    acc = (pred_up == (y == 1)).mean()
    always_up = (y == 1).mean()
    naive = ((p["train_up_rate"] > 0.5) == (y == 1)).mean()
    res = {
        "n_days": len(p),
        "accuracy": acc,
        "always_up_accuracy": always_up,
        "naive_accuracy": naive,
        "auc": roc_auc_score(y, prob) if y.nunique() > 1 else np.nan,
    }
    r = fwd.reindex(p.index)
    # Confidence bands: how often it's right when it's more sure
    bands = []
    for lo_, hi_, label in [(0.0, 0.45, "Strong down"), (0.45, 0.48, "Lean down"), (0.48, 0.52, "No clear call"),
                            (0.52, 0.55, "Lean up"), (0.55, 1.01, "Strong up")]:
        m = (prob >= lo_) & (prob < hi_)
        if m.sum() == 0:
            bands.append({"band": label, "days": 0, "share": 0, "right": np.nan, "avg_next_day_pct": np.nan})
            continue
        if label == "No clear call":
            right = np.nan
        elif "down" in label:
            right = (y[m] == 0).mean()
        else:
            right = (y[m] == 1).mean()
        bands.append({"band": label, "days": int(m.sum()), "share": m.mean(), "right": right,
                      "avg_next_day_pct": r[m].mean() * 100})
    res["bands"] = pd.DataFrame(bands)
    # Per-year accuracy
    yr = pd.DataFrame({"right": pred_up == (y == 1), "up": y == 1}, index=p.index)
    res["by_year"] = yr.groupby(yr.index.year).agg(accuracy=("right", "mean"), up_days=("up", "mean"), days=("right", "size"))
    # Dealer view: sell signal days vs hold signal days
    sell = prob < 0.48
    hold = prob > 0.52
    res["sell_days_avg_next_pct"] = r[sell].mean() * 100
    res["hold_days_avg_next_pct"] = r[hold].mean() * 100
    res["all_days_avg_next_pct"] = r.mean() * 100
    return res
