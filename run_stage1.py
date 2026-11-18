"""Stage one: load 20+ years of daily UK gold, test every indicator, and write
an HTML report saying what actually predicted the next day.

    python run_stage1.py                # real data (needs stooq / yahoo access)
    python run_stage1.py --selftest     # random-walk data: should find NO edge
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine import data as D
from engine.backtest import (evaluate_predictions, information_coefficients, rule_verdict,
                             test_rules, walk_forward)
from engine.features import build_features, build_targets, feature_columns

ROOT = Path(__file__).resolve().parent

PLAIN_NAMES = {
    "ret_1": "Today's move", "ret_2": "2-day move", "ret_3": "3-day move", "ret_5": "1-week move",
    "ret_10": "2-week move", "ret_20": "1-month move", "ret_60": "3-month move",
    "dist_sma5": "Gap to 5-day average", "dist_sma10": "Gap to 10-day average", "dist_sma20": "Gap to 20-day average",
    "dist_sma50": "Gap to 50-day average", "dist_sma100": "Gap to 100-day average", "dist_sma200": "Gap to 200-day average",
    "sma50_over_sma200": "Golden cross state", "macd_norm": "MACD", "macd_hist_norm": "MACD histogram",
    "rsi2": "RSI (2)", "rsi14": "RSI (14)", "stoch_k14": "Stochastic", "williams_r14": "Williams %R",
    "boll_pctb": "Bollinger position", "vol_10": "2-week volatility", "vol_20": "1-month volatility",
    "vol_ratio": "Volatility rising or falling", "atr14_pct": "Average true range", "range_pos": "Close within day's range",
    "donchian20_pos": "Position in 20-day range", "dist_52w_high": "Distance from 1-year high",
    "dist_52w_low": "Distance from 1-year low", "new_20d_high": "New 20-day high", "new_20d_low": "New 20-day low",
    "down_streak": "Down days in a row", "up_streak": "Up days in a row", "dow": "Day of week", "month": "Month",
    "usd_gold_ret_1": "Gold in dollars, today", "usd_gold_ret_5": "Gold in dollars, 1 week",
    "gbpusd_ret_1": "Pound vs dollar, today", "gbpusd_ret_5": "Pound vs dollar, 1 week",
    "gbpusd_dist_sma50": "Pound vs its 50-day average", "dxy_ret_1": "Dollar index, today",
    "dxy_ret_5": "Dollar index, 1 week", "us10y_chg_1": "US 10-year yield, today", "us10y_chg_5": "US 10-year yield, 1 week",
    "gold_silver_ratio_z": "Gold/silver ratio stretch",
}


def synthetic_series(seed: int = 7) -> dict[str, pd.DataFrame]:
    """Random-walk prices with gold's typical drift and volatility. By construction
    nothing here is predictable, so a sound engine must report no real edge."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", "2026-11-13")
    n = len(idx)

    def walk(start, drift, vol):
        r = rng.normal(drift, vol, n)
        c = start * np.exp(np.cumsum(r))
        o = c * np.exp(rng.normal(0, vol / 3, n))
        h = np.maximum(c, o) * np.exp(np.abs(rng.normal(0, vol / 2, n)))
        l = np.minimum(c, o) * np.exp(-np.abs(rng.normal(0, vol / 2, n)))
        return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)

    gu = walk(280, 0.00035, 0.011)
    fx = walk(1.6, -0.00004, 0.006)
    gb = gu.div(fx["close"], axis=0)
    return {"gold_usd": gu, "gbpusd": fx, "gold_gbp": gb, "silver_usd": walk(5, 0.0003, 0.018),
            "dxy": walk(100, 0, 0.005), "us10y": walk(5, -0.00005, 0.02)}


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def pct(x, d=1):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def build_report(series, f, rules, ic, preds, ev, selftest: bool) -> str:
    g = series["gold_gbp"]["close"]
    start, end = g.index[0].date(), g.index[-1].date()
    split = rules_split_date(f)

    rule_rows = []
    for r in sorted(rules, key=lambda r: -np.nan_to_num(min(r.rows["first_half"]["edge_bp"], r.rows["second_half"]["edge_bp"]), nan=-99)):
        a, b = r.rows["first_half"], r.rows["second_half"]
        v = rule_verdict(r)
        cls = {"Works, and held up": "good", "Slight edge both halves, not proven": "maybe"}.get(v, "bad")
        rule_rows.append(
            f"<tr class='{cls}'><td><b>{r.family}</b><br>{r.description}</td>"
            f"<td>{pct(a['hit'])}<small> vs {pct(a['base'])}</small></td>"
            f"<td>{pct(b['hit'])}<small> vs {pct(b['base'])}</small></td>"
            f"<td>{r.rows['all']['n']:,}</td><td>{v}</td></tr>")

    ic_rows = "".join(
        f"<tr><td>{PLAIN_NAMES.get(row.feature, row.feature)}</td><td>{row.ic_first:+.3f}</td>"
        f"<td>{row.ic_second:+.3f}</td><td>{'Yes' if row.consistent else 'No'}</td></tr>"
        for row in ic.head(12).itertuples())

    bands = ev["bands"]
    band_rows = "".join(
        f"<tr><td>{b.band}</td><td>{b.days:,} ({pct(b.share, 0)})</td><td>{pct(b.right)}</td>"
        f"<td>{'–' if np.isnan(b.avg_next_day_pct) else f'{b.avg_next_day_pct:+.2f}%'}</td></tr>"
        for b in bands.itertuples())

    # Charts
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(g.index, g.values, color="#a07a1c", lw=1)
    ax.set_title("Gold in pounds per troy ounce" + (" (synthetic test data)" if selftest else ""), fontsize=10)
    ax.grid(alpha=0.2)
    price_png = fig_to_b64(fig)

    by = ev["by_year"]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(by.index.astype(str), by["accuracy"] * 100, color=["#2e7d32" if a > 0.5 else "#c62828" for a in by["accuracy"]])
    ax.plot(by.index.astype(str), by["up_days"] * 100, color="#555", ls="--", lw=1, label="Share of up days")
    ax.axhline(50, color="#999", lw=0.8)
    ax.set_ylim(35, 70)
    ax.set_ylabel("% right")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend(fontsize=8)
    ax.set_title("Model: next-day direction called right, each year (never trained on that year)", fontsize=10)
    year_png = fig_to_b64(fig)

    works = [r for r in rules if rule_verdict(r) == "Works, and held up"]
    acc, up = ev["accuracy"], ev["always_up_accuracy"]
    strong = bands[bands.band.isin(["Strong up", "Strong down"])]
    strong_days = strong.days.sum()
    strong_right = (strong.right * strong.days).sum() / strong_days if strong_days else np.nan

    summary = []
    if selftest:
        summary.append("<p class='warn'>This run uses random-walk test data, not real gold prices. "
                       "It checks that the engine can't cheat by peeking at the future. A sound engine shows "
                       "accuracy near the base rate and almost no indicators passing.</p>")
    summary.append(f"<p>Tested on {len(g):,} trading days of gold in pounds, {start:%B %Y} to {end:%B %Y}. "
                   f"History was split at {split:%B %Y}: an indicator only counts if it worked in both halves.</p>")
    summary.append(f"<p><b>{len(works)} of {len(rules)}</b> classic trader signals passed that test"
                   + (": " + ", ".join(r.description.lower() for r in works) + "." if works else ".") + "</p>")
    summary.append(f"<p>The combined model called tomorrow's direction right <b>{pct(acc)}</b> of the time, "
                   f"against <b>{pct(up)}</b> for simply saying \"up\" every day. "
                   f"On its strong-signal days ({strong_days:,} of them) it was right <b>{pct(strong_right)}</b>.</p>")
    e5 = ev["h5"]
    summary.append(f"<p>Looking a week ahead (useful for \"hold until it goes up\"), it called the 5-day direction right "
                   f"<b>{pct(e5['accuracy'])}</b> of the time, against <b>{pct(e5['always_up_accuracy'])}</b> for always saying up.</p>")
    summary.append(f"<p>For selling: on days it flagged \"sell tomorrow\", gold moved {ev['sell_days_avg_next_pct']:+.3f}% the next day "
                   f"on average. On \"hold\" days it moved {ev['hold_days_avg_next_pct']:+.3f}%. "
                   f"The average day was {ev['all_days_avg_next_pct']:+.3f}%.</p>")

    css = """body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#222;line-height:1.5}
h1{font-size:24px;margin-bottom:4px}h2{font-size:18px;margin-top:32px;border-bottom:2px solid #a07a1c;padding-bottom:4px}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:6px 8px;border-bottom:1px solid #eee;text-align:left;vertical-align:top}
th{background:#faf6ec}small{color:#888}tr.good td{background:#eef7ee}tr.maybe td{background:#fffbe8}
.warn{background:#fdecea;padding:10px;border-radius:6px}.sub{color:#777;margin-top:0}img{max-width:100%}"""
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>UK Gold Forecast: Stage One Backtest</title><style>{css}</style></head><body>
<h1>UK Gold Forecast: what actually works</h1><p class="sub">Stage one backtest, generated {date.today():%d %B %Y}</p>
<h2>The short version</h2>{''.join(summary)}
<img src="data:image/png;base64,{price_png}">
<h2>Classic trader signals, tested one by one</h2>
<p>Hit rate when the signal made a call, next to what you'd get guessing with the same mix of up and down calls. Green rows beat chance in both halves of history.</p>
<table><tr><th>Signal</th><th>Before {split:%Y}</th><th>{split:%Y} onward</th><th>Calls</th><th>Verdict</th></tr>{''.join(rule_rows)}</table>
<h2>Strongest raw indicators</h2>
<p>How closely each indicator lined up with the next day's move (0 means no link; gold indicators rarely beat 0.05). "Consistent" means it pointed the same way in both halves.</p>
<table><tr><th>Indicator</th><th>First half</th><th>Second half</th><th>Consistent</th></tr>{ic_rows}</table>
<h2>Combined model, tested honestly</h2>
<p>Retrained each year using only earlier years, then scored on a year it had never seen. {ev['n_days']:,} days scored in total.</p>
<table><tr><th>Signal strength</th><th>Days</th><th>Right</th><th>Avg next-day move</th></tr>{band_rows}</table>
<img src="data:image/png;base64,{year_png}">
<h2>What this means for the app</h2>
<p>Only signals that passed here go into the evening forecast. The app shows a firm "sell tomorrow" or "hold" on strong-signal days and says "no clear call" the rest of the time, rather than guessing.</p>
</body></html>"""


_SPLIT = {}


def rules_split_date(f):
    return _SPLIT.get("d", f.index[len(f) // 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    series = synthetic_series() if args.selftest else D.load_all(args.start)
    f = build_features(series)
    t = build_targets(f["close"], horizons=(1, 5))
    f = f.iloc[200:]  # warm-up for the 200-day average
    t = t.loc[f.index]
    split = f.index[len(f) // 2]
    _SPLIT["d"] = split
    cols = feature_columns(f)

    rules = test_rules(f, t["fwd_ret_1"], split)
    ic = information_coefficients(f, cols, t["fwd_ret_1"], split)
    preds = walk_forward(f, cols, t["up_1"], t["fwd_ret_1"], horizon=1)
    ev = evaluate_predictions(preds, t["fwd_ret_1"])
    preds5 = walk_forward(f, cols, t["up_5"], t["fwd_ret_5"], horizon=5)
    ev5 = evaluate_predictions(preds5, t["fwd_ret_5"])
    ev["h5"] = ev5

    out = Path(args.out) if args.out else ROOT / "reports" / ("selftest.html" if args.selftest else "stage1_report.html")
    out.parent.mkdir(exist_ok=True)
    html = build_report(series, f, rules, ic, preds, ev, args.selftest)
    out.write_text(html)
    if not args.selftest:
        web = ROOT / "web"
        (web / "data").mkdir(parents=True, exist_ok=True)
        (web / "report.html").write_text(html)

        def pack(e):
            b = e["bands"].replace({np.nan: None})
            return {"accuracy": e["accuracy"], "always_up": e["always_up_accuracy"], "auc": e["auc"],
                    "days": e["n_days"], "bands": b.to_dict(orient="records")}
        (web / "data" / "backtest.json").write_text(json.dumps({
            "generated": date.today().isoformat(), "from": str(f.index[0].date()), "to": str(f.index[-1].date()),
            "h1": pack(ev), "h5": pack(ev5),
            "rules_passing": [r.description for r in rules if rule_verdict(r) == "Works, and held up"],
        }, indent=1, default=float))

    summary = {
        "days": len(f), "accuracy": round(ev["accuracy"], 4), "always_up": round(ev["always_up_accuracy"], 4),
        "auc": round(ev["auc"], 4), "rules_passing": [r.name for r in rules if rule_verdict(r) == "Works, and held up"],
        "sell_days_avg_next_pct": round(ev["sell_days_avg_next_pct"], 4),
        "hold_days_avg_next_pct": round(ev["hold_days_avg_next_pct"], 4),
    }
    print(json.dumps(summary, indent=2))
    print(ev["bands"].to_string(index=False))
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
