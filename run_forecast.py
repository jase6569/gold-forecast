"""Evening forecast. Trains on all history, gives tomorrow's verdict for UK gold
(sell tomorrow / hold / no clear call), writes web/data/latest.json and the track
record, and sends a phone notification.

    python run_forecast.py              # real data, news and calendar
    python run_forecast.py --selftest   # synthetic prices, no network
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from engine import data as D
from engine import mastermelt as MM
from engine.backtest import _models, select_features
from engine.context import STATUS as NEWS_STATUS, big_events, gold_headlines, read_news
from engine.features import build_features, build_targets, feature_columns
from run_stage1 import PLAIN_NAMES, synthetic_series

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "web" / "data"
GRAMS_PER_OZ = 31.1034768


def fit_and_predict(f, cols, t, h):
    d = f[cols].copy()
    d["y"], d["fwd"] = t[f"up_{h}"], t[f"fwd_ret_{h}"]
    train = d[d["y"].notna()]
    use = select_features(train, cols)
    latest = f[cols].iloc[[-1]]
    if not use:
        return float(train["y"].mean()), [], None
    model = _models()["logistic"]
    model.fit(train[use], train["y"].astype(int))
    p = float(model.predict_proba(latest[use])[0, 1])
    # Contribution of each indicator to today's call (standardised value x weight)
    imp, sc, lr = model.named_steps.values()
    z = sc.transform(imp.transform(latest[use]))[0]
    contrib = sorted(zip(use, z * lr.coef_[0]), key=lambda x: -abs(x[1]))
    return p, contrib, use


def describe(feature: str, value: float) -> str:
    name = PLAIN_NAMES.get(feature, feature)
    if feature.startswith(("ret_", "dist_")) or "_ret_" in feature or feature.endswith("sma50"):
        return f"{name} {value * 100:+.1f}%"
    if feature.startswith(("rsi", "stoch", "williams")):
        return f"{name} at {value:.0f}"
    if feature in {"down_streak", "up_streak"}:
        return f"{name}: {int(value)}"
    if feature == "dow":
        return f"Day of week ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][int(value)]})"
    if feature.startswith("us10y"):
        return f"{name} {value:+.2f} pts"
    return f"{name} {value:+.2f}"


def verdict(p1: float, p5: float) -> tuple[str, str]:
    """Thresholds set from the real backtest (2004 to 2026). Only the next-day
    model showed an edge; the 5-day model didn't, so it's shown but not used.
    p1 >= 0.55 was right 56% of the time (base rate 52%); p1 <= 0.48 caught
    softer next days (average -0.08% to -0.11%). In between, no proven edge."""
    if p1 >= 0.55:
        return "HOLD", "Strong"
    if p1 <= 0.45:
        return "SELL TOMORROW", "Strong"
    if p1 <= 0.48:
        return "SELL TOMORROW", "Lean"
    return "NO CLEAR CALL", ""


def band_hit_rate(bt: dict | None, p: float, key: str) -> float | None:
    if not bt or key not in bt:
        return None
    for b in bt[key]["bands"]:
        if b["lo"] <= p < b["hi"]:
            return b["right"]
    return None


def update_track_record(history: list[dict], close: pd.Series) -> list[dict]:
    idx = close.index
    for h in history:
        if h.get("scored"):
            continue
        d = pd.Timestamp(h["date"])
        if d not in idx:
            continue
        i = idx.get_loc(d)
        c0 = close.iloc[i]
        if h["call"] == "SELL TOMORROW" and i + 1 < len(idx):
            h["next_close"] = round(float(close.iloc[i + 1]), 2)
            h["right"] = bool(close.iloc[i + 1] < c0)
            h["scored"] = True
        elif h["call"] == "HOLD" and i + 1 < len(idx):
            h["next_close"] = round(float(close.iloc[i + 1]), 2)
            h["right"] = bool(close.iloc[i + 1] > c0)
            h["scored"] = True
        elif h["call"] == "NO CLEAR CALL" and i + 1 < len(idx):
            h["next_close"] = round(float(close.iloc[i + 1]), 2)
            h["scored"] = True
    return history


def notify(latest: dict):
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        return
    call = latest["call"]
    title = f"Gold: {latest['strength'] + ' ' if latest['strength'] else ''}{call.title()}"
    mm = latest.get("mastermelt")
    if mm and mm.get("prices"):
        p = mm["prices"]
        price_line = f"Mastermelt 9ct £{p.get('9ct', 0):.2f}/g, 18ct £{p.get('18ct', 0):.2f}/g. "
    else:
        price_line = f"£{latest['price']['per_oz']:,.0f}/oz · £{latest['price']['per_gram']:.2f}/g. "
    body = price_line + (latest["reasons"][0] if latest["reasons"] else "")
    tags = {"HOLD": "chart_with_upwards_trend", "SELL TOMORROW": "moneybag"}.get(call, "zzz")
    headers = {"Title": title, "Tags": tags, "Priority": "4" if latest["strength"] == "Strong" else "3"}
    if os.environ.get("APP_URL"):
        headers["Click"] = os.environ["APP_URL"]
    try:
        requests.post(f"https://ntfy.sh/{topic}", data=body.encode("utf-8"), headers=headers, timeout=20)
    except requests.RequestException:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-notify", action="store_true")
    args = ap.parse_args()

    series = synthetic_series() if args.selftest else D.load_all("2000-01-01")
    f = build_features(series).iloc[200:]
    t = build_targets(f["close"], horizons=(1, 5)).loc[f.index]
    cols = feature_columns(f)

    p1, contrib1, _ = fit_and_predict(f, cols, t, 1)
    p5, contrib5, _ = fit_and_predict(f, cols, t, 5)
    call, strength = verdict(p1, p5)

    last = f.iloc[-1]
    close = float(last["close"])
    g = series["gold_gbp"].loc[: f.index[-1]]
    hi, lo = (g["high"] if "high" in g else g["close"]), (g["low"] if "low" in g else g["close"])
    atr = float(last["atr14_pct"]) * close if not np.isnan(last["atr14_pct"]) else float(last["vol_20"]) * close

    # What moved the UK price today: dollar gold vs the pound
    move = []
    if "usd_gold_ret_1" in f and "gbpusd_ret_1" in f and not np.isnan(last.get("usd_gold_ret_1", np.nan)):
        ug, fx = last["usd_gold_ret_1"] * 100, last["gbpusd_ret_1"] * 100
        move.append(f"Gold {'rose' if ug >= 0 else 'fell'} {abs(ug):.1f}% in dollars and the pound "
                    f"{'strengthened' if fx >= 0 else 'weakened'} {abs(fx):.1f}%, so in pounds gold "
                    f"{'rose' if last['ret_1'] >= 0 else 'fell'} {abs(last['ret_1'] * 100):.1f}%.")

    reasons = []
    for feat, c in contrib1[:3]:
        reasons.append(f"{describe(feat, float(last[feat]))}, pointing {'up' if c > 0 else 'down'}")

    # Mastermelt: their trade counter price per gram, and what tomorrow's likely
    # range would mean for it (their price follows spot, set once a day)
    mm = None
    if not args.selftest:
        MM.main()
    mm_path = OUT / "mastermelt.json"
    if mm_path.exists():
        raw = json.loads(mm_path.read_text())
        spot_g = close / GRAMS_PER_OZ
        if raw.get("prices"):
            lo_f, hi_f = (close - atr) / close, (close + atr) / close
            mm = {"date": raw["date"], "ok": raw.get("ok", False),
                  "pct_of_spot": round(raw["pure_g"] / spot_g * 100, 1),
                  "prices": raw["prices"],
                  "tomorrow": {k: [round(v * lo_f, 2), round(v * hi_f, 2)] for k, v in raw["prices"].items()}}

    if not reasons:
        reasons.append("None of the tested indicators is giving a proven edge today, so there's nothing to act on")

    bt_path = OUT / "backtest.json"
    bt = json.loads(bt_path.read_text()) if bt_path.exists() else None

    news, events = None, []
    if not args.selftest:
        heads = gold_headlines()
        news = read_news(heads, move[0] if move else f"Gold closed at £{close:,.0f}/oz.")
        if news is not None:
            news["headlines"] = heads[:6]
        events = big_events()

    latest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "price_date": f.index[-1].date().isoformat(),
        "selftest": args.selftest,
        "call": call,
        "strength": strength,
        "prob_up_tomorrow": round(p1, 4),
        "prob_up_5d": round(p5, 4),
        "backtest_right_tomorrow": band_hit_rate(bt, p1, "h1"),
        "backtest_right_5d": band_hit_rate(bt, p5, "h5"),
        "price": {"per_oz": round(close, 2), "per_gram": round(close / GRAMS_PER_OZ, 2),
                  "change_pct": round(float(last["ret_1"]) * 100, 2)},
        "today": move,
        "reasons": reasons,
        "levels": {
            "expected_low": round(close - atr, 2), "expected_high": round(close + atr, 2),
            "high_20d": round(float(hi.iloc[-20:].max()), 2), "low_20d": round(float(lo.iloc[-20:].min()), 2),
            "sma50": round(float(g["close"].iloc[-50:].mean()), 2), "sma200": round(float(g["close"].iloc[-200:].mean()), 2),
            "high_52w": round(float(hi.iloc[-252:].max()), 2),
        },
        "rsi14": round(float(last["rsi14"]), 1),
        "mastermelt": mm,
        "news": news,
        "news_status": NEWS_STATUS,
        "events": events,
        "spark": [[d.date().isoformat(), round(float(v), 2)] for d, v in g["close"].iloc[-90:].items()],
    }
    if events:
        latest["reasons"].append(f"Big announcement due: {events[0]['country']} {events[0]['title']}. Expect a bigger swing either way")

    OUT.mkdir(parents=True, exist_ok=True)
    hist_path = OUT / "history.json"
    history = json.loads(hist_path.read_text()) if hist_path.exists() else []
    if not args.selftest:
        history = [h for h in history if h["date"] != latest["price_date"]]
        history.append({"date": latest["price_date"], "call": call, "strength": strength,
                        "close": round(close, 2), "p1": round(p1, 4), "p5": round(p5, 4)})
        history = update_track_record(history, series["gold_gbp"]["close"])
        hist_path.write_text(json.dumps(history[-400:], indent=1))
    (OUT / "latest.json").write_text(json.dumps(latest, indent=1))

    if not args.selftest and not args.no_notify:
        notify(latest)
    print(json.dumps({k: latest[k] for k in ["price_date", "call", "strength", "prob_up_tomorrow", "prob_up_5d", "reasons"]}, indent=1))


if __name__ == "__main__":
    main()
