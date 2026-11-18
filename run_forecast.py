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
from engine import ranges as RG
from engine import widget as WG
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


def verdict(p1: float, bt: dict | None, window: str = "5") -> tuple[str, str]:
    """LOCK IN = fix the price with your dealer now. WAIT = tomorrow looks better.
    Firm calls are the 10% most confident nights in the backtest; the lean lock-in
    threshold is the one the sell-window test picked on years it then proved on."""
    firm = (bt or {}).get("firm") or {}
    rule = (((bt or {}).get("strategy") or {}).get(window) or {}).get("rule") or {}
    hi, lo = firm.get("current_hi", 0.56), firm.get("current_lo", 0.46)
    if p1 <= lo:
        return "LOCK IN", "Strong"
    if p1 >= hi:
        return "WAIT", "Strong"
    t = rule.get("lock_if_up_chance_below")
    if t is not None and p1 < t:
        return "LOCK IN", "Lean"
    return "NO STRONG VIEW", ""


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
        if i + 1 < len(idx):
            h["next_ret"] = round(float(np.log(close.iloc[i + 1] / c0)), 6)
        if h["call"] in ("SELL TOMORROW", "LOCK IN") and i + 1 < len(idx):
            h["next_close"] = round(float(close.iloc[i + 1]), 2)
            h["right"] = bool(close.iloc[i + 1] < c0)
            h["scored"] = True
        elif h["call"] in ("HOLD", "WAIT") and i + 1 < len(idx):
            h["next_close"] = round(float(close.iloc[i + 1]), 2)
            h["right"] = bool(close.iloc[i + 1] > c0)
            h["scored"] = True
        elif h["call"] in ("NO CLEAR CALL", "NO STRONG VIEW") and i + 1 < len(idx):
            h["next_close"] = round(float(close.iloc[i + 1]), 2)
            h["scored"] = True
    return history


def notify(latest: dict):
    # One private channel per person (NTFY_TOPIC for Jase, NTFY_TOPIC_MIKE for Mike)
    topics = [t for t in (os.environ.get("NTFY_TOPIC"), os.environ.get("NTFY_TOPIC_MIKE")) if t]
    if not topics:
        return
    call = latest["call"]
    title = {"LOCK IN": "Gold: lock in your price", "WAIT": "Gold: wait, tomorrow looks better"}.get(call, "Gold: no strong view tonight")
    if latest["strength"] == "Strong" and call != "NO STRONG VIEW":
        title += " (firm call)"
    mm = latest.get("mastermelt")
    if mm and mm.get("prices"):
        p = mm["prices"]
        price_line = f"Mastermelt 9ct £{p.get('9ct', 0):.2f}/g, 18ct £{p.get('18ct', 0):.2f}/g. "
    else:
        price_line = f"24ct £{latest['price']['per_gram']:.2f}/g. "
    if mm and mm.get("tomorrow", {}).get("9ct"):
        lo9, hi9 = mm["tomorrow"]["9ct"]
        price_line += f"Tomorrow 9ct likely £{lo9:.2f} to £{hi9:.2f}. "
    if call == "NO STRONG VIEW":
        body = price_line + "No action needed tonight; follow your batch plan."
    else:
        body = price_line + (latest["reasons"][0] if latest["reasons"] else "")
    tags = {"WAIT": "chart_with_upwards_trend", "LOCK IN": "lock"}.get(call, "zzz")
    headers = {"Title": title, "Tags": tags, "Priority": "4" if latest["strength"] == "Strong" else "3"}
    if os.environ.get("APP_URL"):
        headers["Click"] = os.environ["APP_URL"]
    for topic in topics:
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
    bt_path = OUT / "backtest.json"
    bt = json.loads(bt_path.read_text()) if bt_path.exists() else None
    call, strength = verdict(p1, bt)

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

    rng = RG.latest(g["close"])
    gram = lambda v: round(float(v) / GRAMS_PER_OZ, 2)
    strat5 = ((bt or {}).get("strategy") or {}).get("5") or {}
    plan = {"window_days": 5, "rule": strat5.get("rule"),
            "rule_per_10k": (strat5.get("rule_unseen") or {}).get("per_10k"),
            "wait_per_10k": (strat5.get("sell_at_end_unseen") or {}).get("per_10k"),
            "beats_sell_now": (strat5.get("rule_unseen") or {}).get("beats_sell_now"),
            "firm": (bt or {}).get("firm")}

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
            lo_f, hi_f = rng["h1"]["lo"] / close, rng["h1"]["hi"] / close
            mm = {"date": raw["date"], "ok": raw.get("ok", False),
                  "pct_of_spot": round(raw["pure_g"] / spot_g * 100, 1),
                  "prices": raw["prices"],
                  "tomorrow": {k: [round(v * lo_f, 2), round(v * hi_f, 2)] for k, v in raw["prices"].items()}}

    if not reasons:
        reasons.append("None of the tested indicators is giving a proven edge today, so there's nothing to act on")

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
        # Everything below is £ per gram of pure (24ct) gold
        "levels": {
            "high_20d": gram(hi.iloc[-20:].max()), "low_20d": gram(lo.iloc[-20:].min()),
            "sma50": gram(g["close"].iloc[-50:].mean()), "sma200": gram(g["close"].iloc[-200:].mean()),
            "high_52w": gram(hi.iloc[-252:].max()),
        },
        "ranges": {k: {"lo": gram(v["lo"]), "hi": gram(v["hi"])} for k, v in rng.items()},
        "ranges_hit": {k: v["hit"] for k, v in ((bt or {}).get("ranges") or {}).items()},
        "atr_pct": round(atr / close, 5),
        "plan": plan,
        "rsi14": round(float(last["rsi14"]), 1),
        "mastermelt": mm,
        "news": news,
        "news_status": NEWS_STATUS,
        "events": events,
        "spark": [[d.date().isoformat(), gram(v)] for d, v in g["close"].iloc[-90:].items()],
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

    if not args.selftest:
        WG.main()
    if not args.selftest and not args.no_notify:
        notify(latest)
    print(json.dumps({k: latest[k] for k in ["price_date", "call", "strength", "prob_up_tomorrow", "prob_up_5d", "reasons"]}, indent=1))


if __name__ == "__main__":
    main()
