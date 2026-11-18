# Gold Forecast

Personal daily forecast for UK gold (pounds per troy ounce). One question, answered every evening: sell tomorrow, or hold.

Completely separate from Gold Trader. Purchases are still logged in Gold Trader; this app only gives the signal.

## Stage one: what this repo does now

Loads 20+ years of daily gold in GBP, plus the main drivers (gold in dollars, pound vs dollar, dollar index, US 10-year yields, silver). It then tests everything three ways:

1. **Classic trader signals one by one.** These are RSI, moving averages, golden cross, MACD, Bollinger bands, stochastics, breakouts, streaks, day of week and the driver signals. History is split in half, and a signal only passes if it beat chance in both halves.
2. **Every raw indicator's link to the next day's move**, again in both halves.
3. **A walk-forward model.** Each year, it picks only the indicators that held up in earlier data and trains on earlier years. It's then scored on a year it has never seen. When nothing is proven, it says "no clear call" and doesn't guess.

The output is `reports/stage1_report.html`, written in plain English.

## Run it

```bash
pip install -r requirements.txt
python run_stage1.py              # real data
python run_stage1.py --selftest   # random-walk data, must show NO edge
```

Real data comes from stooq.com and falls back to Yahoo Finance. Neither needs a key. To run offline, put `gold_gbp.csv` (date, open, high, low, close) in `data/`.

## Safety checks built in

The self-test feeds the engine pure random prices. A sound engine can't find an edge there. Current result: 0 of 22 signals pass, accuracy sits at the base rate, and the model says "no clear call" on 92% of days.

A second check plants a small real pattern in random data, and the engine finds it. That proves it can see a genuine edge when one exists.

## Next stages

2. The evening job runs on Google Cloud (Cloud Run job, europe-west2). It runs after the London close, with OpenAI reading the day's gold news and the economic calendar.
3. The phone app (installable, Android first) shows the evening verdict, confidence, levels to watch, reasons and the live track record, with a notification each evening.
