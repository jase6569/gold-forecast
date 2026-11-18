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

## How it runs

GitHub Actions does everything on GitHub's servers:

`forecast.yml` runs at 22:15 UTC, Monday to Friday. It pulls prices, works out tomorrow's verdict, has OpenAI read the day's gold news, checks the calendar for big US and UK announcements, scores past calls, saves to `web/data`, and sends a phone notification through ntfy. `backtest.yml` runs every Sunday. It runs the self-test, then re-runs the full backtest so the model's track record stays current.

The phone app is the static site in `web/`, hosted on Vercel. It redeploys automatically each time the bot saves a new forecast. To install it, open it in Chrome on Android, then tap the menu and choose "Add to Home screen".

## Settings (GitHub repo, Settings, Secrets and variables, Actions)

| Name | Type | Value |
|---|---|---|
| OPENAI_API_KEY | Secret | Your OpenAI key |
| NTFY_TOPIC | Secret | Private notification channel name (subscribe to the same name in the ntfy app) |
| APP_URL | Variable | Your Vercel address, so tapping the notification opens the app |
| OPENAI_MODEL | Variable (optional) | Defaults to gpt-5-mini |

## Giving Mike his own copy later

Mike gets his own repo copy with his own NTFY_TOPIC and his own Vercel site. His alerts never reach you, and yours never reach him.
