"""Context the backtest can't score: today's gold news (read by OpenAI) and the
economic calendar. Shown alongside the verdict, never mixed into the tested model."""
from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import requests

UA = {"User-Agent": "Mozilla/5.0 (gold-forecast)"}


def gold_headlines(limit: int = 20) -> list[dict]:
    url = "https://news.google.com/rss/search"
    params = {"q": "gold price OR gold prices OR bullion when:1d", "hl": "en-GB", "gl": "GB", "ceid": "GB:en"}
    try:
        r = requests.get(url, params=params, headers=UA, timeout=20)
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except (requests.RequestException, ET.ParseError) as e:
        print(f"News feed unavailable: {e}")
        return []
    out = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        src = item.find("source")
        out.append({"title": title, "source": src.text if src is not None else "", "link": item.findtext("link") or ""})
        if len(out) >= limit:
            break
    return out


def read_news(headlines: list[dict], price_note: str) -> dict | None:
    """Ask OpenAI for a plain-English read of the headlines for a UK gold dealer."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key or not headlines:
        print(f"News read skipped (key set: {bool(key)}, headlines: {len(headlines)})")
        return None
    model = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
    lines = "\n".join(f"- {h['title']} ({h['source']})" for h in headlines)
    prompt = (
        "You help a UK gold dealer decide whether to sell scrap gold tomorrow or hold it. "
        "Prices are in pounds, so both the dollar gold price and the pound/dollar rate matter.\n"
        f"Market note: {price_note}\n\nToday's gold headlines:\n{lines}\n\n"
        "Reply in JSON with keys: tone (one of 'supportive', 'neutral', 'negative' for the gold price), "
        "summary (two short plain sentences, no jargon, British English), "
        "drivers (list of up to 3 short phrases naming what's moving gold). "
        "Base it only on the headlines. If they're mixed or thin, say neutral."
    )
    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "response_format": {"type": "json_object"}},
            timeout=60,
        )
        r.raise_for_status()
        data = json.loads(r.json()["choices"][0]["message"]["content"])
        if data.get("tone") not in {"supportive", "neutral", "negative"}:
            data["tone"] = "neutral"
        return {"tone": data["tone"], "summary": str(data.get("summary", ""))[:400],
                "drivers": [str(d)[:80] for d in data.get("drivers", [])][:3]}
    except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as e:
        detail = getattr(getattr(e, "response", None), "text", "")[:300]
        print(f"OpenAI news read failed: {e} {detail}")
        return None


def big_events(hours_ahead: int = 36) -> list[dict]:
    """High-impact US and UK releases coming up (rates, inflation, jobs)."""
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", headers=UA, timeout=20)
        r.raise_for_status()
        rows = r.json()
    except (requests.RequestException, ValueError):
        return []
    now = datetime.now(timezone.utc)
    out = []
    for e in rows:
        if e.get("impact") != "High" or e.get("country") not in {"USD", "GBP"}:
            continue
        try:
            t = datetime.fromisoformat(e["date"]).astimezone(timezone.utc)
        except (KeyError, ValueError):
            continue
        if now <= t <= now + timedelta(hours=hours_ahead):
            out.append({"when_utc": t.isoformat(), "country": e["country"], "title": e.get("title", "")})
    return sorted(out, key=lambda x: x["when_utc"])
