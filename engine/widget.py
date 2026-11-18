"""Write web/data/widget.json for the Android home-screen widget, from the
latest forecast and Mastermelt's newest counter price."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

DATA = Path(__file__).resolve().parent.parent / "web" / "data"
CALL_LINES = {"LOCK IN": "Lock in your price", "WAIT": "Wait, tomorrow looks better",
              "NO STRONG VIEW": "No strong view tonight"}


def main():
    try:
        L = json.loads((DATA / "latest.json").read_text())
    except (OSError, ValueError):
        return
    mm_file = DATA / "mastermelt.json"
    mm = json.loads(mm_file.read_text()) if mm_file.exists() else None
    prices = (mm or {}).get("prices") or (L.get("mastermelt") or {}).get("prices")
    spot_g = L["price"]["per_gram"]
    call = L.get("call", "")
    call_line = CALL_LINES.get(call, call.title())
    if L.get("strength") == "Strong" and call in ("LOCK IN", "WAIT"):
        call_line += " (firm)"
    if prices:
        f = prices["24ct"] / spot_g
        r = L.get("ranges", {}).get("h1")
        price_line = f"£{prices['24ct']:.2f}/g"
        detail = f"9ct £{prices['9ct']:.2f} · 18ct £{prices['18ct']:.2f}"
        if r:
            nine = prices["9ct"] / prices["24ct"]
            detail += f" · tmrw 9ct £{r['lo'] * f * nine:.2f}–{r['hi'] * f * nine:.2f}"
        label = "MASTERMELT 24CT"
    else:
        price_line, detail, label = f"£{spot_g:.2f}/g", "", "GOLD 24CT"
    gen = datetime.fromisoformat(L["generated_utc"].replace("Z", "+00:00"))
    out = {
        "generated_utc": L["generated_utc"][:16],
        "call": call, "call_line": call_line, "label": label,
        "price_line": price_line, "detail_line": detail,
        "updated_line": "Forecast " + gen.astimezone(ZoneInfo("Europe/London")).strftime("%a %H:%M"),
    }
    (DATA / "widget.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()
