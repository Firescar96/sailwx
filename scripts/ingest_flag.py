#!/usr/bin/env python3
"""Ingest Community Boating Inc. (CBI) dockhouse flag color.

CBI uses a color-coded flag (Green/Yellow/Red/Closed) set by the
dockmaster to indicate current sailing conditions on the Charles River
basin -- a categorical human judgment call, not a raw sensor reading, but
one directly informed by wind speed/gusts. Confirmed live 2026-09-08 via
the public API backing the widget on
https://www.community-boating.org/about-us/weather-information/ :

    GET https://api.community-boating.org/api/flag
    -> var FLAG_COLOR = "Y"

(a JS-snippet response, not JSON -- parsed with a small regex below).
Observed codes: G=green, Y=yellow, R=red, C=closed (confirmed live
2026-09-10 -- originally guessed "B" for black/closed based on CBI's own
flag-policy page description before ever observing a closed reading live;
the real code turned out to be "C", not "B").

Flag changes are event-driven (a human updates it), NOT scheduled like a
model run, so this is checked periodically to catch changes with
reasonable timeliness. Cadence history: started at every 30 min, dialed
back to every 4 hours (6x/day) on 2026-09-08 per user request to avoid
overloading the endpoint, then increased to hourly on 2026-09-10 per
user request -- now the flag-band chart overlay has much finer time
resolution to show exactly when conditions changed. Every poll's result
is stored, whether or not the color changed since the last poll --
unlike the observation ingesters, this is NOT deduplicated to
changes-only, because it also lets you infer "flag was still X as of this
timestamp" for gap-free correlation against wind data later (Task:
probability-of-flag-color-given-wind analysis).
"""
import re
import ssl
import sys
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, "/home/firescar96/.hermes/projects/sailing-weather/db")
from common import get_connection

sys.path.insert(0, "/home/firescar96/.hermes/projects/sailing-weather/scripts")
from cbi_hours import is_cbi_open

URL = "https://api.community-boating.org/api/flag"
LOCATION_ID = "cbi_dockhouse"

CODE_TO_COLOR = {
    "G": "green",
    "Y": "yellow",
    "R": "red",
    "C": "closed",
    "B": "closed",  # defensive fallback in case CBI's API ever uses this too; unconfirmed
}

ctx = ssl.create_default_context()


def fetch_flag_code():
    req = urllib.request.Request(URL, headers={"User-Agent": "sailing-weather-archiver/1.0"})
    with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    m = re.search(r'FLAG_COLOR\s*=\s*"([A-Za-z]+)"', text)
    if not m:
        raise ValueError(f"Could not parse FLAG_COLOR from response: {text!r}")
    return m.group(1).upper()


def main():
    code = fetch_flag_code()
    color = CODE_TO_COLOR.get(code, code.lower())  # fall back to raw code if unmapped
    ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    # CBI's Adult Program only operates 9am-sunset (per user 2026-09-11).
    # The live API is just a JS variable on their site -- it can return a
    # stale/leftover color after hours rather than reliably resetting to
    # "closed" the moment they close for the day. Force closed outside
    # the real operating window so the recorded flag history (and
    # anything downstream, e.g. the Bayesian flag-prediction panel)
    # reflects CBI's actual hours rather than whatever the API happened
    # to still be showing.
    if not is_cbi_open(ts):
        if color != "closed":
            print(f"CBI flag: API returned {color!r} outside operating hours (9am-sunset) -- forcing to 'closed'")
        color = "closed"

    con = get_connection()
    try:
        con.execute(
            """
            INSERT INTO flags (location_id, ts_utc, flag_color, source)
            VALUES (?, ?, ?, 'cbi_flag_api')
            ON CONFLICT DO NOTHING
            """,
            [LOCATION_ID, ts, color],
        )
    finally:
        con.close()
    print(f"CBI flag: {color} (raw code {code!r}) recorded at {ts}")


if __name__ == "__main__":
    sys.exit(main())
