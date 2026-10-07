#!/usr/bin/env python3
"""Ingest SailFlow "Courageous Sailing Center" spot (#1842) -- a real
WeatherFlow Networks station at Courageous Sailing's Pier 4 location on
Boston Harbor (Charlestown Navy Yard). ADDED 2026-10-06 (user: "I think
courageous sailing is on sailflow, can they be added as data source" ->
"yes, add the new location").

SOURCE: https://www.sailflow.com/spot/1842. Found via Courageous
Sailing's own public "Whiteboard" conditions page
(https://courageoussailing.org/whiteboard/), which embeds this exact
spot_id in several live SailFlow/WindAlert widget iframes (confirmed
2026-10-06 by inspecting that page's embedded iframe src attributes --
e.g. widgets.sailflow.com/widgets/web/modelTable?spot_id=1842&...&name=
Courageous%20Sailing%20Center). Same legitimacy basis as Harvard
Bridge's spot #1834 (see ingest_harvard_bridge.py) -- a free-tier
"Current Conditions" page requiring no login to view.

IMPORTANT: this is a DIFFERENT BODY OF WATER from every other location
in this project. MIT Pavilion/Harvard Bridge/CBI are all on the Charles
River basin; this station is on BOSTON HARBOR at Pier 4, Charlestown
Navy Yard (confirmed via Nominatim geocoding: 42.3715, -71.0509). It is
NOT a ground-truth substitute for any existing location -- it's its own
genuinely distinct spot, same as NDBC 44013 or KBOS are distinct from
the Charles River cluster.

MECHANISM: identical to ingest_harvard_bridge.py (same SailFlow/
WeatherFlow backend, same wf_token-gated getGraph endpoint, same
confirmed 48-hour rolling window hard limit -- no deeper history is
available through this endpoint; see that script's docstring for the
full investigation of why a plain requests/curl call cannot work here
and a real browser context is required). This script intentionally
mirrors that one's structure closely rather than introducing a new
pattern, to keep the SailFlow-ingestion approach consistent across both
spots.

CONFIRMED LIVE 2026-10-06: wind_avg_data/wind_gust_data/wind_dir_data/
pressure_data all populated with real recent readings (last_ob_time_local
matched the live site at time of testing); air_temp_data came back None
for this station too, same as Harvard Bridge -- apparently neither
Charles-river-area SailFlow spot reports temperature.
"""
import os
import re
import sys
from datetime import datetime, timezone

from playwright.sync_api import sync_playwright

PROJECT_DIR = os.path.expanduser("~/.hermes/projects/sailing-weather")
sys.path.insert(0, os.path.join(PROJECT_DIR, "db"))
from common import bulk_insert, get_connection  # noqa: E402

SPOT_URL = "https://www.sailflow.com/spot/1842"
SPOT_ID = "1842"
LOCATION_ID = "courageous_sailing"
MPH_TO_KT = 0.868976
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Variable name in the getGraph response -> our observations variable name.
# Wind values arrive already in mph (units_wind=mph param) -> convert to
# knots (canonical unit). Direction and pressure need no conversion
# (already deg / hPa-equivalent mb).
FIELD_MAP = {
    "wind_avg_data": ("wind_speed_kt", "mph_to_kt"),
    "wind_gust_data": ("wind_gust_kt", "mph_to_kt"),
    "wind_dir_data": ("wind_dir_deg", "none"),
    "pressure_data": ("pressure_hpa", "none"),
}


def fetch_graph_data():
    """Drives a real headless browser to Courageous Sailing's SailFlow
    page, mints a fresh wf_token the same way a real visitor's browser
    would, then calls the same-origin getGraph JSON endpoint from
    within that page's JS context (a bare requests/curl call cannot
    pass the token/cookie validation -- confirmed via direct testing,
    same as Harvard Bridge). Returns the parsed dict of
    {variable_name: [[ts_ms, value], ...]}.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=USER_AGENT)
        page.goto(SPOT_URL, timeout=45000, wait_until="load")
        page.wait_for_timeout(6000)

        token = page.evaluate(
            """
            () => {
                const m = document.cookie.match(/(^| )wfToken=([^;]+)/);
                return m ? decodeURIComponent(m[2]) : null;
            }
            """
        )
        if not token:
            browser.close()
            raise RuntimeError("could not find wfToken cookie -- page structure may have changed")

        result_text = page.evaluate(
            """
            async ([token, spotId]) => {
                const url = `https://api.weatherflow.com/wxengine/rest/graph/getGraph`
                    + `?units_wind=mph&units_temp=f&units_distance=mi&units_precip=in`
                    + `&show_nc_wind_toggle=false&units_height=ft`
                    + `&fields=wind,pressure&format=json&null_ob_min_from_now=30`
                    + `&show_virtual_obs=true&spot_id=${spotId}`
                    + `&time_start_offset_hours=-48&time_end_offset_hours=0`
                    + `&type=dataonly&model_ids=-101&wf_token=${token}`;
                const resp = await fetch(url);
                return await resp.text();
            }
            """,
            [token, SPOT_ID],
        )
        browser.close()

    import json
    data = json.loads(result_text)
    status = data.get("status", {})
    if status.get("status_code") != 0:
        raise RuntimeError(f"getGraph returned non-success status: {status}")
    return data


def process_rows(data):
    """Returns a list of (location_id, ts_utc, variable, value) rows,
    deduped/ready for bulk_insert. Skips null values (station gaps)."""
    rows = []
    for field_key, (our_var, conversion) in FIELD_MAP.items():
        series = data.get(field_key)
        if not series:
            continue
        for ts_ms, value in series:
            if value is None:
                continue
            ts = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
            if conversion == "mph_to_kt":
                value = round(value * MPH_TO_KT, 1)
            else:
                value = round(float(value), 2)
            rows.append((LOCATION_ID, ts, our_var, value))
    return rows


def write_rows_to_db(con, rows):
    """Bulk-load into an unconstrained staging table first (fast, no
    PK/FK checks), then a single set-based anti-join insert into the
    real table -- same pattern as ingest_harvard_bridge.py/ingest_ndbc.py.
    Insert-if-new (not upsert): a rolling 48h re-poll of the SAME real
    reading should be identical, so ON CONFLICT DO NOTHING is correct
    here."""
    if not rows:
        return 0
    con.execute(
        "CREATE OR REPLACE TEMP TABLE stage_courageous_sailing "
        "(location_id TEXT, ts_utc TIMESTAMP, variable TEXT, value DOUBLE)"
    )
    bulk_insert(con, "stage_courageous_sailing", ["location_id", "ts_utc", "variable", "value"], rows)
    before = con.execute(
        "SELECT count(*) FROM observations WHERE location_id = ?", [LOCATION_ID]
    ).fetchone()[0]
    con.execute(
        """
        INSERT INTO observations (location_id, ts_utc, variable, value, source)
        SELECT s.location_id, s.ts_utc, s.variable, ANY_VALUE(s.value), 'sailflow_courageous_sailing'
        FROM stage_courageous_sailing s
        ANTI JOIN observations o
            ON o.location_id = s.location_id AND o.ts_utc = s.ts_utc AND o.variable = s.variable
        GROUP BY s.location_id, s.ts_utc, s.variable
        """
    )
    con.execute("DROP TABLE stage_courageous_sailing")
    after = con.execute(
        "SELECT count(*) FROM observations WHERE location_id = ?", [LOCATION_ID]
    ).fetchone()[0]
    return after - before


def main():
    # FETCH-THEN-WRITE split (same project-wide concurrency pattern as
    # ingest_harvard_bridge.py) -- the browser-driven fetch is the
    # slowest part of this script, so it must complete with NO DB
    # connection open at all.
    data = fetch_graph_data()
    rows = process_rows(data)

    con = get_connection()
    try:
        new_count = write_rows_to_db(con, rows)
    finally:
        con.close()

    print(f"Courageous Sailing Center (SailFlow #1842): fetched {len(rows)} readings in the "
          f"48h rolling window, {new_count} new rows inserted "
          f"(station last_ob_time_local={data.get('last_ob_time_local')})")


if __name__ == "__main__":
    sys.exit(main())
