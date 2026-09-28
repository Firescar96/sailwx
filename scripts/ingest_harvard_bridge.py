#!/usr/bin/env python3
"""Ingest SailFlow "Harvard Bridge" spot (#1834) -- a real WeatherFlow
Networks station on the Charles River, between MIT Sailing Pavilion and
CBI. ADDED 2026-09-27 (user: "start tracking the harvard bridge location
... to get wind speed, gust, etc. that's available for free").

SOURCE: https://www.sailflow.com/spot/1834 (SailFlow/WindAlert, both
owned by WeatherFlow -- see the same-day research conversation on
WindAlert's lack of a real developer API). No public REST API exists;
the free-tier "Current Conditions" widget on that page IS genuinely
public (no login required to view), same legitimacy basis as MIT's own
pixel-scraped graphs, which this project has scraped from day one.

MECHANISM (found via live investigation, several dead ends along the
way):
  - The page is a client-rendered SPA. A plain requests.get() sees only
    ~850 bytes of empty shell HTML -- the real data loads via a
    same-origin fetch() call to api.weatherflow.com, gated behind a
    `wf_token` that's minted server-side per PAGE LOAD and tied to
    cookies/session state. A bare `curl` (even with a freshly-copied
    wf_token AND cookies) gets "Invalid weatherflow token" / "Unknown
    Error" -- this genuinely requires a real browser context, not just
    header replication.
  - Endpoint: /wxengine/rest/graph/getGraph?...&format=json&type=dataonly
    returns wind_avg_data/wind_gust_data/wind_lull_data/wind_dir_data
    (and pressure_data when fields= includes "pressure") as
    [timestamp_ms, value] arrays at ~5-9min native resolution.
  - HARD LIMIT CONFIRMED: only time_end_offset_hours=0 (ending "now")
    works; any non-zero end offset (trying to page further into the
    past) returns a generic "Unknown Error", and time_start_offset_hours
    beyond -48 ALSO fails outright. So this endpoint only ever exposes a
    rolling 48-HOUR window -- no deep backfill is possible through this
    endpoint. (A separate getObservationSummary endpoint DOES accept
    arbitrary time_start_local/time_end_local for older dates, but only
    returns daily max/min SUMMARY stats, not a time series -- not useful
    for our per-reading ingestion model.)
  - air_temp_data/water_temp_data came back None for this station even
    when explicitly requested -- Harvard Bridge apparently only reports
    wind + pressure, no temperature sensor.

Because of the 48h-window-only limit, this ingester is designed to run
frequently enough that consecutive polls' 48h windows overlap
comfortably (matching the project's existing "rolling window,
self-healing on a missed run" pattern used by scripts/ingest_mit_all.py)
-- see the cron job for the actual schedule.
"""
import os
import re
import sys
from datetime import datetime, timezone

from playwright.sync_api import sync_playwright

PROJECT_DIR = os.path.expanduser("~/.hermes/projects/sailing-weather")
sys.path.insert(0, os.path.join(PROJECT_DIR, "db"))
from common import bulk_insert, get_connection  # noqa: E402

SPOT_URL = "https://www.sailflow.com/spot/1834"
SPOT_ID = "1834"
LOCATION_ID = "harvard_bridge"
MPH_TO_KT = 0.868976
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Variable name in the getGraph response -> our observations variable name.
# Wind values arrive already in mph (units_wind=mph param) -> convert to
# knots (canonical unit, confirmed 2026-09-09). Direction and pressure
# need no conversion (already deg / hPa-equivalent mb).
FIELD_MAP = {
    "wind_avg_data": ("wind_speed_kt", "mph_to_kt"),
    "wind_gust_data": ("wind_gust_kt", "mph_to_kt"),
    "wind_dir_data": ("wind_dir_deg", "none"),
    "pressure_data": ("pressure_hpa", "none"),
}


def fetch_graph_data():
    """Drives a real headless browser to Harvard Bridge's SailFlow page,
    mints a fresh wf_token the same way a real visitor's browser would,
    then calls the same-origin getGraph JSON endpoint from within that
    page's JS context (a bare requests/curl call cannot pass the
    token/cookie validation -- confirmed via direct testing). Returns the
    parsed dict of {variable_name: [[ts_ms, value], ...]}.
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
    real table -- same pattern as ingest_ndbc.py. Insert-if-new (not
    upsert): a rolling 48h re-poll of the SAME real reading should be
    identical, so ON CONFLICT DO NOTHING is correct here (unlike MIT's
    pixel-scrape, which genuinely needs upsert since a later poll's OCR
    can be a more reliable re-read of the SAME pixel column)."""
    if not rows:
        return 0
    con.execute(
        "CREATE OR REPLACE TEMP TABLE stage_harvard_bridge "
        "(location_id TEXT, ts_utc TIMESTAMP, variable TEXT, value DOUBLE)"
    )
    bulk_insert(con, "stage_harvard_bridge", ["location_id", "ts_utc", "variable", "value"], rows)
    before = con.execute(
        "SELECT count(*) FROM observations WHERE location_id = ?", [LOCATION_ID]
    ).fetchone()[0]
    con.execute(
        """
        INSERT INTO observations (location_id, ts_utc, variable, value, source)
        SELECT s.location_id, s.ts_utc, s.variable, ANY_VALUE(s.value), 'sailflow_harvard_bridge'
        FROM stage_harvard_bridge s
        ANTI JOIN observations o
            ON o.location_id = s.location_id AND o.ts_utc = s.ts_utc AND o.variable = s.variable
        GROUP BY s.location_id, s.ts_utc, s.variable
        """
    )
    con.execute("DROP TABLE stage_harvard_bridge")
    after = con.execute(
        "SELECT count(*) FROM observations WHERE location_id = ?", [LOCATION_ID]
    ).fetchone()[0]
    return after - before


def main():
    # FETCH-THEN-WRITE split (matches the project-wide concurrency fix
    # from 2026-09-27, "i need more concurrency and less db locks") --
    # the browser-driven fetch here is the slowest part of this whole
    # script (~10+ real seconds), so it must complete with NO DB
    # connection open at all.
    data = fetch_graph_data()
    rows = process_rows(data)

    con = get_connection()
    try:
        new_count = write_rows_to_db(con, rows)
    finally:
        con.close()

    print(f"Harvard Bridge (SailFlow #1834): fetched {len(rows)} readings in the "
          f"48h rolling window, {new_count} new rows inserted "
          f"(station last_ob_time_local={data.get('last_ob_time_local')})")


if __name__ == "__main__":
    sys.exit(main())
